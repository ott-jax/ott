# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ott import utils

if TYPE_CHECKING:
  from ott.problems.linear import linear_problem
  from ott.solvers.linear import sinkhorn

__all__ = ["AndersonAcceleration", "Momentum"]


@utils.register_pytree_node
class AndersonAcceleration:
  """Implements Anderson acceleration for Sinkhorn."""

  # TODO(michalk8): use memory=0 as no Anderson acceleration?
  memory: int = 2  # Number of iterates considered to form interpolation.
  refresh_every: int = 1  # Recompute interpolation periodically.
  ridge_identity: float = 1e-2  # Ridge used in the linear system.

  def extrapolation(self, xs: jnp.ndarray, fxs: jnp.ndarray) -> jnp.ndarray:
    """Compute Anderson extrapolation from past observations."""
    # Remove -inf values to instantiate quadratic problem. All others
    # remain since they might be caused by a valid issue.
    fxs_clean = jnp.nan_to_num(fxs, nan=jnp.nan, posinf=jnp.inf, neginf=0.0)
    xs_clean = jnp.nan_to_num(xs, nan=jnp.nan, posinf=jnp.inf, neginf=0.0)
    residuals = fxs_clean - xs_clean
    gram_matrix = jnp.matmul(residuals.T, residuals)
    gram_matrix /= jnp.linalg.norm(gram_matrix)

    # Solve linear system to obtain weights
    weights = jax.scipy.sparse.linalg.cg(
        gram_matrix + self.ridge_identity * jnp.eye(xs.shape[1]),
        jnp.ones(xs.shape[1])
    )[0]
    weights /= jnp.sum(weights)

    # Recover linear combination and return it with NaN (caused
    # by 0 weights leading to -jnp.inf potentials, mixed with weights
    # coefficients of different signs), disambiguated to -inf.
    combination = jnp.sum(fxs * weights[None, :], axis=1)
    return jnp.where(jnp.isfinite(combination), combination, -jnp.inf)

  def update(
      self, state: "sinkhorn.SinkhornState", iteration: int,
      prob: "linear_problem.LinearProblem", lse_mode: bool
  ) -> "sinkhorn.SinkhornState":
    """Anderson acceleration update.

    When using Anderson acceleration, first update the dual variable f_u with
    previous updates (if iteration count sufficiently large), then record
    new iterations in array.

    Anderson acceleration always happens in potentials (not scalings) space,
    regardless of the lse_mode setting. If the iteration count is large
    enough the update below will output a potential variable.

    Args:
      state: Sinkhorn state.
      iteration: the current iteration.
      prob: linear OT problem.
      lse_mode: whether to compute in log-sum-exp or in scalings.

    Returns:
      A potential variable.
    """
    geom = prob.geom
    trigger_update = jnp.logical_and(
        iteration > self.memory, iteration % self.refresh_every == 0
    )
    fu = jnp.where(
        trigger_update, self.extrapolation(state.old_fus, state.old_mapped_fus),
        state.fu
    )
    # If the interpolation was triggered, we store it in memory
    # Otherwise we add the previous value (converting it to potential form if
    # it was initially stored in scaling form).
    old_fus = jnp.where(
        trigger_update,
        jnp.concatenate((state.old_fus[:, 1:], fu[:, None]), axis=1),
        jnp.concatenate((
            state.old_fus[:, 1:],
            (fu if lse_mode else geom.potential_from_scaling(fu))[:, None]
        ),
                        axis=1)
    )

    # If update was triggered, ensure a scaling is returned, since the result
    # from the extrapolation was outputted in potential form.
    fu = jnp.where(
        trigger_update, fu if lse_mode else geom.scaling_from_potential(fu), fu
    )
    return state.set(potentials=(fu, state.gv), old_fus=old_fus)

  def init_maps(
      self, pb, state: "sinkhorn.SinkhornState"
  ) -> "sinkhorn.SinkhornState":
    """Initialize log matrix used in Anderson acceleration with *NaN* values."""
    fus = jnp.ones((pb.geom.shape[0], self.memory)) * jnp.nan
    return state.set(old_fus=fus, old_mapped_fus=fus)

  def update_history(
      self, state: "sinkhorn.SinkhornState", pb, lse_mode: bool
  ) -> "sinkhorn.SinkhornState":
    """Update history of mapped dual variables."""
    f = state.fu if lse_mode else pb.geom.potential_from_scaling(state.fu)
    mapped = jnp.concatenate((state.old_mapped_fus[:, 1:], f[:, None]), axis=1)
    return state.set(old_mapped_fus=mapped)


@utils.register_pytree_node
class Momentum:
  """Momentum for Sinkhorn updates.

  Can be either constant :cite:`thibault:21` or adaptive :cite:`lehmann:21`.
  """

  start: int = 0
  error_threshold: float = jnp.inf
  value: float = 1.0
  inner_iterations: int = 1

  def weight(self, state: "sinkhorn.SinkhornState", iteration: int) -> float:
    """Compute momentum term if needed, using previously seen errors."""
    if self.start == 0:
      return self.value
    idx = self.start // self.inner_iterations

    return jax.lax.cond(
        jnp.logical_and(
            iteration >= self.start, state.errors[idx - 1, -1]
            < self.error_threshold
        ), lambda state: self.lehmann(state), lambda state: self.value, state
    )

  def lehmann(self, state: "sinkhorn.SinkhornState") -> float:
    """Momentum formula :cite:`lehmann:21`, eq. 5."""
    idx = self.start // self.inner_iterations
    error_ratio = jnp.minimum(
        state.errors[idx - 1, -1] / state.errors[idx - 2, -1], 0.99
    )
    power = 1.0 / self.inner_iterations
    return 2.0 / (1.0 + jnp.sqrt(1.0 - error_ratio ** power))

  def __call__(  # noqa: D102
      self,
      weight: float,
      value: jnp.ndarray,
      new_value: jnp.ndarray,
      lse_mode: bool = True
  ) -> jnp.ndarray:
    if lse_mode:
      value = jnp.where(jnp.isfinite(value), value, 0.0)
      return (1.0 - weight) * value + weight * new_value
    value = jnp.where(value > 0.0, value, 1.0)
    return value ** (1.0 - weight) * new_value ** weight
