# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for Anderson acceleration."""
import jax
import jax.numpy as jnp

from ott.core import dataclasses


@dataclasses.register_pytree_node
class AndersonAcceleration:
  """Implements Anderson acceleration for Sinkhorn."""

  memory: int = 2  # Number of iterates considered to form interpolation.
  refresh_every: int = 1  # Recompute interpolation periodically.
  ridge_identity: float = 1e-2  # Ridge used in the linear system.

  def extrapolation(self, xs, fxs):
    """Computes Anderson extrapolation from past observations."""
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
        jnp.ones(xs.shape[1]))[0]
    weights /= jnp.sum(weights)

    # Recover linear combination and return it with NaN (caused
    # by 0 weights leading to -jnp.inf potentials, mixed with weights
    # coefficiences of different signs), disambiguated to -inf.
    combination = jnp.sum(fxs * weights[None, :], axis=1)
    return jnp.where(jnp.isfinite(combination), combination, -jnp.inf)

  def update(self, pb, lse_mode, iteration, f_u, old_f_u_s, old_mapped_f_u_s):
    """Anderson acceleration update.

    When using Anderson acceleration, first update the dual variable f_u with
    previous updates (if iteration count sufficiently large), then record
    new iterations in array.

    Anderson acceleration always happens in potentials (not scalings) space,
    regardless of the lse_mode setting. If the iteration count is large
    enough the update below will output a potential variable.

    Args:
     pb: a problem.LinearProblem defining the OT problem.
     lse_mode: whether to compute in log-sum-exp or in scalings.
     iteration: current iteration
     f_u: current potential/scaling
     old_f_u_s: previous updates
     old_mapped_f_u_s: previous updates.

    Returns:
      A potential variable.
    """
    geom = pb.geom
    trigger_update = jnp.logical_and(
        iteration > self.memory, iteration % self.refresh_every == 0)

    f_u = jnp.where(
        trigger_update, self.extrapolation(old_f_u_s, old_mapped_f_u_s), f_u)
    # If the interpolation was triggered, we store it in memory
    # Otherwise we add the previous value (converting it to potential form if
    # it was initially stored in scaling form).
    old_f_u_s = jnp.where(
        trigger_update,
        jnp.concatenate((old_f_u_s[:, 1:], f_u[:, None]), axis=1),
        jnp.concatenate(
            (old_f_u_s[:, 1:],
             (f_u if lse_mode
              else geom.potential_from_scaling(f_u))[:, None]),
            axis=1))

    # If update was triggered, ensure a scaling is returned, since the result
    # from the extrapolation was outputted in potential form.
    f_u = jnp.where(
        trigger_update,
        f_u if lse_mode else geom.scaling_from_potential(f_u),
        f_u)
    return f_u, old_f_u_s

  def init_maps(self, pb):
    fus = jnp.ones((pb.geom.shape[0], self.memory)) * jnp.nan
    return fus, fus
