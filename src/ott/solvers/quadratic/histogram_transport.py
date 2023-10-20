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
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np

from ott import utils
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.tools import soft_sort

__all__ = ["HistogramTransport", "HTState"]

LinearOutput = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput]

ProgressCallbackFn_t = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "HTState"]], None]


class HTOutput(NamedTuple):
  """Holds the output of the Histogram Transport solver.

  Args:
  costs:
  linear_convergence:
  converged:
  errors:
  linear_state:
  geom:
  old_transport_mass:
  """

  converged: bool = False
  errors: Optional[jnp.ndarray] = None
  linear_state: Optional[LinearOutput] = None
  geom: Optional[geometry.Geometry] = None

  def set(self, **kwargs: Any) -> "HTOutput":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix."""
    return self.linear_state.matrix

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply the transport to an array; axis=1 for its transpose."""
    return self.linear_state.apply(inputs, axis=axis)

  @property
  def reg_ht_cost(self) -> float:
    """Regularized optimal transport cost of the histogram transport."""
    return self.linear_state.reg_ot_cost

  @property
  def primal_cost(self) -> float:
    """Return transport cost of current linear OT solution at geometry."""
    return self.linear_state.transport_cost_at_geom(other_geom=self.geom)


class HTState(NamedTuple):
  """State of the Histogram-Transport solver.

  Attributes:
  costs:
  linear_convergence:
  linear_state:
  linear_pb:
  rngs:
  errors:
  """

  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_state: LinearOutput
  linear_pb: linear_problem.LinearProblem
  rngs: Optional[jax.random.PRNGKeyArray] = None
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "HTState":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(  # noqa: D102
      self,
      iteration: int,
      linear_sol: LinearOutput,
      linear_pb: linear_problem.LinearProblem,
      store_errors: bool,
  ) -> "HTState":
    costs = self.costs.at[iteration].set(linear_sol.reg_ot_cost)
    errors = None
    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :].set(linear_sol.errors)
    linear_convergence = self.linear_convergence.at[iteration].set(
        linear_sol.converged
    )

    return self.set(
        linear_state=linear_sol,
        linear_pb=linear_pb,
        costs=costs,
        linear_convergence=linear_convergence,
        errors=errors,
    )


# @jax.tree_util.register_pytree_node_class
class HistogramTransport:
  """Histogram Transport solver.

  Computes the First Lower Bound distance between two distributions.
  The current implementation requires uniform marginals.
  If there are an uneven number of points in the two distributions,
  then we perform a stratified subsample of the distribution of
  distances to be able to approximate the Wasserstein distance between
  the local distributions of distnaces.

  Args:
  epsilon_1d: regularization for soft sort. Set to `0.0` for normal sorting
  p: exponent for computing the transport distance betweeen histograms
  epsilon: regularization parameter for the resulting Sinkhorn problem
  min_iterations: minimum iterations for computing Sinkhorn distance
  max_iterations: maximum iterations for computing Sinkhorn distance
  """

  def __init__(
      self,
      epsilon_1d: float = 0.0,
      p: float = 1.0,
      epsilon: float = 1.0,
      min_iterations: int = 10,
      max_iterations: int = 100,
  ):
    self.epsilon_1d = epsilon_1d
    self.p = p
    self.epsilon = epsilon
    self.linear_ot_solver = sinkhorn.Sinkhorn(
        max_iterations=max_iterations, min_iterations=min_iterations
    )

  def __call__(
      self,
      prob: quadratic_problem.QuadraticProblem,
      rng: Optional[jax.random.PRNGKeyArray] = None,
  ) -> HTOutput:
    """Run the Histogram Transport solver.

    Args:
    prob: Quadratic OT problem.
    rng: Random number key.

    Returns:
    The Histogram Transport output.
    """
    rng = utils.default_prng_key(rng)

    dists_xx = prob.geom_xx.cost_matrix
    dists_yy = prob.geom_yy.cost_matrix
    m, n = dists_xx.shape[0], dists_yy.shape[0]
    min_num_pts = min([m, n])
    small_indices = jnp.arange(min_num_pts)
    x_indices = jnp.round(small_indices * (m / min_num_pts)).astype(int)
    y_indices = jnp.round(small_indices * (n / min_num_pts)).astype(int)

    if self.epsilon_1d <= 0.0:
      sorted_dists_xx = jax.lax.sort(dists_xx, dimension=-1)
      sorted_dists_yy = jax.lax.sort(dists_yy, dimension=-1)
    else:
      sorted_dists_xx = soft_sort.sort(
          dists_xx, axis=-1, epsilon=self.epsilon_1d
      )
      sorted_dists_yy = soft_sort.sort(
          dists_yy, axis=-1, epsilon=self.epsilon_1d
      )

    # Uniformly subsample distances
    sorted_dists_xx = jnp.take_along_axis(
        sorted_dists_xx, x_indices[None, :], -1
    )
    sorted_dists_yy = jnp.take_along_axis(
        sorted_dists_yy, y_indices[None, :], -1
    )

    match self.p:
      case 1.0:
        cost_xy = jnp.sum(
            jnp.abs(sorted_dists_xx[:, None, :] - sorted_dists_yy[None, :, :]),
            axis=-1,
        )
      case 2.0:
        cost_xy = jax.lax.sqrt(
            jnp.sum(
                jnp.square(
                    sorted_dists_xx[:, None, :] - sorted_dists_yy[None, :, :]
                ),
                axis=-1,
            )
        )
      case _:
        cost_xy = jnp.power(
            jnp.sum(
                jnp.power(
                    sorted_dists_xx[:, None, :] - sorted_dists_yy[None, :, :],
                    self.p,
                ),
                axis=-1,
            ),
            1 / self.p,
        )

    geom_xy = geometry.Geometry(cost_matrix=cost_xy, epsilon=self.epsilon)

    self.linear_pb = linear_problem.LinearProblem(geom=geom_xy)

    wass_out = self.linear_ot_solver(self.linear_pb)

    return HTOutput(
        converged=wass_out.converged,
        errors=wass_out.errors,
        linear_state=wass_out,
        geom=self.linear_pb.geom,
    )
