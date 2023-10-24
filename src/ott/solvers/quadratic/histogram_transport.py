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
    Optional,
)

import jax
import jax.experimental

from ott import utils
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn, univariate

__all__ = ["HistogramTransport"]


# @jax.tree_util.register_pytree_node_class
class HistogramTransport:
  """Histogram Transport solver.

  WARNING: As implemented, this solver assumes uniform marginals,
  non-uniform marginal solver coming soon!

  Computes the First Lower Bound distance between two distributions.
  The current implementation requires uniform marginals.
  If there are an uneven number of points in the two distributions,
  then we perform a stratified subsample of the distribution of
  distances to be able to approximate the Wasserstein distance between
  the local distributions of distnaces.

  Args:
  `epsilon`: regularization parameter for the resulting Sinkhorn problem
  `min_iterations`: minimum iterations for computing Sinkhorn distance
  `max_iterations`: maximum iterations for computing Sinkhorn distance
  `**kwargs`: keyword arguments for the 1D Wasserstein computation
  """

  def __init__(
      self,
      epsilon: float = 1.0,
      min_iterations: int = 10,
      max_iterations: int = 100,
      **kwargs,
  ):
    self.epsilon = epsilon
    self.linear_ot_solver = sinkhorn.Sinkhorn(
        max_iterations=max_iterations, min_iterations=min_iterations
    )
    wass_solver_1d = univariate.UnivariateSolver(**kwargs)
    self.wass_solver_1d_vmap = jax.vmap(
        jax.vmap(wass_solver_1d, in_axes=(0, None), out_axes=-1),
        in_axes=(None, 0),
        out_axes=-1,
    )

  def __call__(
      self,
      prob: quadratic_problem.QuadraticProblem,
      rng: Optional[jax.random.PRNGKeyArray] = None,
  ) -> sinkhorn.SinkhornOutput:
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
    cost_xy = self.wass_solver_1d_vmap(dists_xx, dists_yy)

    geom_xy = geometry.Geometry(cost_matrix=cost_xy, epsilon=self.epsilon)

    self.linear_pb = linear_problem.LinearProblem(geom=geom_xy)

    return self.linear_ot_solver(self.linear_pb)
