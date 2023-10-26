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

from typing import Any, Optional

import jax
import jax.experimental

from ott.geometry import geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers import linear
from ott.solvers.linear import sinkhorn, univariate

__all__ = ["HistogramTransport"]


@jax.tree_util.register_pytree_node_class
class HistogramTransport:
  """Histogram Transport solver.

  warning::
  As implemented, this solver assumes uniform marginals,
  non-uniform marginal solver coming soon!

  Computes the First Lower Bound distance from :cite:`memoli:11` between two
  distributions. The current implementation requires uniform marginals.
  If there are an uneven number of points in the two distributions,
  then we perform a stratified subsample of the distribution of
  distances to be able to approximate the Wasserstein distance between
  the local distributions of distnaces.

  Args:
    epsilon: regularization parameter for the resulting Sinkhorn problem
    min_iterations: minimum iterations for computing Sinkhorn distance
    max_iterations: maximum iterations for computing Sinkhorn distance
    kwargs: keyword arguments for the 1D Wasserstein computation
  """

  def __init__(
      self,
      epsilon: Optional[float] = None,
      **kwargs: Any,
  ):
    self.epsilon = epsilon
    self.univariate_solver = univariate.UnivariateSolver(**kwargs)

  def __call__(
      self,
      prob: quadratic_problem.QuadraticProblem,
      rng: Optional[jax.random.PRNGKeyArray] = None,
      **kwargs: Any,
  ) -> sinkhorn.SinkhornOutput:
    """Run the Histogram Transport solver.

    Args:
      prob: quadratic OT problem.
      rng: random number key (not used)
      kwargs: keyword arguments for the Sinkhorn solver

    Returns:
      The Histogram Transport output.
    """
    del rng

    dists_xx = prob.geom_xx.cost_matrix
    dists_yy = prob.geom_yy.cost_matrix
    cost_xy = jax.vmap(
        jax.vmap(self.univariate_solver, in_axes=(0, None), out_axes=-1),
        in_axes=(None, 0),
        out_axes=-1,
    )(dists_xx, dists_yy)

    geom_xy = geometry.Geometry(cost_matrix=cost_xy, epsilon=self.epsilon)

    return linear.solve(geom_xy, **kwargs)

  def tree_flatten(self):  # noqa: D102
    aux = vars(self).copy()
    univariate_solver = aux.pop("univariate_solver")
    return [univariate_solver], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    ht_solver = cls(**aux_data)
    ht_solver.univariate_solver = children[0]
    return ht_solver
