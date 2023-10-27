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

from ott.geometry import geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers import linear
from ott.solvers.linear import sinkhorn, univariate

__all__ = ["LowerBoundSolver"]


@jax.tree_util.register_pytree_node_class
class LowerBoundSolver:
  """Lower bound OT solver :cite:`memoli:11`.

  .. warning::
    As implemented, this solver assumes uniform marginals,
    non-uniform marginal solver coming soon!

  Computes the first lower bound distance from :cite:`memoli:11`, def. 6.1.
  there is an uneven number of points in the distributions, then we perform a
  stratified subsample of the distribution of distances to approximate
  the Wasserstein distance between the local distributions of distances.

  Args:
    epsilon: Entropy regularization for the resulting linear problem.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.linear.univariate.UnivariateSolver`.
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
      **kwargs: Any,
  ) -> sinkhorn.SinkhornOutput:
    """Run the Histogram transport solver.

    Args:
      prob: Quadratic OT problem.
      kwargs: Keyword arguments for :func:`~ott.solvers.linear.solve`.

    Returns:
      The Histogram transport output.
    """
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
    return [self.epsilon, self.univariate_solver], {}

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    epsilon, solver = children
    obj = cls(epsilon, **aux_data)
    obj.univariate_solver = solver
    return obj
