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

from ott.geometry import distrib_cost, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers import linear
from ott.solvers.linear import sinkhorn

__all__ = ["LowerBoundSolver"]


@jax.tree_util.register_pytree_node_class
class LowerBoundSolver:
  """Lower bound OT solver :cite:`memoli:11`.

  Computes the third lower bound distance from :cite:`memoli:11`, def. 6.3.

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

  def __call__(
      self,
      prob: quadratic_problem.QuadraticProblem,
      rng: Optional[jax.Array] = None,
      kwargs_univsolver: Optional[Any] = None,
      epsilon: Optional[float] = None,
      **kwargs
  ) -> sinkhorn.SinkhornOutput:
    """Run the Histogram transport solver.

    Args:
      prob: Quadratic OT problem.
      kwargs_univsolver: keyword args to
        create the :class:`~ott.solvers.linear.univariate.UnivariateSolver`,
        used to compute a ``[n,m]`` cost matrix, using the linearization
        approach. This might rely, for instance, on subsampling or quantile
        reduction to speed up computations.
      rng: random key, possibly used when computing 1D costs when using
        subsampling.
      epsilon: entropic regularization passed on to solve the linearization of
        the quadratic problem using 1D costs.
      kwargs: Keyword arguments for :func:`~ott.solvers.linear.solve`.

    Returns:
      The Histogram transport output.
    """
    dists_xx = prob.geom_xx.cost_matrix
    dists_yy = prob.geom_yy.cost_matrix
    kwargs_univsolver = {} if kwargs_univsolver is None else kwargs_univsolver
    geom_xy = pointcloud.PointCloud(
        dists_xx,
        dists_yy,
        cost_fn=distrib_cost.UnivariateWasserstein(**kwargs_univsolver),
        epsilon=self.epsilon
    )

    return linear.solve(geom_xy, **kwargs)

  def tree_flatten(self):  # noqa: D102
    return [self.epsilon, self.univariate_solver], {}

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    epsilon, solver = children
    obj = cls(epsilon, **aux_data)
    obj.univariate_solver = solver
    return obj
