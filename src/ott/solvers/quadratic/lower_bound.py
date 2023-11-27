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

from typing import TYPE_CHECKING, Any, Optional

import jax

from ott.geometry import pointcloud

if TYPE_CHECKING:
  from ott.solvers.linear import distrib_costs
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
    cost_fn: Univariate Wasserstein cost, used to compare two point clouds in
      different spaces, where each point is seen as its distribution of costs
      to other points in its point-cloud.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.linear.univariate.UnivariateSolver`.
  """

  def __init__(
      self,
      epsilon: Optional[float] = None,
      distrib_cost: Optional["distrib_costs.UnivariateWasserstein"] = None,
  ):
    from ott.geometry import distrib_costs
    self.epsilon = epsilon
    if distrib_cost is None:
      distrib_cost = distrib_costs.UnivariateWasserstein()
    self.distrib_cost = distrib_cost

  def __call__(
      self,
      prob: quadratic_problem.QuadraticProblem,
      epsilon: Optional[float] = None,
      rng: Optional[jax.Array] = None,
      **kwargs: Any
  ) -> sinkhorn.SinkhornOutput:
    """Compute a lower-bound for the GW problem using a simple linearization.

    This solver handles a quadratic problem by computing first a proxy ``[n,m]``
    cost-matrix, inject it into a linear OT solver, to output a first OT matrix
    that can be used either to linearize/initialize the resolution of the GW
    problem, or more simply as a simple GW solution.

    Args:
      prob: Quadratic OT problem.
      epsilon: entropic regularization passed on to solve the linearization of
        the quadratic problem using 1D costs.
      rng: random key, possibly used when computing 1D costs when using
        subsampling.
      kwargs: Keyword arguments for :func:`~ott.solvers.linear.solve`.

    Returns:
      A linear OT output, an approximation of the OT coupling obtained using
      the lower bound provided by :cite:`memoli:11`.
    """
    dists_xx = prob.geom_xx.cost_matrix
    dists_yy = prob.geom_yy.cost_matrix

    geom_xy = pointcloud.PointCloud(
        dists_xx, dists_yy, cost_fn=self.distrib_cost, epsilon=self.epsilon
    )
    return linear.solve(geom_xy, **kwargs)

  def tree_flatten(self):  # noqa: D102
    return (self.epsilon, self.distrib_cost), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del aux_data
    return cls(*children)
