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

from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers import linear
from ott.solvers.linear import sinkhorn

if TYPE_CHECKING:
  from ott.geometry import distrib_costs

__all__ = ["third_lower_bound"]


def third_lower_bound(
    prob: quadratic_problem.QuadraticProblem,
    distrib_cost: "distrib_costs.UnivariateWasserstein",
    epsilon: Optional[float] = None,
    **kwargs: Any,
) -> sinkhorn.SinkhornOutput:
  """Computes the third lower bound distance from :cite:`memoli:11`, def. 6.3.

  Args:
    prob: Quadratic OT problem.
    distrib_cost: Univariate Wasserstein cost used to compare two point clouds
      in different spaces. Each point is seen as its distribution of costs
      to other points in its respective point cloud.
    epsilon: Entropy regularization.
    kwargs: Keyword arguments for :func:`~ott.solvers.linear.solve`.

  Returns:
    An approximation of the GW coupling that can be used to initialize
    the solution of the quadratic OT problem.
  """
  dists_xx = prob.geom_xx.cost_matrix
  dists_yy = prob.geom_yy.cost_matrix
  geom_xy = pointcloud.PointCloud(
      dists_xx, dists_yy, cost_fn=distrib_cost, epsilon=epsilon
  )

  return linear.solve(geom_xy, **kwargs)
