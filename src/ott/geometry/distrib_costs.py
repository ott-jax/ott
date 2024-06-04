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
from typing import Callable, Optional

import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import univariate

__all__ = ["UnivariateWasserstein"]


@jtu.register_pytree_node_class
class UnivariateWasserstein(costs.CostFn):
  """1D Wasserstein cost for two 1D distributions.

  This ground cost between considers vectors as a family of values.
  The Wasserstein distance between them is the 1D OT cost, using a user-defined
  ground cost.

  Args:
    solve_fn: 1D optimal transport solver, e.g.,
      :func:`~ott.solvers.linear.univariate.uniform_distance`.
    ground_cost: Cost used to compute the 1D optimal transport between vectors.
      Should be a translation-invariant (TI) cost for correctness.
      If :obj:`None`, defaults to :class:`~ott.geometry.costs.SqEuclidean`.
  """

  def __init__(
      self,
      solve_fn: Callable[[linear_problem.LinearProblem],
                         univariate.UnivariateOutput],
      ground_cost: Optional[costs.TICost] = None,
  ):
    super().__init__()
    self.ground_cost = (
        costs.SqEuclidean() if ground_cost is None else ground_cost
    )
    self._solve_fn = solve_fn

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Wasserstein distance between :math:`x` and :math:`y` seen as a 1D dist.

    Args:
      x: Array of shape ``[n,]``.
      y: Array of shape ``[m,]``.

    Returns:
      The transport cost.
    """
    geom = pointcloud.PointCloud(
        x[:, None], y[:, None], cost_fn=self.ground_cost
    )
    prob = linear_problem.LinearProblem(geom)
    out = self._solve_fn(prob)
    return jnp.squeeze(out.ot_costs)

  def tree_flatten(self):  # noqa: D102
    return (self.ground_cost,), (self._solve_fn,)

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(solve_fn=aux_data[0], ground_cost=children[0])
