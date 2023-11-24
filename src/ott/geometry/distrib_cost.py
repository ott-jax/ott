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
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import univariate

__all__ = [
    "UnivariateWasserstein",
]


@jax.tree_util.register_pytree_node_class
class UnivariateWasserstein(costs.CostFn):
  """1D Wasserstein cost for two 1D distributions.

  This ground cost between considers vectors as a family of values. The
  Wasserstein distance between them is the 1D OT cost, using a user-defined
  ground cost.
  """

  def __init__(
      self,
      ground_cost: Optional[costs.TICost] = None,
      kwargs_solve: Optional[Any] = None,
      **kwargs: Any
  ):
    super().__init__()
    if ground_cost is None:
      self.ground_cost = costs.SqEuclidean()
    else:
      self.ground_cost = ground_cost
    self._kwargs_solve = {} if kwargs_solve is None else kwargs_solve
    self._kwargs = kwargs
    self._solver = univariate.UnivariateSolver(**kwargs)

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Wasserstein distance between :math:`x` and :math:`y` seen as a 1D dist.

    Args:
      x: vector
      y: vector
      kwargs: arguments passed on when calling the
        :class:`~ott.solvers.linear.univariate.UnivariateSolver`. May include
        random key, or specific instructions to subsample or compute using
        quantiles.

    Returns:
      The transport cost.
    """
    out = self._solver(
        linear_problem.LinearProblem(
            pointcloud.PointCloud(
                x[:, None], y[:, None], cost_fn=self.ground_cost
            )
        ), **self._kwargs_solve
    )
    return jnp.squeeze(out.ot_costs)

  def tree_flatten(self):  # noqa: D102
    return (), (self.ground_cost, self._kwargs_solve, self._kwargs)

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del children
    gc, kws, kw = aux_data
    return cls(gc, kws, **kw)
