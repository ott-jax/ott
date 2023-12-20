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

import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import univariate

__all__ = [
    "UnivariateWasserstein",
]


@jtu.register_pytree_node_class
class UnivariateWasserstein(costs.CostFn):
  """1D Wasserstein cost for two 1D distributions.

  This ground cost between considers vectors as a family of values.
  The Wasserstein distance between them is the 1D OT cost, using a user-defined
  ground cost.

  Args:
    ground_cost: Cost used to compute the 1D optimal transport between vector,
      should be a translation-invariant (TI) cost for correctness.
      If :obj:`None`, defaults to :class:`~ott.geometry.costs.SqEuclidean`.
    solver: 1D optimal transport solver.
    kwargs: Arguments passed on when calling the
      :class:`~ott.solvers.linear.univariate.UnivariateSolver`. May include
      random key, or specific instructions to subsample or compute using
      quantiles.
  """

  def __init__(
      self,
      ground_cost: Optional[costs.TICost] = None,
      solver: Optional[univariate.UnivariateSolver] = None,
      **kwargs: Any
  ):
    super().__init__()

    self.ground_cost = (
        costs.SqEuclidean() if ground_cost is None else ground_cost
    )
    self._solver = univariate.UnivariateSolver() if solver is None else solver
    self._kwargs_solve = kwargs
    # ensure transport solutions are neither computed nor stored
    self._kwargs_solve["return_transport"] = False

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Wasserstein distance between :math:`x` and :math:`y` seen as a 1D dist.

    Args:
      x: Array of shape ``[n,]``.
      y: Array of shape ``[m,]``.

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
    return (self.ground_cost,), (self._solver, self._kwargs_solve)

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    ground_cost, = children
    solver, solve_kwargs = aux_data
    return cls(ground_cost, solver, **solve_kwargs)
