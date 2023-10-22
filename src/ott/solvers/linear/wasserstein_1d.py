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

from typing import Literal, Union

import jax
import jax.numpy as jnp

from ott.geometry.costs import PNormP, TICost
from ott.tools import soft_sort


class WassersteinSolver_1d:
  """One-Dimensional Optimal Transport Solver.

  WARNING: As implemented, this solver assumes uniform marginals,
  non-uniform marginal solver coming soon!

  Computes the 1-Dimensional Optimal Transport distance between two histograms
  via a variety of methods.

  Args:
  `epsilon`: regularization parameter for sorting. Set to 0.0 for hard-sorting.
  `cost_fn`: The cost function for transport. Defaults to Euclidean distance.
  `method`: The method used for computing the distance on the line. Options
  currently supported are:
  `subsample`: Take a stratfied subsample of the distances,
  `quantile`: Take equally spaced quantiles of the distances,
  `equal`: No subsampling is performed--requires distributions to have the same
  number of points.
  `n_subsamples`: The number of subsamples to draw for the "quantile" or
  "subsample" methods
  """

  def __init__(
      self,
      epsilon: float = 0.0,
      cost_fn: Union[TICost, None] = None,
      method: Literal["subsample", "quantile", "equal"] = "subsample",
      n_subsamples: int = 100,
  ):
    self.epsilon = epsilon
    self.method = method
    self.n_subsamples = n_subsamples
    if cost_fn is None:
      cost_fn = PNormP(2.0)
    self.cost_fn = cost_fn

  def __call__(self, x, y):
    """Computes the 1D Wasserstein Distance between `x` and `y`."""
    match self.method:
      case "subsample":
        return self.cost_fn.pairwise(
            self._sort(x)[jnp.linspace(0, x.shape[0],
                                       num=self.n_subsamples).astype(int)],
            self._sort(y)[jnp.linspace(0, y.shape[0],
                                       num=self.n_subsamples).astype(int)],
        )
      case "equal":
        return self.cost_fn.pairwise(self._sort(x), self._sort(y))
      case "quantile":
        return self.cost_fn.pairwise(
            jnp.quantile(
                a=self._sort(x), q=jnp.linspace(0, 1, self.n_subsamples)
            ),
            jnp.quantile(
                a=self._sort(y), q=jnp.linspace(0, 1, self.n_subsamples)
            ),
        )
      case _:
        raise KeyError(f"Method {self.method} not implemented!")

  def _power_cost(self, sorted_x, sorted_y):
    match self.p:
      case 1.0:
        cost = jnp.sum(jnp.abs(sorted_x - sorted_y))
      case 2.0:
        cost = jax.lax.sqrt(jnp.sum(jnp.square(sorted_x - sorted_y)))
      case _:
        cost = jnp.power(
            jnp.sum(jnp.power(jnp.abs(sorted_x - sorted_y), self.p)),
            1 / self.p,
        )
    return cost

  def _sort(self, x):
    return jax.lax.cond(
        self.epsilon <= 0,
        jax.lax.sort,
        lambda v: soft_sort.sort(v, epsilon=self.epsilon),
        x,
    )
