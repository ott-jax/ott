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

from typing import Literal, Optional

import jax
import jax.numpy as jnp

from ott.geometry import costs
from ott.tools import soft_sort

__all__ = ["UnivariateSolver"]


@jax.tree_util.register_pytree_node_class
class UnivariateSolver:
  r"""1-D optimal transport solver.

  .. warning::

    As implemented, this solver assumes uniform marginals,
    non-uniform marginal solver coming soon!

  Computes the 1-Dimensional Optimal Transport distance between two histograms
  via a variety of methods.

  Args:
    epsilon_sort: regularization parameter for sorting. If :math:`\le 0` use
      `hard-sorting <jax.numpy.sort>`.
    cost_fn: The cost function for transport. Defaults to Squared Euclidean
      distance.
    method: The method used for computing the distance on the line. Options
      currently supported are:
      - `'subsample'`: Take a stratified sub-sample of the distances,
      - `'quantile'`: Take equally spaced quantiles of the distances,
      - `'equal'`: No sub-sampling is performed--requires distributions to have
        the same number of points.
    n_subsamples: The number of sub-samples to draw for the "quantile" or
      "subsample" methods
    requires_sort: Whether to assume that the inputted arrays are sorted.
    n_iterations: The number of iterations for computing the soft sort. Ignored
      when `epsilon_sort = 0`.
  """

  def __init__(
      self,
      epsilon_sort: Optional[float] = 0.0,
      cost_fn: Optional[costs.CostFn] = None,
      method: Literal["subsample", "quantile", "equal"] = "subsample",
      n_subsamples: int = 100,
      requires_sort: bool = False,
      min_iterations: int = 50,
      max_iterations: int = 50,
  ):
    self.epsilon_sort = epsilon_sort
    self.method = method
    self.n_subsamples = n_subsamples
    if cost_fn is None:
      cost_fn = costs.PNormP(2.0)
    self.cost_fn = cost_fn
    self.requires_sort = requires_sort
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Computes the Univariate OT Distance between `x` and `y`."""
    sorted_x, sorted_y = (self._sort(x),
                          self._sort(y)) if self.requires_sort else (x, y)
    if self.method == "subsample":
      return self.cost_fn.pairwise(
          sorted_x[jnp.linspace(0, x.shape[0],
                                num=self.n_subsamples).astype(int)],
          sorted_y[jnp.linspace(0, y.shape[0],
                                num=self.n_subsamples).astype(int)],
      )
    if self.method == "equal":
      return self.cost_fn.pairwise(sorted_x, sorted_y)
    if self.method == "quantile":
      return self.cost_fn.pairwise(
          jnp.quantile(a=sorted_x, q=jnp.linspace(0, 1, self.n_subsamples)),
          jnp.quantile(a=sorted_y, q=jnp.linspace(0, 1, self.n_subsamples)),
      )
    raise KeyError(f"Method {self.method} not implemented!")

  def _sort(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.epsilon_sort > 0 or self.epsilon_sort is None:
      return soft_sort.sort(
          x,
          epsilon=self.epsilon_sort,
          min_iterations=self.min_iterations,
          max_iterations=self.max_iterations
      )
    return jnp.sort(x)

  def tree_flatten(self):  # noqa: D102
    aux = vars(self).copy()
    aux.pop("cost_fn")
    aux.pop("epsilon_sort")
    return [self.cost_fn, self.epsilon_sort], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)
