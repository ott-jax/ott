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

from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp

from ott.geometry import costs

__all__ = ["UnivariateSolver"]


@jax.tree_util.register_pytree_node_class
class UnivariateSolver:
  r"""1-D optimal transport solver.

  .. warning::
    This solver assumes uniform marginals, but a non-uniform marginal solver
    is coming soon.

  Computes the 1-Dimensional optimal transport distance between two histograms.

  Args:
    sort_fn: The sorting function. If :obj:`None`,
      use :func:`hard-sorting <jax.numpy.sort>`.
    cost_fn: The cost function for transport. If :obj:`None`, defaults to
      :class:`PNormP(2) <ott.geometry.costs.PNormP>`.
    method: The method used for computing the distance on the line. Options
      currently supported are:

      - `'subsample'`: Take a stratified sub-sample of the distances.
      - `'quantile'`: Take equally spaced quantiles of the distances.
      - `'equal'`: No sub-sampling is performed--requires distributions to have
        the same number of points.

    n_subsamples: The number of subsamples to draw for the "quantile" or
      "subsample" methods.
  """

  def __init__(
      self,
      sort_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      cost_fn: Optional[costs.CostFn] = None,
      method: Literal["subsample", "quantile", "equal"] = "subsample",
      n_subsamples: int = 100,
  ):
    self.sort_fn = jnp.sort if sort_fn is None else sort_fn
    self.cost_fn = costs.PNormP(2) if cost_fn is None else cost_fn
    self.method = method
    self.n_subsamples = n_subsamples

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Computes the Univariate OT Distance between `x` and `y`.

    Args:
      x: The first distribution.
      y: The second distribution.

    Returns:
      The OT distance.
    """
    sorted_x = self.sort_fn(x)
    sorted_y = self.sort_fn(y)
    n, m = sorted_x.shape[0], sorted_y.shape[0]

    if self.method == "equal":
      xx, yy = sorted_x, sorted_y
    elif self.method == "subsample":
      xx = sorted_x[jnp.linspace(0, n, num=self.n_subsamples).astype(int)]
      yy = sorted_y[jnp.linspace(0, m, num=self.n_subsamples).astype(int)]
    elif self.method == "quantile":
      xx = jnp.quantile(a=sorted_x, q=jnp.linspace(0, 1, self.n_subsamples))
      yy = jnp.quantile(a=sorted_y, q=jnp.linspace(0, 1, self.n_subsamples))
    else:
      raise NotImplementedError(f"Method `{self.method}` not implemented.")

    return self.cost_fn.pairwise(xx, yy)

  def tree_flatten(self):  # noqa: D102
    aux_data = vars(self).copy()
    return [], aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)
