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
import dataclasses
import functools
from typing import Any, Iterable, Iterator, Literal, Optional, Tuple

import jax
import jax.random as jr

from ott.geometry import costs, pointcloud
from ott.solvers import linear

__all__ = ["LinearOTDataloader"]


@dataclasses.dataclass(frozen=False, repr=False)
class LinearOTDataloader:
  """Linear OT dataloader.

  This dataloader wraps a dataloader that generates ``(source, target)``
  arrays with shape ``[batch, ...]`` and aligns them
  using the :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` algorithm.

  Args:
    rng: Random number generator.
    dataset: Iterable dataset which yields a tuple of source and target arrays
      of shape ``[batch, ...]``.
    epsilon: Epsilon regularization.
      See :class:`~ott.geometry.geometry.Geometry` for more information.
    relative_epsilon: Whether ``epsilon`` refers to a fraction of the
      :attr:`~ott.geometry.pointcloud.PointCloud.mean_cost_matrix` or
      :attr:`~ott.geometry.pointcloud.PointCloud.std_cost_matrix`.
    cost_fn: Cost function between two points.
    threshold: Convergence threshold for
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.
    max_iterations: Maximum number of Sinkhorn iterations.
    replace: Whether to sample with replacement.
    shardings: Input and output shardings for the source and target arrays.
  """
  rng: jax.Array
  dataset: Iterable[Tuple[jax.Array, jax.Array]]
  epsilon: Optional[float] = None
  relative_epsilon: Optional[Literal["mean", "std"]] = None
  cost_fn: Optional[costs.CostFn] = None
  threshold: float = 1e-3
  max_iterations: int = 2000
  replace: bool = True
  shardings: Optional[jax.sharding.Sharding] = None

  def __post_init__(self) -> None:
    self._align_fn = jax.jit(
        functools.partial(
            _align,
            threshold=self.threshold,
            max_iterations=self.max_iterations
        ),
        static_argnames=["cost_fn", "epsilon", "relative_epsilon", "replace"],
        in_shardings=(None, self.shardings, self.shardings),
        out_shardings=(self.shardings, self.shardings),
    )
    self._data_it: Optional[Iterator[Tuple[jax.Array, jax.Array]]] = None
    self._rng_it: Optional[jax.Array] = None

  def __iter__(self) -> "LinearOTDataloader":
    """Return self."""
    self._data_it = iter(self.dataset)
    self._rng_it = self.rng
    return self

  def __next__(self) -> Tuple[jax.Array, jax.Array]:
    """Align source and target samples in a batch.

    Returns:
      The aligned source and target arrays of shape ``[batch, ...]``.
    """
    assert self._data_it is not None, "Please call `iter()` first."
    assert self._rng_it is not None, "Please call `iter()` first."
    self._rng_it, rng_sample = jr.split(self._rng_it, 2)
    x, y = next(self._data_it)
    return self._align_fn(
        rng_sample,
        x,
        y,
        self.cost_fn,
        self.epsilon,
        self.relative_epsilon,
        self.replace,
    )


def _align(
    rng: jax.Array,
    x: jax.Array,
    y: jax.Array,
    cost_fn: costs.CostFn,
    epsilon: Optional[float],
    relative_epsilon: Optional[Literal["mean", "std"]],
    replace: bool,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array]:
  geom = pointcloud.PointCloud(
      x,
      y,
      cost_fn=cost_fn,
      epsilon=epsilon,
      relative_epsilon=relative_epsilon,
  )
  out = linear.solve(geom, **kwargs)

  n, m = geom.shape
  probs = out.matrix.ravel()
  probs = probs / probs.sum()

  ixs = jr.choice(rng, n * m, shape=(n,), p=probs, replace=replace)
  row_ixs, col_ixs = ixs // m, ixs % m
  return x[row_ixs], y[col_ixs]
