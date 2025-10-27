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
  """TODO."""
  rng: jax.Array
  dataset: Iterable[Tuple[jax.Array, jax.Array]]
  epsilon: Optional[float] = None
  relative_epsilon: Optional[Literal["mean", "std"]] = None
  cost_fn: Optional[costs.CostFn] = None
  threshold: float = 1e-3
  max_iterations: int = 2000
  replace: bool = True
  in_shardings: Optional[jax.sharding.Sharding] = None
  out_shardings: Optional[jax.sharding.Sharding] = None

  def __post_init__(self):
    self._align_fn = jax.jit(
        functools.partial(
            _align,
            threshold=self.threshold,
            max_iterations=self.max_iterations
        ),
        static_argnames=["cost_fn", "epsilon", "relative_epsilon", "replace"],
        in_shardings=self.in_shardings,
        out_shardings=self.out_shardings,
    )
    self._data_it: Optional[Iterator[Tuple[jax.Array, jax.Array]]] = None
    self._rng_it: Optional[jax.Array] = None

  def __iter__(self) -> "LinearOTDataloader":
    """Return self."""
    self._data_it = iter(self.dataset)
    self._rng_it = self.rng
    return self

  def __next__(self) -> Tuple[jax.Array, jax.Array]:
    """TODO."""
    assert self._data_it is not None, "Please call `iter()` first."
    assert self._rng_it is not None, "Please call `iter()` first."
    self._rng_it, rng_sample = jr.split(self._rng_it, 2)
    x, y = next(self._data_it)
    return self._align_fn(
        rng_sample,
        x,
        y,
        cost_fn=self.cost_fn,
        epsilon=self.epsilon,
        relative_epsilon=self.relative_epsilon,
        replace=self.replace,
    )


def _align(
    rng: jax.Array,
    x: jax.Array,
    y: jax.Array,
    *,
    cost_fn: costs.CostFn,
    epsilon: Optional[float],
    relative_epsilon: Optional[...],
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
  jax.debug.print("r={},c={}", row_ixs[:5], col_ixs[:5])
  return x[row_ixs], y[col_ixs]
