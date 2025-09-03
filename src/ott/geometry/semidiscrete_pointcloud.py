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
from typing import Callable, Literal, Optional, Tuple, Union

import jax
import jax.tree_util as jtu

from ott.geometry import costs, pointcloud

__all__ = ["SemidiscretePointCloud"]


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscretePointCloud:
  """TODO."""
  sampler: Callable[[jax.Array, Tuple[int, ...]],
                    jax.Array] = dataclasses.field(metadata={"static": True})
  y: jax.Array
  epsilon: Union[float, jax.Array]
  relative_epsilon: Optional[Literal["mean", "std"]] = dataclasses.field(
      default=None, metadata={"static": True}
  )
  cost_fn: costs.CostFn = costs.SqEuclidean()
  scale_cost: Union[float, Literal["mean", "max_norm", "max_bound", "max_cost",
                                   "median"]] = dataclasses.field(
                                       default=1.0, metadata={"static": True}
                                   )

  def materialize(
      self, rng: jax.Array, num_samples: int
  ) -> pointcloud.PointCloud:
    """TODO."""
    x = self.sampler(rng, (num_samples, *self.y.shape[1:]))
    return self._from_samples(x)

  def _from_samples(self, x: jax.Array) -> pointcloud.PointCloud:
    return pointcloud.PointCloud(
        x,
        self.y,
        cost_fn=self.cost_fn,
        epsilon=self.epsilon,
        relative_epsilon=self.relative_epsilon,
        scale_cost=self.scale_cost,
    )

  @property
  def shape(self) -> tuple[float, int]:
    """TODO."""
    return float("inf"), self.y.shape[0]
