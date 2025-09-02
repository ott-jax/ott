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
from typing import Callable, Tuple, Union

import jax
import jax.tree_util as jtu

from ott.geometry import costs, pointcloud

__all__ = ["SemidiscretePointCloud"]


@functools.partial(jtu.register_dataclass, meta_fields=["sampler"])
@dataclasses.dataclass
class SemidiscretePointCloud:
  """TODO."""

  sampler: Callable[[jax.Array, Tuple[int, ...]], jax.Array]
  y: jax.Array
  epsilon: Union[float, jax.Array]
  cost_fn: costs.CostFn = costs.SqEuclidean()

  def materialize(
      self, rng: jax.Array, num_samples: int
  ) -> pointcloud.PointCloud:
    """TODO."""
    x = self.sampler(rng, (num_samples, *self.y.shape[1:]))
    return pointcloud.PointCloud(
        x,
        self.y,
        cost_fn=self.cost_fn,
        epsilon=self.epsilon,
    )
