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
from typing import Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from ott.geometry import costs, pointcloud

__all__ = ["SemidiscretePointCloud"]


@jtu.register_pytree_node_class
class SemidiscretePointCloud:
  """TODO."""

  def __init__(
      self,
      sampler: Callable[[jax.Array, Tuple[int, ...]], jax.Array],
      y: jax.Array,
      cost_fn: Optional[costs.CostFn] = None,
      epsilon: Optional[Union[float, jax.Array]] = None,
      relative_epsilon: Optional[Literal["mean", "std"]] = None,
      scale_cost: Union[float, Literal["mean", "max_norm", "max_bound",
                                       "max_cost", "median"]] = 1.0,
      epsilon_rng: Optional[jax.Array] = None,
      epsilon_num_samples: int = 1024,
  ):
    assert epsilon_num_samples > 0, \
      "Number of samples for epsilon must be positive."
    self.sampler = sampler
    self.y = y
    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    self._epsilon = epsilon
    self._relative_epsilon = relative_epsilon
    self._scale_cost = scale_cost
    self._epsilon_rng = epsilon_rng
    self._epsilon_num_samples = epsilon_num_samples

  def sample(self, rng: jax.Array, num_samples: int) -> pointcloud.PointCloud:
    """TODO."""
    assert num_samples > 0, "Number of samples must be positive."
    x = self.sampler(rng, (num_samples, *self.y.shape[1:]))
    return self._from_samples(x, self.epsilon)

  def _from_samples(
      self, x: jax.Array, epsilon: Union[float, jax.Array]
  ) -> pointcloud.PointCloud:
    return pointcloud.PointCloud(
        x,
        self.y,
        cost_fn=self.cost_fn,
        epsilon=epsilon,
        relative_epsilon=self._relative_epsilon,
        scale_cost=self._scale_cost,
    )

  @property
  def epsilon(self) -> float:
    """TODO."""
    rng = jr.key(0) if self._epsilon_rng is None else self._epsilon_rng
    x = self.sampler(rng, (self._epsilon_num_samples, *self.y.shape[1:]))
    geom = self._from_samples(x, self._epsilon)
    return geom.epsilon

  @property
  def is_entropy_regularized(self) -> bool:
    """TODO."""
    return self._epsilon is not None and self._epsilon > 0.0

  @property
  def shape(self) -> tuple[float, int]:
    """TODO."""
    return float("inf"), self.y.shape[0]

  @property
  def dtype(self) -> jnp.dtype:
    """TODO."""
    return self.y.dtype

  def tree_flatten(self):  # noqa: D102
    return (self.y, self.cost_fn), (
        self.sampler, {
            "epsilon": self._epsilon,
            "relative_epsilon": self._relative_epsilon,
            "scale_cost": self._scale_cost,
            "epsilon_rng": self._epsilon_rng,
            "epsilon_num_samples": self._epsilon_num_samples,
        }
    )

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    y, cost_fn = children
    sampler, aux_data = aux_data
    return cls(sampler, y, cost_fn=cost_fn, **aux_data)
