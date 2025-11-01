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
  """Semidiscrete point cloud geometry.

  Instances of this geometry can be sampled using the :meth:`sample` method.

  Args:
    sampler: Function with a signature ``(rng, shape, dtype) -> array``
      corresponding to the source distribution.
    y: Array of shape ``[m, ...]`` corresponding to the target distribution.
    cost_fn: Cost function. If :obj:`None`,
      use :class:`~ott.geometry.costs.SqEuclidean`.
    epsilon: Regularization parameter. Can be set to :math:`0` to solve the
      unregularized :term:`semidiscrete problem`.
    relative_epsilon: Whether ``epsilon`` refers to a fraction of the
      :attr:`~ott.geometry.pointcloud.PointCloud.mean_cost_matrix` or
      :attr:`~ott.geometry.pointcloud.PointCloud.std_cost_matrix`.
    scale_cost: Option to rescale the cost matrix.
    relative_epsilon_seed: Random seed when estimating the :attr:`epsilon`.
    relative_epsilon_num_samples: Number of samples when estimating
      the :attr:`epsilon`.
  """

  def __init__(
      self,
      sampler: Callable[[jax.Array, Tuple[int, ...], Optional[jnp.dtype]],
                        jax.Array],
      y: jax.Array,
      cost_fn: Optional[costs.CostFn] = None,
      epsilon: Optional[Union[float, jax.Array]] = None,
      relative_epsilon: Optional[Literal["mean", "std"]] = None,
      scale_cost: Union[float, Literal["mean", "max_norm", "max_bound",
                                       "max_cost", "median"]] = 1.0,
      relative_epsilon_seed: int = 0,
      relative_epsilon_num_samples: int = 1024,
  ):
    assert relative_epsilon_num_samples > 0, \
      "Number of samples when estimating relative epsilon must be positive."
    self.sampler = sampler
    self.y = y
    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    self._epsilon = epsilon
    self._relative_epsilon = relative_epsilon
    self._scale_cost = scale_cost
    self._relative_epsilon_seed = relative_epsilon_seed
    self._relative_epsilon_num_samples = relative_epsilon_num_samples

  def sample(
      self,
      rng: jax.Array,
      num_samples: int,
      *,
      epsilon: Optional[float] = None
  ) -> pointcloud.PointCloud:
    """Sample a point cloud.

    .. note::
      When :attr:`is_entropy_regularized = False <is_entropy_regularized>`,
      some methods and attributes of the sampled
      :class:`~ott.geometry.pointcloud.PointCloud` are not meaningful.
      However, this does not impact the usage of the
      :class:`~ott.solvers.linear.semidiscrete.SemidiscreteSolver`.

    Args:
      rng: Random key used for seeding.
      num_samples: Number of samples.
      epsilon: Epsilon regularization. If :obj:`None`, use :attr:`epsilon`.

    Returns:
      The sampled point cloud.
    """
    assert num_samples > 0, f"Number of samples must be > 0, got {num_samples}."
    shape = (num_samples, *self.y.shape[1:])
    x = self.sampler(rng, shape, self.dtype)
    if epsilon is None:
      epsilon = self.epsilon
    return self._from_samples(x, epsilon)

  def _from_samples(
      self, x: jax.Array, epsilon: jax.Array
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
  def epsilon(self) -> jax.Array:
    """Epsilon regularization value."""
    if not self.is_entropy_regularized:
      return jnp.array(0.0, dtype=self.dtype)
    rng = jr.key(self._relative_epsilon_seed)
    shape = (self._relative_epsilon_num_samples, *self.y.shape[1:])
    x = self.sampler(rng, shape, self.dtype)
    geom = self._from_samples(x, self._epsilon)
    return jnp.array(geom.epsilon, dtype=self.dtype)

  @property
  def is_entropy_regularized(self) -> bool:
    """Whether ``epsilon > 0``."""
    return self._epsilon is None or self._epsilon > 0.0

  @property
  def shape(self) -> tuple[float, int]:
    """Shape of the geometry."""
    return float("inf"), self.y.shape[0]

  @property
  def dtype(self) -> jnp.dtype:
    """The data type."""
    return self.y.dtype

  def tree_flatten(self):  # noqa: D102
    return (self.y, self.cost_fn), (
        self.sampler, {
            "epsilon": self._epsilon,
            "relative_epsilon": self._relative_epsilon,
            "scale_cost": self._scale_cost,
            "relative_epsilon_seed": self._relative_epsilon_seed,
            "relative_epsilon_num_samples": self._relative_epsilon_num_samples,
        }
    )

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    y, cost_fn = children
    sampler, aux_data = aux_data
    return cls(sampler, y, cost_fn=cost_fn, **aux_data)
