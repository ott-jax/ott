import abc
from typing import Any, Dict, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import pointcloud

__all__ = ["EntropicMap"]
# TODO(michalk8): finish the type
Potential = Union[jnp.ndarray]


# TODO(michalk8): better name
@jtu.register_pytree_node_class
class DualPotentials(abc.ABC):

  def __init__(self, f: Potential, g: Potential):
    self._f = f
    self._g = g

  # TODO(michalk8): batch size
  @abc.abstractmethod
  def transport(self, x: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """TODO(michalk8).

    Args:
      vec: TODO(michalk8)
      axis: TODO(michalk8)

    Returns:
      TODO(michalk8)
    """

  @property
  def f(self) -> Potential:
    return self._f

  @property
  def g(self) -> Potential:
    return self._g

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self.f, self.g], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "DualPotentials":
    return cls(*children, **aux_data)


@jtu.register_pytree_node_class
class EntropicMap(DualPotentials):
  """TODO(michalk8)."""

  def __init__(
      self, f: jnp.ndarray, g: jnp.ndarray, geom: pointcloud.PointCloud
  ):
    super().__init__(f, g)
    self._geom = geom

  def transport(
      self,
      vec: jnp.ndarray,
      axis: int = 1,
  ) -> jnp.ndarray:
    if vec.ndim == 1:
      vec = vec.reshape((1, -1))

    f = jnp.zeros(vec.shape[0])  # (k, d)
    if axis == 0:
      y, g = self._geom.x, self.f  # (n, d), (n,)
    else:
      y, g = self._geom.y, self.g  # (m, d), (m,)

    geom = pointcloud.PointCloud(vec, y, epsilon=self.epsilon)  # (k, {n, m})
    # (d, k), (d, k)
    res, sgn = jax.vmap(lambda v: geom._softmax(f, g, self.epsilon, v, 1))(y.T)
    res, sgn = res.T, sgn.T  # (k, d), (k, d)
    norm, _ = geom._softmax(f, g, self.epsilon, vec=None, axis=1)  # (k,)

    return sgn * jnp.exp((res - norm[:, None]) / self.epsilon)  # (k, d)

  @property
  def epsilon(self) -> float:
    """TODO(michalk8)."""
    return self._geom.epsilon

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    aux_data["geom"] = self._geom
    return children, aux_data
