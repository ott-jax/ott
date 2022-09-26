import abc

import jax
import jax.numpy as jnp

from ott.geometry import pointcloud

__all__ = ["EntropicMap"]


# TODO(michalk8): better name
class DualPotentials(abc.ABC):

  @abc.abstractmethod
  def transport(self, x: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """TODO(michalk8).

    Args:
      vec: TODO(michalk8)
      axis: TODO(michalk8)

    Returns:
      TODO(michalk8)
    """


class EntropicMap(DualPotentials):
  """TODO(michalk8)."""

  def __init__(
      self, f: jnp.ndarray, g: jnp.ndarray, geom: pointcloud.PointCloud
  ):
    self._f = f
    self._g = g
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
      y, g = self._geom.x, self._f  # (n, d), (n,)
    else:
      y, g = self._geom.y, self._g  # (m, d), (m,)

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
