import abc
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import pointcloud

__all__ = ["EntropicMap", "DualPotentials"]
Potential_t = Union[jnp.ndarray, Callable[[jnp.ndarray], float]]


@jtu.register_pytree_node_class
class BaseDualPotentials(abc.ABC):
  """Base class holding the Kantorovich dual potentials.

  Args:
    f: The first dual potential.
    g: The second dual potential.

  Notes:
    Both potentials can be represented either as an :class:`jax.numpy.ndarray`
    or as a function taking :class:`jax.numpy.ndarray` and returning
    :class:`float`.
  """

  def __init__(self, f: Potential_t, g: Potential_t):
    self._f = f
    self._g = g

  @abc.abstractmethod
  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport points using the dual potentials :attr:`f` and :attr:`g`.

    Args:
      vec: Points to transport, array of shape ``[n, d]``.
      forward: Whether to transport the points from source  to the target
        distribution or vice-versa.

    Returns:
      The transported points.
    """

  @property
  def f(self) -> Potential_t:
    """The first dual potential."""
    return self._f

  @property
  def g(self) -> Potential_t:
    """The second dual potential."""
    return self._g

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self.f, self.g], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "BaseDualPotentials":
    return cls(*children, **aux_data)


@jtu.register_pytree_node_class
class EntropicMap(BaseDualPotentials):
  """Entropic map estimator :cite:`pooladian:21`.

  Args:
    f: The first dual potential.
    g: The second dual potential.
    geom: Geometry associated with the dual potentials.
  """

  def __init__(
      self, f: jnp.ndarray, g: jnp.ndarray, geom: pointcloud.PointCloud
  ):
    assert geom.is_squared_euclidean, \
        "Entropic map is only implemented for squared Euclidean cost."
    super().__init__(f, g)
    self._geom = geom

  def transport(
      self,
      vec: jnp.ndarray,
      forward: bool = True,
  ) -> jnp.ndarray:
    vec = jnp.atleast_2d(vec)
    f = jnp.zeros(vec.shape[0])  # (k, d)
    if forward:
      y, g = self._geom.y, self.g  # (m, d), (m,)
    else:
      y, g = self._geom.x, self.f  # (n, d), (n,)

    geom = pointcloud.PointCloud(vec, y, epsilon=self.epsilon)  # (k, {n, m})
    # (d, k), (d, k)
    res, sgn = jax.vmap(lambda v: geom._softmax(f, g, self.epsilon, v, 1))(y.T)
    res, sgn = res.T, sgn.T  # (k, d), (k, d)
    norm, _ = geom._softmax(f, g, self.epsilon, vec=None, axis=1)  # (k,)

    return sgn * jnp.exp((res - norm[:, None]) / self.epsilon)  # (k, d)

  @property
  def epsilon(self) -> float:
    """Epsilon regularizer."""
    return self._geom.epsilon

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    aux_data["geom"] = self._geom
    return children, aux_data


@jtu.register_pytree_node_class
class DualPotentials(BaseDualPotentials):
  r"""The Kantorovich dual potentials :math:`f` and :math:`g`.

  :math:`\nabla g` transports the source distribution to the target distribution
  and :math:`\nabla f` the target distribution to the source distribution.

  Args:
    f: The first dual potential function.
    g: The second dual potential function.
  """

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    vec = jnp.atleast_2d(vec)
    fn = self._grad_g if forward else self._grad_f
    return fn(vec)

  def distance(self, src: jnp.ndarray, tgt: jnp.ndarray) -> float:
    r"""Given the dual potentials functions, compute the transport distance.

    Args:
      src: Samples from the source distribution, array of shape ``[n, d]``.
      tgt: Samples from the target distribution, array of shape ``[m, d]``.

    Returns:
      Wasserstein distance :math:`W^2_2`, assuming :math:`|x-y|^2`
      as the ground distance.
    """
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)

    f_t = self.f(tgt)
    grad_g_s = self._grad_g(src)
    f_grad_g_s = self.f(grad_g_s)
    s_dot_grad_g_s = jnp.sum(src * grad_g_s, axis=-1)

    s_sq = 0.5 * jnp.sum(src ** 2, axis=-1)
    t_sq = 0.5 * jnp.sum(tgt ** 2, axis=-1)

    # compute final wasserstein distance assuming ground metric |x-y|^2
    # thus an additional multiplication by 2
    return 2. * jnp.mean(f_grad_g_s - f_t - s_dot_grad_g_s + t_sq + s_sq)

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`f`."""
    return jax.vmap(jax.grad(self.f, argnums=0))

  @property
  def _grad_g(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`g`."""
    return jax.vmap(jax.grad(self.g, argnums=0))
