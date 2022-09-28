from typing import Any, Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from typing_extensions import Literal

from ott.geometry import pointcloud

__all__ = ["DualPotentials", "EntropicPotentials"]
Potential_t = Callable[[jnp.ndarray], float]


@jtu.register_pytree_node_class
class DualPotentials:
  r"""The Kantorovich dual potentials :math:`f` and :math:`g`.

  :math:`\nabla g` transports the source distribution to the target distribution
  and :math:`\nabla f` the target distribution to the source distribution.

  Args:
    f: The first dual potential function.
    g: The second dual potential function.
  """

  def __init__(self, f: Potential_t, g: Potential_t):
    self._f = f
    self._g = g

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport points using the dual potentials :attr:`f` and :attr:`g`.

    Args:
      vec: Points to transport, array of shape ``[n, d]``.
      forward: Whether to transport the points from source  to the target
        distribution or vice-versa.

    Returns:
      The transported points.
    """
    vec = jnp.atleast_2d(vec)
    return self._grad_g(vec) if forward else self._grad_f(vec)

  def distance(self, src: jnp.ndarray, tgt: jnp.ndarray) -> float:
    """Using the dual potentials, evaluate the 2-Wasserstein distance \
      between the samples.

    Uses the eq. 5 from :cite:`makkuva:20`.

    Args:
      src: Samples from the source distribution, array of shape ``[n, d]``.
      tgt: Samples from the target distribution, array of shape ``[m, d]``.

    Returns:
      Wasserstein distance :math:`W^2_2`, assuming :math:`|x-y|^2`
      as the ground distance.
    """
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)
    f = jax.vmap(self.f)

    grad_g_y = self._grad_g(src)
    term1 = -jnp.mean(f(tgt))
    term2 = -jnp.mean(jnp.sum(src * grad_g_y, axis=-1) - f(grad_g_y))

    C = jnp.mean(jnp.sum(src ** 2, axis=-1)) + \
        jnp.mean(jnp.sum(tgt ** 2, axis=-1))

    # compute final wasserstein distance assuming ground metric |x-y|^2,
    # thus an additional multiplication by 2
    return 2. * (term1 + term2) + C

  @property
  def f(self) -> Potential_t:
    """The first dual potential."""
    return self._f

  @property
  def g(self) -> Potential_t:
    """The second dual potential."""
    return self._g

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`f`."""
    return jax.vmap(jax.grad(self.f, argnums=0))

  @property
  def _grad_g(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`g`."""
    return jax.vmap(jax.grad(self.g, argnums=0))

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self.f, self.g], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "DualPotentials":
    return cls(*children, **aux_data)


@jtu.register_pytree_node_class
class EntropicPotentials(DualPotentials):
  """Entropic map estimator :cite:`pooladian:21`.

  See also :meth:`from_potentials` on how to instantiate it using potentials
  represented as a :class:`jax.numpy.ndarray`, in which they are assumed to be
  tied to the sample points in a :class:`~ott.geometry.pointcloud.PointCloud`.

  Args:
    f: The first dual potential function.
    g: The second dual potential function.
  """

  @classmethod
  def from_sinkhorn_potentials(
      cls, f: jnp.ndarray, g: jnp.ndarray, geom: pointcloud.PointCloud
  ) -> "EntropicPotentials":
    """Dual potential functions computed from finite samples as introduced in \
      :cite:`pooladian:21`.

    Args:
      f: The first dual potential vector of shape ``[n,]``.
      g: The second dual potential vector of shape ``[m,]``.
      geom: Geometry used to compute the dual potentials using
        :class:`~ott.core.sinkhorn.Sinkhorn`.

    Returns:
      The estimator, built with pair of dual potential functions.
    """
    f = cls._create_potential_function(geom, f, kind="f")
    g = cls._create_potential_function(geom, g, kind="g")

    return cls(f, g)

  @staticmethod
  def _create_potential_function(
      geom: pointcloud.PointCloud, potential: jnp.ndarray, *, kind: Literal["f",
                                                                            "g"]
  ) -> Callable[[jnp.ndarray], float]:

    def callback(x: jnp.ndarray) -> float:
      cost = pointcloud.PointCloud(
          jnp.atleast_2d(x), y, epsilon=eps
      ).cost_matrix
      return 0.5 * eps * jsp.special.logsumexp((potential - cost) / eps)

    y = geom.x if kind == "f" else geom.y
    eps = geom.epsilon

    return callback

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    return vec + super().transport(vec, forward=forward)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    aux_data["geom"] = self._geom
    return children, aux_data
