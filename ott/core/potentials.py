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
  r"""The Kantorovich dual potential functions :math:`f` and :math:`g`.

  :math:`f` and :math:`g` are a pair of functions, candidates for the dual
  OT Kantorovich problem, supposedly optimal for a given pair of measures.

  Args:
    f: The first dual potential function.
    g: The second dual potential function.
  """

  def __init__(self, f: Potential_t, g: Potential_t):
    self._f = f
    self._g = g

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport ``vec`` according to Benamou-Brenier formula.

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
    """Evaluate 2-Wasserstein distance between samples using dual potentials.

    Uses Eq. 5 from :cite:`makkuva:20`.

    Args:
      src: Samples from the source distribution, array of shape ``[n, d]``.
      tgt: Samples from the target distribution, array of shape ``[m, d]``.

    Returns:
      Wasserstein distance :math:`W^2_2`, assuming :math:`|x-y|^2` as the
      ground distance.
    """
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)
    f = jax.vmap(self.f)

    grad_g_y = self._grad_g(src)
    term1 = -jnp.mean(f(tgt))
    term2 = -jnp.mean(jnp.sum(src * grad_g_y, axis=-1) - f(grad_g_y))

    C = jnp.mean(jnp.sum(src ** 2, axis=-1)) + \
        jnp.mean(jnp.sum(tgt ** 2, axis=-1))

    # compute the final Wasserstein distance assuming ground metric |x-y|^2,
    # thus an additional multiplication by 2
    return 2. * (term1 + term2) + C

  @property
  def f(self) -> Potential_t:
    """The first dual potential function."""
    return self._f

  @property
  def g(self) -> Potential_t:
    """The second dual potential function."""
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
    return [self._f, self._g], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "DualPotentials":
    return cls(*children, **aux_data)


@jtu.register_pytree_node_class
class EntropicPotentials(DualPotentials):
  """Dual potential functions from finite samples :cite:`pooladian:21`.

  Args:
    f: The first dual potential vector of shape ``[n,]``.
    g: The second dual potential vector of shape ``[m,]``.
    geom: Geometry used to compute the dual potentials using
      :class:`~ott.core.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self, f: jnp.ndarray, g: jnp.ndarray, geom: pointcloud.PointCloud
  ):
    assert geom.is_squared_euclidean, \
        "Entropic map is only implemented for squared Euclidean cost."
    n, m = geom.shape
    assert f.shape == (n,), \
        f"Expected `f` to be of shape `{n,}`, found `{f.shape}`."
    assert g.shape == (m,), \
        f"Expected `g` to be of shape `{m,}`, found `{g.shape}`."

    # we pass directly the arrays and override the properties
    # since only the properties need to be callable
    super().__init__(f, g)
    self._geom = geom

  @property
  def f(self) -> Potential_t:
    return self._create_potential_function(kind="f")

  @property
  def g(self) -> Potential_t:
    return self._create_potential_function(kind="g")

  def _create_potential_function(
      self, *, kind: Literal["f", "g"]
  ) -> Potential_t:

    def callback(x: jnp.ndarray) -> float:
      cost = pointcloud.PointCloud(
          jnp.atleast_2d(x), y, epsilon=eps
      ).cost_matrix
      return 0.5 * eps * jsp.special.logsumexp((potential - cost) / eps)

    eps = self.epsilon
    if kind == "f":
      potential = self._f
      y = self._geom.x
    else:
      potential = self._g
      y = self._geom.y

    return callback

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    return vec + super().transport(vec, forward=forward)

  @property
  def epsilon(self) -> float:
    """Entropy regularizer."""
    return self._geom.epsilon

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    return children + [self._geom], aux_data
