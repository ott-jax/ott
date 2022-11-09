from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from typing_extensions import Literal

from ott.geometry import costs, pointcloud

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
    cost_fn: The cost function used to solve the OT problem.
    cor: whether the duals solve the problem in distance form, or correlation
      form (as used for instance for ICNNs, see e.g. top right of p.3 in
      http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf)
  """

  def __init__(
      self,
      f: Potential_t,
      g: Potential_t,
      *,
      cost_fn: Optional[costs.CostFn] = None,
      cor: bool = False
  ):
    self._f = f
    self._g = g
    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    self._cor = cor

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport ``vec`` according to Brenier formula.

    Theorem 1.17 in http://math.univ-lyon1.fr/~santambrogio/OTAM-cvgmt.pdf
    for case h(.) = ||.||^2, ∇h(.) = 2 .,
    h*(.) = ||.||^2 / 4, [∇h*](.) = [∇h]^-1(.) = 0.5 * .

    or, when solved in correlation form, as ∇g for forward, ∇f for backward.

    Args:
      vec: Points to transport, array of shape ``[n, d]``.
      forward: Whether to transport the points from source  to the target
        distribution or vice-versa.

    Returns:
      The transported points.
    """
    vec = jnp.atleast_2d(vec)
    if self._cor and isinstance(self.cost_fn, costs.SqEuclidean):
      return self._grad_g(vec) if forward else self._grad_f(vec)
    grad_h_inv = jax.vmap(jax.grad(self.cost_fn.h_legendre))
    if forward:
      return vec - grad_h_inv(self._grad_f(vec))
    else:
      return vec - grad_h_inv(self._grad_g(vec))

  def distance(self, src: jnp.ndarray, tgt: jnp.ndarray) -> float:
    """Evaluate 2-Wasserstein distance between samples using dual potentials.

    Uses Eq. 5 from :cite:`makkuva:20` when given in cor form, direct estimation
    by integrating dual function against points when using dual form.

    Args:
      src: Samples from the source distribution, array of shape ``[n, d]``.
      tgt: Samples from the target distribution, array of shape ``[m, d]``.

    Returns:
      Wasserstein distance.
    """
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)

    f = jax.vmap(self.f)

    if self._cor:
      grad_g_y = self._grad_g(tgt)
      term1 = -jnp.mean(f(src))
      term2 = -jnp.mean(jnp.sum(tgt * grad_g_y, axis=-1) - f(grad_g_y))

      C = jnp.mean(jnp.sum(src ** 2, axis=-1))
      C += jnp.mean(jnp.sum(tgt ** 2, axis=-1))
      return 2. * (term1 + term2) + C
    else:
      g = jax.vmap(self.g)
      C = jnp.mean(f(src))
      C += jnp.mean(g(tgt))
      return C

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
    return [self._f, self._g], {"cor": self._cor}

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
      self, f: jnp.ndarray, g: jnp.ndarray, geom: pointcloud.PointCloud,
      a: jnp.ndarray, b: jnp.ndarray
  ):
    n, m = geom.shape
    assert f.shape == (n,) and a.shape == (n,), \
        f"Expected `f` and `a` to be of shape `{n,}`, found `{f.shape}`."
    assert g.shape == (m,) and b.shape == (m,), \
        f"Expected `g` and `b` to be of shape `{m,}`, found `{g.shape}`."

    # we pass directly the arrays and override the properties
    # since only the properties need to be callable
    super().__init__(f, g, cost_fn=geom.cost_fn, cor=False)
    self._geom = geom
    self._a = a
    self._b = b

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
          jnp.atleast_2d(x), y, cost_fn=self._geom.cost_fn
      ).cost_matrix
      z = (potential - cost) / epsilon
      lse = -epsilon * jsp.special.logsumexp(z, b=prob_weights, axis=-1)
      return jnp.squeeze(lse)

    epsilon = self.epsilon
    if kind == "g":
      # When seeking to evaluate 2nd potential function, 1st set of potential
      # values and support should be used,
      # see proof of Prop. 2 in https://arxiv.org/pdf/2109.12004.pdf
      potential = self._f
      y = self._geom.x
      prob_weights = self._a
    else:
      potential = self._g
      y = self._geom.y
      prob_weights = self._b

    return callback

  @property
  def epsilon(self) -> float:
    """Entropy regularizer."""
    return self._geom.epsilon

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self._f, self._g, self._geom, self._a, self._b], {}
