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
import abc
import functools
from typing import Any, Callable, Optional, Tuple, Union

import lineax as lx

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

__all__ = [
    "PostComposition",
    "Regularization",
    "Orthogonal",
    "Quadratic",
    "L1",
    "SqL2",
    "STVS",
    "SqKOverlap",
]


class ProximalOperator(abc.ABC):
  """Proximal operator base class."""

  @abc.abstractmethod
  def __call__(self, x: jnp.ndarray) -> float:
    """Function.

    Args:
      x: Array of shape ``[d,]``.

    Returns:
      The value.
    """

  @abc.abstractmethod
  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:
    """Proximal operator.

    Args:
      v: Array of shape ``[d,]``.
      tau: Positive weight.

    Returns:
        The prox of ``v``.
    """

  def prox_dual(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:
    r"""Proximal operator of the convex conjugate.

    Uses Moreau's decomposition:

    .. math::
        v = \prox_{\tau f} \left(v\right) +
        \tau \prox_{\frac{1}{\tau} f^*} \left(\frac{v}{\tau}\right)

    Args:
      v: Array of shape ``[d,]``.
      tau: Positive weight.

    Returns:
      The prox dual of ``v``.
    """
    return v - tau * self.prox(v / tau, 1.0 / tau)

  def moreau_envelope(self, x: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:
    r"""Moreau Envelope.

    Uses Remark 12.24 from :cite:`bauschke:17`:

    .. math::
      {^\tau}f\left(x\right) = f\left(\prox_{\tau f}\left(x\right)\right)
      + \frac{1}{2\tau}\|x - \prox_{\tau f}\left(x\right)|_2^2

    Args:
      x: Array of shape ``[d,]``.
      tau: Positive weight.

    Returns:
      The Moreau Envelope of ``x``.
    """
    prox_x = self.prox(x, tau)
    return self(prox_x) + (1.0 / (2.0 * tau)) * jnp.sum((x - prox_x) ** 2)

  def tree_flatten(self):  # noqa: D102
    return (), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)


@jtu.register_pytree_node_class
class PostComposition(ProximalOperator):
  r"""Postcomposition operator :math:`\alpha f\left(x\right) + b`.

  Args:
    f: Function :math:`f`.
    alpha: Scaling factor.
    b: Offset.
  """

  def __init__(self, f: ProximalOperator, alpha: float = 1.0, b: float = 0.0):
    super().__init__()
    self.f = f
    self.alpha = alpha
    self.b = b

  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    return self.alpha * self.f(x) + self.b

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:  # noqa: D102
    return self.f.prox(v, tau * self.alpha)

  def tree_flatten(self):  # noqa: D102
    return (self.f, self.alpha, self.b), {}


@jtu.register_pytree_node_class
class Regularization(ProximalOperator):
  r"""Regularization operator :math:`f\left(x\right) + \frac{\rho}{2}\|x - a\|_2^2`.

  Args:
    f: Function :math:`f`.
    a: Offset :math:`a`. If :obj:`None`, use array of 0s.
    rho: Scaling factor.
  """  # noqa: E501

  def __init__(
      self,
      f: ProximalOperator,
      a: Optional[jnp.ndarray] = None,
      rho: float = 1.0,
  ):
    super().__init__()
    self.f = f
    self.a = a
    self.rho = rho

  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    norm = jnp.sum(x ** 2) if self.a is None else jnp.sum((x - self.a) ** 2)
    return self.f(x) + (0.5 * self.rho) * norm

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:  # noqa: D102
    tau_tilde = tau / (1.0 + tau * self.rho)
    # (tau_tilde / tau) * v
    vv = 1.0 / (1 + tau * self.rho) * v
    if self.a is not None:
      vv = vv + (self.rho * tau_tilde) * self.a
    # section 2.2 of :cite:`parikh:14`
    return self.f.prox(vv, tau_tilde)

  def tree_flatten(self):  # noqa: D102
    return (self.f, self.a, self.rho), {}


@jtu.register_pytree_node_class
class Orthogonal(ProximalOperator):
  r"""Orthogonal operator :math:`f\left( Ax \right) + b`.

  The computation of the :meth:`prox` uses the
  Proposition 11 of :cite:`combettes:07`.

  Args:
    f: Function :math:`f` applied to :math:`Ax`.
    A: Linear operator :math:`A`.
    b: Offset :math:`b`. If :obj:`None`, use array of 0s.
    nu: Value for which :math:`AA^T = \nu I` holds.
  """

  def __init__(
      self,
      f: ProximalOperator,
      A: Optional[Union[jnp.ndarray, lx.AbstractLinearOperator]],
      b: Optional[jnp.ndarray] = None,
      nu: float = 1.0,
  ):
    assert nu > 0.0, nu
    super().__init__()
    self.f = f
    # AA^T = alpha I
    self.A = lx.MatrixLinearOperator(A) if isinstance(A, jnp.ndarray) else A
    self.b = b
    self.nu = nu

  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    z = self.A.mv(x)
    if self.b is not None:
      z = z + self.b
    return self.f(z)

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:  # noqa: D102
    w = self.A.mv(v)
    if self.b is None:
      tmp = self.f.prox(w, tau * self.nu)
    else:
      tmp = self.f.prox(w + self.b, tau * self.nu) - self.b
    return v - (1.0 / self.nu) * (self.A.T.mv(w - tmp))

  @property
  def is_fully_orthogonal(self) -> bool:
    r"""Whether :math:`\nu = 1`."""
    return self.nu == 1.0

  def tree_flatten(self):  # noqa: D102
    return (self.f, self.A, self.b), {"nu": self.nu}


@jtu.register_pytree_node_class
class Quadratic(ProximalOperator):
  r"""Quadratic operator :math:`\frac{1}{2} \left<x, Q x\right> + b`.

  The matrix :math:`Q` is defined as:

  - :math:`Q := A` if not factored and not an orthogonal complement.
  - :math:`Q := A^{\perp}` if not factored and a complement.
  - :math:`Q := A^TA` if factored and not a complement.
  - :math:`Q := \left(A^{\perp}\right)^TA^{\perp}` if factored and
    a complement.

  Args:
    A: Linear operator :math:`A`. If :obj:`None`, use identity.
    b: Offset :math:`b`. If :obj:`None`, use array of 0s.
    is_complement: Whether to regularize in the orthogonal complement of
      :math:`A`, defined as :math:`A^{\perp} := I - A^T (AA^T)^{-1} A`.
    is_orthogonal: Whether :math:`AA^T = I`.
    is_factor: Whether to factor the matrix :math:`Q` as mentioned above.
    solver: Linear solver. If :obj:`None`, use :func:`lineax.linear_solve`.
  """

  def __init__(
      self,
      A: Optional[Union[jnp.ndarray, lx.AbstractLinearOperator]] = None,
      b: Optional[jnp.ndarray] = None,
      *,
      is_complement: bool = False,
      is_orthogonal: bool = False,
      is_factor: bool = False,
      solver: Optional[Callable[[lx.AbstractLinearOperator, jnp.ndarray],
                                jnp.ndarray]] = None,
  ):
    super().__init__()
    self.A = lx.MatrixLinearOperator(A) if isinstance(A, jnp.ndarray) else A
    self.b = b
    self._is_complement = is_complement
    self._is_orthogonal = is_orthogonal
    self._is_factor = is_factor
    self._solver = solver

  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    Q = self.Q
    y = 0.5 * (jnp.dot(x, x) if Q is None else jnp.dot(x, Q.mv(x)))
    return y if self.b is None else (y + jnp.dot(x, self.b))

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:  # noqa: D102
    # section 6.1.1 in :cite:`parikh:14`
    Q = self.Q
    b = v if self.b is None else (v - tau * self.b)

    if Q is None:
      return (1.0 / (1.0 + tau)) * b

    iden = lx.IdentityLinearOperator(Q.out_structure())
    if self.is_factor:  # use matrix inversion lemma
      if self.is_complement:
        # eq. 14 in :cite:`klein:24`
        # A_comp = I - A^T(AA^T)^{-1}A
        # prox(v) = (I + tau A_comp^T A_comp)^{-1} (v - tau * b)
        op = iden + tau * (iden - self.A_comp)
        return (1.0 / (1.0 + tau)) * op.mv(b)
      if self.is_orthogonal:
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        op = iden - (tau / (1.0 + tau)) * (self.A.T @ self.A)
        return op.mv(b)

    A = iden + tau * Q

    if self._solver is None:
      # use default solver
      return lx.linear_solve(A, b).value
    return self._solver(A, b)

  @property
  def A_comp(self) -> Optional[lx.AbstractLinearOperator]:
    r"""Orthogonal complement :math:`A^{\perp}` of :math:`A`."""
    return _complement(
        self.A, self.is_orthogonal
    ) if self.is_complement else None

  @property
  def is_complement(self) -> bool:
    r"""Whether :attr:`Q` is defined using :math:`A_{\perp}` or :math:`A`."""
    return self.A is not None and self._is_complement

  @property
  def is_factor(self) -> bool:
    r"""Whether :attr:`Q` is factored."""
    return self.A is not None and self._is_factor

  @property
  def is_orthogonal(self) -> bool:
    r"""Whether :attr:`AA^T = I`."""
    return self.A is not None and self._is_orthogonal

  @property
  def Q(self) -> Optional[lx.AbstractLinearOperator]:
    r"""Linear operator :math:`Q`."""
    Q = self.A_comp if self.is_complement else self.A
    if Q is None:
      return None
    return (Q.T @ Q) if self.is_factor else Q

  def tree_flatten(self):  # noqa: D102
    return (self.A, self.b), {
        "is_complement": self.is_complement,
        "is_orthogonal": self.is_orthogonal,
        "is_factor": self.is_factor,
        "solver": self._solver
    }


@jtu.register_pytree_node_class
class L1(ProximalOperator):
  r"""L1-norm regularizer :math:`\ell_1`."""

  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    return jnp.linalg.norm(x, ord=1)

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:  # noqa: D102
    return jnp.sign(v) * jax.nn.relu(jnp.abs(v) - tau)


@jtu.register_pytree_node_class
class SqL2(ProximalOperator):
  r"""Squared L2-norm regularizer :math:`\ell_2^2`.

  Args:
    A: Linear operator :math:`A` in :math:`\frac{1}{2} \left<x, A^TAx\right>`.
      If :obj:`None`, use identity.
    kwargs: Keyword arguments for :class:`Quadratic`.
  """

  def __init__(
      self,
      A: Optional[Union[jnp.ndarray, lx.AbstractLinearOperator]] = None,
      **kwargs: Any,
  ):
    super().__init__()
    self.f = Quadratic(A, is_factor=True, **kwargs)
    self._init_kwargs = kwargs

  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    return self.f(x)

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:  # noqa: D102
    return self.f.prox(v, tau)

  def tree_flatten(self):  # noqa: D102
    return (self.f.A,), self._init_kwargs


@jtu.register_pytree_node_class
class STVS(ProximalOperator):
  r"""Soft thresholding operator with vanishing shrinkage regularizer :cite:`schreck:15`.

  The operator is defined as:

  .. math::
    \gamma^2 \mathbf{1}_d^T \left(\sigma(x) -
    \frac{1}{2} \exp\left(-2\sigma(x)\right) + \frac{1}{2}\right)

  where :math:`\sigma(x) := \text{asinh}\left(\frac{x}{2\gamma}\right)`.

  Args:
    gamma: Strength of the regularization.
  """  # noqa: E501

  def __init__(self, gamma: float = 1.0):
    super().__init__()
    self.gamma = gamma

  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    # Lemma 2.1 of `schreck:15`
    u = jnp.arcsinh(jnp.abs(x) / (2.0 * self.gamma))
    y = u - 0.5 * jnp.exp(-2.0 * u)
    return self.gamma ** 2 * jnp.sum(y + 0.5)  # make positive

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> jnp.ndarray:  # noqa: D102
    s = (tau * self.gamma) ** 2
    return jnp.where(v ** 2 <= s, 0.0, v - s / jnp.where(v == 0.0, 1.0, v))

  def tree_flatten(self):  # noqa: D102
    return (self.gamma,), {}


@jtu.register_pytree_node_class
class SqKOverlap(ProximalOperator):
  r"""Squared k-overlap norm regularizer :cite:`argyriou:12`.

  The regularizer is defined as:

  .. math::
    \frac{1}{2} \left(\|x\|_k^{\text{ov}}\right)^2

  where :math:`\left(\|x\|_k^{\text{ov}}\right)^2` is the squared k-overlap
  norm, defined in :cite:`argyriou:12`, def. 2.1.

  Args:
    k: Number of groups in :math:`[0, d)`, where :math:`d` is the dimensionality
      of the data.
  """

  def __init__(self, k: int):
    super().__init__()
    self.k = k

  def __call__(self, z: jnp.ndarray) -> float:  # noqa: D102
    # Prop 2.1 in :cite:`argyriou:12`
    k = self.k
    top_w = jax.lax.top_k(jnp.abs(z), k)[0]  # Fetch largest k values
    top_w = jnp.flip(top_w)  # Sort k-largest from smallest to largest
    # sum (dim - k) smallest values
    sum_bottom = jnp.sum(jnp.abs(z)) - jnp.sum(top_w)
    cumsum_top = jnp.cumsum(top_w)
    # Cesaro mean of top_w (each term offset with sum_bottom).
    cesaro = sum_bottom + cumsum_top
    cesaro /= jnp.arange(k) + 1
    # Choose first index satisfying constraint in Prop 2.1
    lower_bound = cesaro - top_w >= 0
    # Last upper bound is always True.
    upper_bound = jnp.concatenate(((top_w[1:] - cesaro[:-1]
                                    > 0), jnp.array((True,))))
    r = jnp.argmax(lower_bound * upper_bound)
    s = jnp.sum(jnp.where(jnp.arange(k) < k - r - 1, jnp.flip(top_w) ** 2, 0))

    return 0.5 * (s + (r + 1) * cesaro[r] ** 2)

  def prox(self, v: jnp.ndarray, tau: float = 1.0) -> float:  # noqa: D102

    @functools.partial(jax.vmap, in_axes=[0, None, None])
    def find_indices(r: int, l: jnp.ndarray,
                     z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

      @functools.partial(jax.vmap, in_axes=[None, 0, None])
      def inner(r: int, l: int,
                z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        i = k - r - 1
        res = jnp.sum(z * ((i <= ixs) & (ixs < l)))
        res /= l - k + (beta + 1) * r + beta + 1

        cond1_left = jnp.logical_or(i == 0, (z[i - 1] / beta + 1) > res)
        cond1_right = res >= (z[i] / (beta + 1))
        cond1 = jnp.logical_and(cond1_left, cond1_right)

        cond2_left = z[l - 1] > res
        cond2_right = jnp.logical_or(l == d, res >= z[l])
        cond2 = jnp.logical_and(cond2_left, cond2_right)

        return res, cond1 & cond2

      return inner(r, l, z)

    # Alg. 1 of :cite:`argyriou:12`
    k, d, beta = self.k, v.shape[-1], 1.0 / tau

    ixs = jnp.arange(d)
    v, sgn = jnp.abs(v), jnp.sign(v)
    z_ixs = jnp.argsort(v)[::-1]
    z_sorted = v[z_ixs]

    # (k, d - k + 1)
    T, mask = find_indices(jnp.arange(k), jnp.arange(k, d + 1), z_sorted)
    (r,), (l,) = jnp.where(mask, size=1)  # size=1 for jitting
    T = T[r, l]

    q1 = (beta / (beta + 1)) * z_sorted * (ixs < (k - r - 1))
    q2 = (z_sorted - T) * jnp.logical_and((k - r - 1) <= ixs, ixs < (l + k))
    q = q1 + q2

    # change sign and reorder
    return sgn * q[jnp.argsort(z_ixs.astype(float))]

  def tree_flatten(self):  # noqa: D102
    return (), {"k": self.k}


def _invert(A: lx.AbstractLinearOperator) -> lx.MatrixLinearOperator:
  d = A.out_size()
  b = jnp.zeros(d)

  solve_fn = jax.vmap(lambda ix: lx.linear_solve(A, b.at[ix].set(1.0)).value)
  inv = solve_fn(jnp.arange(d))
  return lx.MatrixLinearOperator(inv)


@functools.partial(jax.jit, static_argnums=1)
def _complement(
    A: lx.AbstractLinearOperator, is_orthogonal: bool
) -> lx.AbstractLinearOperator:
  iden = lx.IdentityLinearOperator(A.in_structure())
  if is_orthogonal:
    # AA^T = I
    return iden - (A.T @ A)

  A_inv = _invert(lx.TaggedLinearOperator(A @ A.T, tags={lx.symmetric_tag}))
  return iden - A.T @ (A_inv @ A)
