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
import functools
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp

if TYPE_CHECKING:
  from ott.geometry import costs

__all__ = [
    "safe_log",
    "norm",
    "kl",
    "gen_kl",
    "gen_js",
    "logsumexp",
    "softmin",
    "barycentric_projection",
    "sort_and_argsort",
    "lambertw",
]


def safe_log(  # noqa: D103
    x: jnp.ndarray,
    *,
    eps: Optional[float] = None
) -> jnp.ndarray:
  if eps is None:
    eps = jnp.finfo(x.dtype).tiny
  return jnp.where(x > 0.0, jnp.log(x), jnp.log(eps))


@functools.partial(jax.custom_jvp, nondiff_argnums=[1, 2, 3])
@functools.partial(jax.jit, static_argnames=("ord", "axis", "keepdims"))
def norm(
    x: jnp.ndarray,
    ord: Union[int, str, None] = None,
    axis: Union[None, Sequence[int], int] = None,
    keepdims: bool = False
) -> jnp.ndarray:
  """Computes order ord norm of vector, using `jnp.linalg` in forward pass.

  Evaluations of distances between a vector and itself using translation
  invariant costs, typically  norms, result in functions of the form
  ``lambda x : jnp.linalg.norm(x-x)``. Such functions output `NaN` gradients,
  because they involve computing the derivative of a negative exponent of 0
  (e.g. when differentiating the Euclidean norm, one gets a 0-denominator in the
  expression, see e.g. https://github.com/google/jax/issues/6484 for context).

  While this makes sense mathematically, in the context of optimal transport
  such distances between a point and itself can be safely ignored when they
  contribute to an OT cost (when, for instance, computing Sinkhorn divergences,
  involving computing the OT cost of a point cloud with itself).

  To avoid such `NaN` values, this custom norm implementation uses the
  double-where trick, to avoid having branches that output any `NaN`, and
  safely output a 0 instead.

  Args:
    x: Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
      is None. If both `axis` and `ord` are None, the 2-norm of ``x.ravel``
      will be returned.
    ord: `{non-zero int, jnp.inf, -jnp.inf, 'fro', 'nuc'}`, Order of the norm.
      The default is `None`, which is equivalent to `2.0` for vectors.
    axis: `{None, int, 2-tuple of ints}`, optional. If `axis` is an integer, it
      specifies the axis of `x` along which to compute the vector norms.
      If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices, and
      the matrix norms of these matrices are computed.  If `axis` is None then
      either a vector norm (when `x` is 1-D) or a matrix norm (when `x` is 2-D)
      is returned. The default is None.
    keepdims: If set to True, the axes which are normed over are left in the
      result as dimensions with size one.  With this option the result will
      broadcast correctly against the original `x`.

  Returns:
    float or ndarray, Norm of the matrix or vector(s).
  """
  return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


@norm.defjvp
def norm_jvp(ord, axis, keepdims, primals, tangents):
  """Custom_jvp for norm, that returns 0.0 when evaluated at 0."""
  x, = primals
  x_is_zero = jnp.all(jnp.logical_not(x))
  clean_x = jnp.where(x_is_zero, jnp.ones_like(x), x)
  primals, tangents = jax.jvp(
      functools.partial(jnp.linalg.norm, ord=ord, axis=axis, keepdims=keepdims),
      (clean_x,), tangents
  )
  return primals, jnp.where(x_is_zero, 0.0, tangents)


# TODO(michalk8): add axis argument
def kl(p: jnp.ndarray, q: jnp.ndarray) -> float:
  """Kullback-Leibler divergence."""
  return jnp.vdot(p, (safe_log(p) - safe_log(q)))


def gen_kl(p: jnp.ndarray, q: jnp.ndarray) -> float:
  """Generalized Kullback-Leibler divergence."""
  return jnp.vdot(p, (safe_log(p) - safe_log(q))) + jnp.sum(q) - jnp.sum(p)


# TODO(michalk8): add axis argument
def gen_js(p: jnp.ndarray, q: jnp.ndarray, c: float = 0.5) -> float:
  """Jensen-Shannon divergence."""
  return c * (gen_kl(p, q) + gen_kl(q, p))


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 4))
def logsumexp(  # noqa: D103
    mat, axis=None, keepdims=False, b=None, return_sign=False
):
  return jax.scipy.special.logsumexp(
      mat, axis=axis, keepdims=keepdims, b=b, return_sign=return_sign
  )


@logsumexp.defjvp
def logsumexp_jvp(axis, keepdims, return_sign, primals, tangents):
  """Custom derivative rule for lse that does not blow up with -inf.

  This logsumexp implementation uses the standard jax one in forward mode but
  implements a custom rule to differentiate. Given the preference of jax for
  jvp over vjp, and the fact that this is a simple linear rule, jvp is used.
  This custom differentiation address issues when the output of lse is
  -inf (which corresponds to the case where all inputs in a slice are -inf,
  which happens typically when ``a`` or ``b`` weight vectors have zeros.)

  Although both exp(lse) and its derivative should be 0, automatic
  differentiation returns a NaN derivative because of a -inf - (-inf) operation
  appearing in the definition of centered_exp below. This is corrected in the
  implementation below.

  Args:
    axis: argument from original logsumexp
    keepdims: argument from original logsumexp
    return_sign: argument from original logsumexp
    primals: mat and b, the two arguments against which we differentiate.
    tangents: of same size as mat and b.

  Returns:
    original primal outputs + their tangent.
  """  # noqa: D401
  mat, b = primals
  tan_mat, tan_b = tangents
  lse = logsumexp(mat, axis, keepdims, b, return_sign)
  if return_sign:
    lse, sign = lse
  lse = jnp.where(jnp.isfinite(lse), lse, 0.0)

  if axis is not None:
    centered_exp = jnp.exp(mat - jnp.expand_dims(lse, axis=axis))
  else:
    centered_exp = jnp.exp(mat - lse)

  if b is None:
    res = jnp.sum(centered_exp * tan_mat, axis=axis, keepdims=keepdims)
  else:
    res = jnp.sum(b * centered_exp * tan_mat, axis=axis, keepdims=keepdims)
    res += jnp.sum(tan_b * centered_exp, axis=axis, keepdims=keepdims)
  if return_sign:
    return (lse, sign), (sign * res, jnp.zeros_like(sign))
  return lse, res


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def softmin(
    x: jnp.ndarray,
    gamma: float,
    axis: Optional[Union[int, Sequence[int]]] = None
) -> jnp.ndarray:
  r"""Soft-min operator.

  Args:
    x: Input data.
    gamma: Smoothing parameter :math:`> 0`.
    axis: Axis or axes over which to operate. If ``None``, use flattened input.

  Returns:
    The soft minimum.
  """
  return -gamma * jsp.special.logsumexp(x / -gamma, axis=axis)


softmin.defvjp(
    lambda x, gamma, axis: (softmin(x, gamma, axis), (x / -gamma, axis)),
    lambda axis, res, g: (
        jnp.where(
            jnp.isinf(res[0]), 0.0,
            jax.nn.softmax(res[0], axis=axis) *
            (g if axis is None else jnp.expand_dims(g, axis=axis))
        ), None
    )
)


@functools.partial(jax.vmap, in_axes=[0, 0, None])
def barycentric_projection(
    matrix: jnp.ndarray, y: jnp.ndarray, cost_fn: "costs.CostFn"
) -> jnp.ndarray:
  """Compute the barycentric projection of a matrix.

  Args:
    matrix: a matrix of shape (n, m)
    y: a vector of shape (m,)
    cost_fn: a CostFn instance.

  Returns:
    a vector of shape (n,) containing the barycentric projection of matrix.
  """
  return jax.vmap(
      lambda m, y: cost_fn.barycenter(m, y)[0], in_axes=[0, None]
  )(matrix, y)


def sort_and_argsort(
    x: jnp.array,
    *,
    argsort: bool = False
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
  """Unified function that returns both sort and argsort, if latter needed."""
  if argsort:
    i_x = jnp.argsort(x)
    return x[i_x], i_x
  return jnp.sort(x), None


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def lambertw(
    z: jnp.ndarray, tol: float = 1e-8, max_iter: int = 100
) -> jnp.ndarray:
  """Principal branch of the
  `Lambert W function <https://en.wikipedia.org/wiki/Lambert_W_function>`_.

  This implementation uses Halley's iteration and the global initialization
  proposed in :cite:`iacono:17`, Eq. 20 .

  Args:
     z: Array.
     tol: Tolerance threshold.
     max_iter: Maximum number of iterations.

  Returns:
    The Lambert W evaluated at ``z``.
  """  # noqa: D205

  def initial_iacono(x: jnp.ndarray) -> jnp.ndarray:
    y = jnp.sqrt(1.0 + jnp.e * x)
    num = 1.0 + 1.14956131 * y
    denom = 1.0 + 0.45495740 * jnp.log1p(y)
    return -1.0 + 2.036 * jnp.log(num / denom)

  def cond_fun(container):
    it, converged, _ = container
    return jnp.logical_and(jnp.any(~converged), it < max_iter)

  def halley_iteration(container):
    it, _, w = container

    # modified from `tensorflow_probability`
    f = w - z * jnp.exp(-w)
    delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))

    w_next = w - delta

    not_converged = jnp.abs(delta) <= tol * jnp.abs(w_next)
    return it + 1, not_converged, w_next

  w0 = initial_iacono(z)
  converged = jnp.zeros_like(w0, dtype=bool)

  _, _, w = jax.lax.while_loop(
      cond_fun=cond_fun, body_fun=halley_iteration, init_val=(0, converged, w0)
  )
  return w


@lambertw.defjvp
def _lambertw_jvp(
    tol: float, max_iter: int, primals: Tuple[jnp.ndarray, ...],
    tangents: Tuple[jnp.ndarray, ...]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  z, = primals
  dz, = tangents
  w = lambertw(z, tol=tol, max_iter=max_iter)
  pz = jnp.where(z == 0.0, 1.0, w / ((1.0 + w) * z))
  return w, pz * dz
