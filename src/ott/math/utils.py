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
from typing import TYPE_CHECKING, Optional, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp

if TYPE_CHECKING:
  from ott.geometry import costs

__all__ = [
    "safe_log", "kl", "js", "sparse_scale", "logsumexp",
    "barycentric_projection"
]

# TODO(michalk8): move to typing.py when refactoring types
Sparse_t = Union[jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]


def safe_log(x: jnp.ndarray, *, eps: Optional[float] = None) -> jnp.ndarray:
  if eps is None:
    eps = jnp.finfo(x.dtype).tiny
  return jnp.where(x > 0., jnp.log(x), jnp.log(eps))


def kl(p: jnp.ndarray, q: jnp.ndarray) -> float:
  """Kullback-Leilbler divergence."""
  return jnp.vdot(p, (safe_log(p) - safe_log(q)))


def js(p: jnp.ndarray, q: jnp.ndarray, *, c: float = 0.5) -> float:
  """Jensen-Shannon divergence."""
  return c * (kl(p, q) + kl(q, p))


def sparse_scale(c: float, mat: Sparse_t) -> Sparse_t:
  """Scale a sparse matrix by a constant."""
  if isinstance(mat, jesp.BCOO):
    # most feature complete, defer to original impl.
    return c * mat
  (data, *children), aux_data = mat.tree_flatten()
  return type(mat).tree_unflatten(aux_data, [c * data] + children)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 4))
def logsumexp(mat, axis=None, keepdims=False, b=None, return_sign=False):
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
  centered_exp = jnp.exp(mat - jnp.expand_dims(lse, axis=axis))

  if b is None:
    res = jnp.sum(centered_exp * tan_mat, axis=axis, keepdims=keepdims)
  else:
    res = jnp.sum(b * centered_exp * tan_mat, axis=axis, keepdims=keepdims)
    res += jnp.sum(tan_b * centered_exp, axis=axis, keepdims=keepdims)
  if return_sign:
    return (lse, sign), (sign * res, jnp.zeros_like(sign))
  else:
    return lse, res


@functools.partial(jax.vmap, in_axes=[0, 0, None])
def barycentric_projection(
    matrix: jnp.ndarray, y: jnp.ndarray, cost_fn: "costs.CostFn"
) -> jnp.ndarray:
  return jax.vmap(cost_fn.barycenter, in_axes=[0, None])(matrix, y)
