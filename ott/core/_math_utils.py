from typing import Optional, Union

import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.scipy as jsp

__all__ = ["safe_log", "kl"]

Sparse_t = Union[jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]


def safe_log(x: jnp.ndarray, *, eps: Optional[float] = None) -> jnp.ndarray:
  if eps is None:
    eps = jnp.finfo(x.dtype).tiny
  return jnp.where(x > 0., jnp.log(x), jnp.log(eps))


def kl(q1: jnp.ndarray, q2: jnp.ndarray) -> float:
  res_1 = -jsp.special.entr(q1)
  res_2 = q1 * safe_log(q2)
  return jnp.sum(res_1 - res_2)


def sparse_scale(c: float, mat: Sparse_t) -> Sparse_t:
  """Scale a sparse matrix by a constant."""
  if isinstance(mat, jesp.BCOO):
    # most feature complete, defer to original impl.
    return c * mat
  (data, *children), aux_data = mat.tree_flatten()
  return type(mat).tree_unflatten(aux_data, [c * data] + children)
