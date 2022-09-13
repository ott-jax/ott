from typing import Optional, Union

import jax.experimental.sparse as jesp
import jax.numpy as jnp

__all__ = ["safe_log", "kl", "js"]

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
