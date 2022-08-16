import abc
import functools
from typing import Any, Dict, Mapping, NamedTuple, Sequence, Tuple

import jax
import jax.experimental.host_callback as hcb
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.sparse as sp
import sksparse.cholmod

__all__ = ["DenseCholeskyDecomposition", "SparseCholeskyDecomposition"]


class CSC(NamedTuple):
  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray


@jax.tree_util.register_pytree_node_class
class CholeskyDecomposition(abc.ABC):
  LOWER = True

  @functools.partial(jax.jit, static_argnums=0)
  def __new__(cls, A: jnp.ndarray) -> "CholeskyDecomposition":
    obj = super().__new__(cls)
    obj._A = A
    obj._L = obj._decompose(jax.lax.stop_gradient(A))
    return obj

  def __call__(self, b: jnp.ndarray) -> jnp.ndarray:
    return self._solve(self.L, b)

  @abc.abstractmethod
  def _decompose(self, A: jnp.ndarray) -> jnp.ndarray:
    pass

  @abc.abstractmethod
  def _solve(self, L: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    pass

  @property
  def A(self) -> jnp.ndarray:
    return self._A

  @property
  def L(self) -> jnp.ndarray:
    return self._L

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return (self.A, self.L), {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Mapping[str, Any], children: Sequence[Any]
  ) -> "CholeskyDecomposition":
    del aux_data
    A, L = children
    self = super().__new__(cls)
    self._A, self._L = A, L
    return self


@jax.tree_util.register_pytree_node_class
class DenseCholeskyDecomposition(CholeskyDecomposition):

  def _decompose(self, A: jnp.ndarray) -> jnp.ndarray:
    return jsp.linalg.cholesky(A, lower=self.LOWER)

  def _solve(self, L: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jsp.linalg.solve_triangular(L, b, lower=self.LOWER)


@jax.tree_util.register_pytree_node_class
class SparseCholeskyDecomposition(CholeskyDecomposition):
  _FACTOR_CACHE = {}

  def _host_decompose(self, mat: jesp.CSR) -> jesp.CSR:
    # TODO(michalk8): more conversion to CSC
    # TODO(michalk8): test on GPU
    csc_mat = sp.csr_matrix(
        (np.array(mat.data), np.array(mat.indices), np.array(mat.indptr)),
        dtype=float
    ).tocsc()
    self._FACTOR_CACHE[hash(self)] = sksparse.cholmod.cholesky(csc_mat)

  def _decompose(self, A: jesp.CSR) -> jnp.ndarray:
    return hcb.call(self._host_decompose, A, result_shape=None)

  def _host_solve(self, b: jnp.ndarray) -> jnp.ndarray:
    factor = self._FACTOR_CACHE[hash(self)]
    x = factor.solve_A(np.array(b, dtype=float))
    return jnp.asarray(x, dtype=b.dtype)

  def _solve(self, A: jesp.CSR, b: jnp.ndarray) -> jnp.ndarray:
    return hcb.call(self._host_solve, b, result_shape=b)

  def __hash__(self):
    # TODO(michalk8): has based on A?
    return 0
