import abc
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jax
import jax.experimental.host_callback as hcb
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.sparse as sp
import sksparse.cholmod

__all__ = ["DenseCholeskyDecomposition", "SparseCholeskyDecomposition"]

# TODO(michalk8): bounds
T = TypeVar("T")


@jax.tree_util.register_pytree_node_class
class CholeskyDecomposition(abc.ABC, Generic[T]):
  LOWER = True

  def __init__(self, A: Union[T, sp.spmatrix], L: Optional[T] = None, **_: Any):
    if isinstance(A, sp.spmatrix):
      A = _scipy_sparse_to_jax(A)
    self._A = A

    if L is None:
      L = self._decompose(jax.lax.stop_gradient(A))
    self._L = L

  def __call__(self, b: jnp.ndarray) -> jnp.ndarray:
    return self._solve(self.L, b)

  @abc.abstractmethod
  def _decompose(self, A: T) -> Optional[T]:
    pass

  @abc.abstractmethod
  def _solve(self, L: Optional[T], b: jnp.ndarray) -> jnp.ndarray:
    pass

  @classmethod
  def create(cls, A: Union[T, sp.spmatrix], **kwargs: Any):
    if isinstance(A, sp.spmatrix):
      A = _scipy_sparse_to_jax(A)
    if isinstance(A, (jesp.CSR, jesp.CSC, jesp.COO)):
      return SparseCholeskyDecomposition(A, **kwargs)
    return DenseCholeskyDecomposition(A, **kwargs)

  @property
  def A(self) -> jnp.ndarray:
    return self._A

  @property
  def L(self) -> Optional[T]:
    return self._L

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return (self.A, self.L), {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Mapping[str, Any], children: Sequence[Any]
  ) -> "CholeskyDecomposition":
    return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class DenseCholeskyDecomposition(CholeskyDecomposition[jnp.ndarray]):

  def _decompose(self, A: T) -> Optional[T]:
    return jsp.linalg.cholesky(A, lower=self.LOWER)

  def _solve(self, L: Optional[T], b: jnp.ndarray) -> jnp.ndarray:
    return jsp.linalg.solve_triangular(L, b, lower=self.LOWER)


@jax.tree_util.register_pytree_node_class
class SparseCholeskyDecomposition(CholeskyDecomposition[jesp.CSR]):
  # TODO(michalk8): find a better impl.
  _FACTOR_CACHE = {}

  def __init__(
      self,
      A: T,
      L: Optional[T] = None,
      key: Optional[Hashable] = None,
      callback: Callable[[sp.csc_matrix], sp.csc_matrix] = None,
  ):
    self._key = key  # must be set before calling init
    self._calback = callback
    super().__init__(A, L)

  # TODO(michalk8): test on GPU
  def _host_decompose(self, A: T) -> None:
    # use float64 since it's required by CHOLMOD
    mat = _jax_sparse_to_scipy(A, dtype=float).tocsc()
    if self._calback is not None:
      mat = self._calback(mat)
    self._FACTOR_CACHE[hash(self)] = sksparse.cholmod.cholesky(mat)

  def _decompose(self, A: T) -> Optional[T]:
    return hcb.call(self._host_decompose, A, result_shape=None)

  # TODO(michalk8): test on GPU
  def _host_solve(self, b: jnp.ndarray) -> jnp.ndarray:
    factor = self._FACTOR_CACHE[hash(self)]
    return factor.solve_A(np.array(b, dtype=float))

  def _solve(self, _: Optional[T], b: jnp.ndarray) -> jnp.ndarray:
    # ideally, we would do a sparse triangular solve here
    return hcb.call(self._host_solve, b, result_shape=b)

  @classmethod
  def clear_factor_cache(cls) -> None:
    cls._FACTOR_CACHE.clear()

  def __hash__(self) -> int:
    return object.__hash__(self) if self._key is None else self._key

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    return children, {**aux_data, "key": self._key}


def _jax_sparse_to_scipy(
    A: Union[jesp.CSR, jesp.CSC, jesp.COO], **kwargs: Any
) -> sp.spmatrix:
  toarr = np.asarray

  if isinstance(A, (jesp.CSR, jesp.CSC)):
    data, indices, indptr = toarr(A.data), toarr(A.indices), toarr(A.indptr)
    return sp.csr_matrix((data, indices, indptr), **kwargs)
  if isinstance(A, jesp.COO):
    row, col, data = toarr(A.row), toarr(A.col), toarr(A.data)
    return sp.coo_matrix((data, (row, col)), **kwargs)

  raise TypeError(type(A))


def _scipy_sparse_to_jax(A: sp.spmatrix,
                         **kwargs: Any) -> Union[jesp.CSR, jesp.CSC, jesp.COO]:
  toarr = jnp.asarray
  kwargs["shape"] = A.shape

  if sp.isspmatrix_csr(A):
    return jesp.CSR((toarr(A.data), toarr(A.indices), toarr(A.indptr)),
                    **kwargs)
  if sp.isspmatrix_csc(A):
    return jesp.CSC((toarr(A.data), toarr(A.indices), toarr(A.indptr)),
                    **kwargs)
  if sp.isspmatrix_coo(A):
    return jesp.COO((toarr(A.data), toarr(A.row), toarr(A.col)), **kwargs)

  raise TypeError(type(A))
