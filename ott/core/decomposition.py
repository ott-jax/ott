import abc
from typing import (
    Any,
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

try:
  from sksparse import cholmod
except ImportError:
  cholmod = None

__all__ = ["CholeskySolver", "DenseCholeskySolver", "SparseCholeskySolver"]

T = TypeVar("T")


@jax.tree_util.register_pytree_node_class
class CholeskySolver(abc.ABC, Generic[T]):
  """Base class for Cholesky linear solver.

  Args:
    A: Symmetric positive definite matrix of shape ``[n, n]``.
  """

  def __init__(self, A: T):
    self._A = A
    self._L: Optional[T] = None  # Cholesky factor

  def solve(self, b: jnp.ndarray) -> jnp.ndarray:
    """Solve the linear system :math:`A * x = b`.

    Args:
        b: Vector of shape ``[n,]``.

    Returns:
        The solution of shape ``[n,]``.
    """
    return self._solve(self.L, b)

  @abc.abstractmethod
  def _decompose(self, A: T) -> Optional[T]:
    """Decompose matrix ``A`` into Cholesky factor."""

  @abc.abstractmethod
  def _solve(self, L: Optional[T], b: jnp.ndarray) -> jnp.ndarray:
    """Solve a triangular linear system :math:`L * x = b`."""

  @classmethod
  def create(cls, A: Union[T, sp.spmatrix], **kwargs: Any) -> "CholeskySolver":
    """Instantiate sparse or dense Cholesky solver.

    Optionally converts :class:`scipy.sparse.spmatrix` to its
    :mod:`jax` equivalent.

    Args:
      A: Symmetric positive definite matrix of shape ``[n, n]``.
      kwargs: Keyword arguments for the initialization.

    Returns:
      Sparse or dense Cholesky solver.
    """
    if isinstance(A, sp.spmatrix):
      A = _scipy_sparse_to_jax(A)
    if isinstance(A, (jesp.CSR, jesp.CSC, jesp.BCOO)):
      return SparseCholeskySolver(A, **kwargs)
    return DenseCholeskySolver(A, **kwargs)

  @property
  def A(self) -> jnp.ndarray:
    """Symmetric positive definite matrix of shape ``[n, n]``."""
    return self._A

  @property
  def L(self) -> Optional[T]:
    """Cholesky factor of :attr:`A`."""
    if self._L is None:
      self._L = self._decompose(self.A)
    return self._L

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return (self.A, self.L), {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Mapping[str, Any], children: Sequence[Any]
  ) -> "CholeskySolver":
    A, L = children
    obj = cls(A, **aux_data)
    obj._L = L
    return obj


@jax.tree_util.register_pytree_node_class
class DenseCholeskySolver(CholeskySolver[jnp.ndarray]):
  """Dense Cholesky solver.

  Args:
    A: Symmetric positive definite matrix of shape ``[n, n]``.
    lower: Whether to compute lower-triangular Cholesky factor.
    kwargs: Additional keyword arguments, currently ignored.
  """

  def __init__(self, A: T, lower: bool = True, **kwargs: Any):
    del kwargs
    super().__init__(A)
    self._lower = lower

  def _decompose(self, A: T) -> Optional[T]:
    return jsp.linalg.cholesky(A, lower=self._lower)

  def _solve(self, L: Optional[T], b: jnp.ndarray) -> jnp.ndarray:
    return jsp.linalg.solve_triangular(L, b, lower=self._lower)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    aux_data["lower"] = self._lower
    return children, aux_data


@jax.tree_util.register_pytree_node_class
class SparseCholeskySolver(
    CholeskySolver[Union[jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]]
):
  r"""Sparse Cholesky solver using :func:`jax.experimental.host_callback.call`.

  Uses the CHOLMOD :cite:`cholmod:08` bindings from :mod:`sksparse.cholmod`.

  Args:
    A: Symmetric positive definite matrix of shape ``[n, n]``.
    beta: Decompose :math:`A + \beta * I` instead of :math:`A`.
    key: Key used to cache :class:`sksparse.cholmod.Factor`.
      This key **must** be unique to ``A`` to achieve correct results.
      If `None`, use :func:`hash` of this object.
    kwargs: Keyword arguments for :func:`sksparse.cholmod.cholesky`.
  """

  # TODO(michalk8): in the future, define a jax primitive + use CHOLMOD directly
  _FACTOR_CACHE = {}

  def __init__(
      self,
      A: T,
      beta: float = 0.0,
      key: Optional[Hashable] = None,
      **kwargs: Any,
  ):
    if cholmod is None:
      raise ImportError(
          "Unable to import scikit-sparse. "
          "Please install it as `pip install scikit-sparse`."
      )
    super().__init__(A)
    self._key = key
    self._beta = beta
    self._kwargs = kwargs

  def _host_decompose(self, A: T) -> None:
    # use float64 because CHOLMOD uses it internally
    # convert to CSC explicitly for efficiency/to avoid warnings
    mat = _jax_sparse_to_scipy(A, sum_duplicates=True, dtype=float).tocsc()
    self._FACTOR_CACHE[hash(self)] = cholmod.cholesky(
        mat, beta=self._beta, **self._kwargs
    )

  def _decompose(self, A: T) -> Optional[T]:
    return hcb.call(self._host_decompose, A, result_shape=None)

  def _host_solve(self, b: jnp.ndarray) -> jnp.ndarray:
    factor = self._FACTOR_CACHE[hash(self)]
    return factor.solve_A(np.asarray(b, dtype=float))

  def _solve(self, _: Optional[T], b: jnp.ndarray) -> jnp.ndarray:
    return hcb.call(self._host_solve, b, result_shape=b)

  @property
  def L(self) -> None:
    """Compute the lower-triangular factor of :attr:`A` and cache the result.

    The factor is not returned, but is used in subsequent :meth:`solve` calls.
    """
    if hash(self) not in self._FACTOR_CACHE:
      self._decompose(jax.lax.stop_gradient(self.A))

  @classmethod
  def clear_factor_cache(cls) -> None:
    """Clear the :class:`sksparse.cholmod.Factor` cache."""
    cls._FACTOR_CACHE.clear()

  def __hash__(self) -> int:
    return object.__hash__(self) if self._key is None else self._key

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    return children, {
        **aux_data, "beta": self._beta,
        "key": self._key,
        **self._kwargs
    }


def _jax_sparse_to_scipy(
    A: Union[jesp.CSR, jesp.CSC, jesp.COO],
    sum_duplicates: bool = False,
    **kwargs: Any
) -> sp.spmatrix:
  toarr = np.asarray

  if isinstance(A, (jesp.CSR, jesp.CSC)):
    data, indices, indptr = toarr(A.data), toarr(A.indices), toarr(A.indptr)
    return sp.csr_matrix((data, indices, indptr), **kwargs)
  if isinstance(A, jesp.COO):
    row, col, data = toarr(A.row), toarr(A.col), toarr(A.data)
    return sp.coo_matrix((data, (row, col)), **kwargs)
  if isinstance(A, jesp.BCOO):
    assert A.indices.ndim == 2, "Only 2D batched COO matrix is supported."
    row, col = A.indices[:, 0], A.indices[:, 1]
    data, row, col = toarr(A.data), toarr(row), toarr(col)
    mat = sp.coo_matrix((data, (row, col)), **kwargs)
    if not sum_duplicates:
      return mat

    # the original matrix can contain duplicates => shape will be wrong
    # we optionally correct it here
    mat.sum_duplicates()
    mat.eliminate_zeros()

    return sp.coo_matrix((mat.data, (mat.row, mat.col)), **kwargs)

  raise TypeError(type(A))


def _scipy_sparse_to_jax(A: sp.spmatrix,
                         **kwargs: Any) -> Union[jesp.CSR, jesp.CSC, jesp.BCOO]:
  toarr = jnp.asarray
  kwargs["shape"] = A.shape

  if sp.isspmatrix_csr(A):
    return jesp.CSR((toarr(A.data), toarr(A.indices), toarr(A.indptr)),
                    **kwargs)
  if sp.isspmatrix_csc(A):
    return jesp.CSC((toarr(A.data), toarr(A.indices), toarr(A.indptr)),
                    **kwargs)
  if sp.isspmatrix_coo(A):
    # prefer BCOO since it's more feature-complete
    data, indices = toarr(A.data), jnp.c_[toarr(A.row), toarr(A.col)]
    return jesp.BCOO((data, indices), **kwargs)

  raise TypeError(type(A))
