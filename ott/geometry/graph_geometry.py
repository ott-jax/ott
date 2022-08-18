from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import scipy.sparse as sp
from typing_extensions import Literal

from ott.core import decomposition, fixed_point_loop
from ott.geometry import geometry

Sparse_t = Union[jsparse.CSR, jsparse.CSC, jsparse.COO]


@jax.tree_util.register_pytree_node_class
class GraphGeometry(geometry.Geometry):

  def __init__(
      self,
      laplacian: Union[jnp.ndarray, Sparse_t],
      epsilon: float = 1e-2,
      n_iter: int = 100,
      numerical_scheme: Literal["backward_euler",
                                "crank_nicolson"] = "backward_euler",
      **kwargs: Any
  ):
    super().__init__(epsilon=epsilon, **kwargs)
    self._laplacian = laplacian
    self._solver: Optional[decomposition.CholeskyDecomposition] = None
    self.n_iter = n_iter
    self.numerical_scheme = numerical_scheme

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:

    def body_fn(
        iteration: int, solver: decomposition.CholeskyDecomposition,
        b: jnp.ndarray, compute_errors: bool
    ) -> jnp.ndarray:
      del iteration, compute_errors
      return solver(b)

    # eps we cannot use since it would require a re-solve
    # axis we can ignore since the matrix is symmetric
    del eps, axis

    return fixed_point_loop.fixpoint_iter(
        cond_fn=lambda *_, **__: True,
        body_fn=body_fn,
        min_iterations=self.n_iter,
        max_iterations=self.n_iter,
        inner_iterations=1,
        constants=self.solver,
        state=scaling,
    )

  def apply_transport_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    raise ValueError("Not implemented.")

  @property
  def kernel_matrix(self) -> jnp.ndarray:
    raise NotImplementedError("TODO")

  @property
  def solver(self) -> decomposition.CholeskyDecomposition:
    if self._solver is not None:
      return self._solver

    if self.is_sparse:
      mat, callback = self.laplacian, _laplacian_to_M
    else:
      mat, callback = self._M, None

    self._solver = decomposition.CholeskyDecomposition.create(
        mat, callback=callback
    )
    return self._solver

  @property
  def _M(self) -> Union[jnp.ndarray, Sparse_t]:
    if self.is_sparse:
      raise NotImplementedError("TODO")
    return self.laplacian + self._scale * jnp.eye(self.shape[0])

  @property
  def _scale(self) -> float:
    if self.numerical_scheme == "backward_euler":
      return self.epsilon / self.n_iter
    raise NotImplementedError(self.numerical_scheme)

  @property
  def shape(self) -> Tuple[int, int]:
    return self.laplacian.shape

  @property
  def is_sparse(self) -> bool:
    return isinstance(self.laplacian, Sparse_t.__args__)

  @property
  def laplacian(self) -> Union[jnp.ndarray, Sparse_t]:
    return self._laplacian

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self.laplacian, self._solver], {
        "epsilon": self.epsilon,
        "n_iter": self.n_iter,
        **self._kwargs,
    }

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "GraphGeometry":
    laplacian, solver = children
    obj = cls(laplacian, **aux_data)
    obj._solver = solver
    return obj


def _laplacian_to_M(laplacian: sp.csc_matrix, scale: float) -> sp.csc_matrix:
  return laplacian + scale * sp.eye(
      laplacian.shape[0], format="csc", dtype=laplacian.dtype
  )
