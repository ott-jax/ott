from functools import cached_property
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.scipy as jsp
import scipy.sparse as sp

from ott.core import fixed_point_loop
from ott.geometry import geometry

Sparse_t = Union[jsparse.CSR, jsparse.CSC, jsparse.COO, jsparse.BCOO]


@jax.tree_util.register_pytree_node_class
class GraphGeometry(geometry.Geometry):

  def __init__(
      self, laplacian: Union[jnp.ndarray, Sparse_t], t: float, k: int,
      **kwargs: Any
  ):
    super().__init__(**kwargs)
    self.laplacian = laplacian
    self.t = t
    self.k = k

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:

    def body_fn(
        iteration: int, solver: Any, b: jnp.ndarray, compute_errors: bool
    ) -> jnp.ndarray:
      del iteration, compute_errors
      return solver(b)

    if self.is_sparse:
      solver = jax.jit(
          lambda x: jax.tree_util.
          Partial(jsp.sparse.linalg.gmres, self.laplacian)(x)[0]
      )
    else:
      solver = jax.jit(
          jax.tree_util.Partial(jsp.linalg.solve_triangular, self.laplacian)
      )

    return fixed_point_loop.fixpoint_iter(
        cond_fn=lambda *_, **__: True,
        body_fn=body_fn,
        min_iterations=self.k,
        max_iterations=self.k,
        inner_iterations=1,
        constants=solver,
        state=scaling,
    )

  def _apply_kernel_sparse(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    import jax.experimental.host_callback as hcb
    import sksparse.cholmod

    def callback(vec: jnp.ndarray) -> jnp.ndarray:
      if not isinstance(vec, jnp.ndarray):
        return vec

      lap = self.laplacian
      M = to_sklearn_sparse(lap)
      M = sp.eye(M.shape[0], dtype=M.dtype) + self._lap_scale * M
      factor = sksparse.cholmod.cholesky(M.tocsc())
      dtype = vec.dtype

      for i in range(self.k):
        vec = factor(vec)
      return jnp.array(vec, dtype=dtype)

    return hcb.call(
        callback,
        scaling,
        result_shape=jax.ShapedArray(scaling.shape, scaling.dtype)
    )

  def _triangular_solver(self):
    import sksparse.cholmod
    M = to_sklearn_sparse(self.laplacian)
    # M = sp.eye(511, dtype=M.dtype) + self._lap_scale * M
    return sksparse.cholmod.cholesky(M.tocsc()).solve_A

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
  def _lap_scale(self) -> float:
    # TODO(michalk8): not only Euler
    return self.t / self.k

  @property
  def _M(self) -> Union[jnp.ndarray, Sparse_t]:
    if self.is_sparse:
      raise NotImplementedError("No sparse implementation yet.")
    n, _ = self.shape
    laplacian = self._lap_scale * self.laplacian
    return jnp.eye(n) + laplacian

  @cached_property
  def _L(self) -> jnp.ndarray:
    return jnp.linalg.cholesky(self._M)

  @property
  def shape(self) -> Tuple[int, int]:
    return self.laplacian.shape

  @property
  def is_sparse(self) -> bool:
    return isinstance(self.laplacian, Sparse_t.__args__)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [
        self.laplacian, self._epsilon_init, self._relative_epsilon,
        self._scale_epsilon, self._kwargs
    ], {
        "t": self.t,
        "k": self.k
    }

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "GraphGeometry":
    lap, eps_init, rel_eps, scale_eps, kwargs = children
    return cls(
        lap,
        epsilon=eps_init,
        relative_epsilon=rel_eps,
        scale_epsilon=scale_eps,
        **kwargs,
        **aux_data
    )


def to_sklearn_sparse(mat: Sparse_t):
  if isinstance(mat, jsparse.CSR):
    return sp.csr_matrix((mat.data, mat.indices, mat.indptr), dtype=mat.dtype)
  if isinstance(mat, jsparse.CSC):
    return sp.csc_matrix((mat.data, mat.indices, mat.indptr), dtype=mat.dtype)
  raise NotImplementedError(type(mat))
