from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.scipy as jsp

from ott.core import fixed_point_loop
from ott.geometry import geometry

Sparse_t = jsparse.BCOO


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

    solver = jax.tree_util.Partial(jsp.linalg.solve_triangular, self._L)
    return fixed_point_loop.fixpoint_iter(
        cond_fn=lambda *_, **__: True,
        body_fn=body_fn,
        min_iterations=self.k,
        max_iterations=self.k,
        inner_iterations=1,
        constants=solver,
        state=scaling,
    )

  @property
  def kernel_matrix(self) -> jnp.ndarray:
    raise NotImplementedError("TODO")

  @property
  def _lap_scale(self) -> float:
    # TODO(michalk8): not only Euler
    return self.t / self.k

  @property
  def _L(self) -> Union[jnp.ndarray, Sparse_t]:
    n, _ = self.shape
    laplacian = self._lap_scale * self.laplacian
    if self.is_sparse:
      raise NotImplementedError("No sparse implementation yet.")
    M = jnp.eye(n) + laplacian
    return jnp.linalg.cholesky(M)

  @property
  def shape(self) -> Tuple[int, int]:
    return self.laplacian.shape

  @property
  def is_sparse(self) -> bool:
    return isinstance(self.laplacian, Sparse_t)

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
