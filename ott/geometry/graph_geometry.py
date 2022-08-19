from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import decomposition, fixed_point_loop
from ott.geometry import geometry


@jax.tree_util.register_pytree_node_class
class GraphGeometry(geometry.Geometry):

  def __init__(
      self,
      laplacian: Union[jnp.ndarray, jesp.BCOO],
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

    def body_fn(carry: None, ix: int) -> Tuple[None, jnp.ndarray]:
      vec = jnp.zeros(n).at[ix].set(1.)
      return carry, self.apply_kernel(vec)

    # TODO(michalk8): enable batching once sparse solver (as primitive) is added
    # batching rules are not implemented for `hcb.call`
    # return jax.vmap(self.apply_kernel)(jnp.eye(self.shape[0]))
    n, _ = self.shape
    _, kernel = jax.lax.scan(body_fn, None, jnp.arange(n))
    return kernel

  @property
  def is_symmetric(self) -> bool:
    return True

  @property
  def solver(self) -> decomposition.CholeskyDecomposition:
    if self._solver is None:
      self._solver = decomposition.CholeskyDecomposition.create(self._M)
    return self._solver

  @property
  def _M(self) -> Union[jnp.ndarray, jesp.BCOO]:
    n, _ = self.shape
    if self.is_sparse:
      return self._scale * self.laplacian + _speye(n)
    return self._scale * self.laplacian + jnp.eye(n)

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
    return isinstance(self.laplacian, jesp.BCOO)

  @property
  def laplacian(self) -> Union[jnp.ndarray, jesp.BCOO]:
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


def _speye(n: int) -> jesp.BCOO:
  ixs = jnp.arange(n)
  return jesp.BCOO((jnp.ones(n), jnp.c_[ixs, ixs]), shape=(n, n))
