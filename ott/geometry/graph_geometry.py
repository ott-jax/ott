from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import decomposition, fixed_point_loop
from ott.geometry import geometry

Sparse_t = Union[jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]


# TODO(michalk8): consider passing passing the graph directly instead of
# the Laplacian
@jax.tree_util.register_pytree_node_class
class GraphGeometry(geometry.Geometry):
  """Graph geodesic distance approximation using heat kernel :cite:`heitz:21`.

  Args:
    laplacian: Symmetric graph Laplacian.
    epsilon: TODO.
    n_iter: Number of iterations used for the heat diffusion.
    numerical_scheme: Numerical scheme to solve the heat diffusion.
      Currently, only ``'backward_euler'`` is implemented.
    kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

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
    self._solver: Optional[decomposition.CholeskySolver] = None
    self.n_iter = n_iter
    self.numerical_scheme = numerical_scheme

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:

    def body_fn(
        iteration: int, solver: decomposition.CholeskySolver, b: jnp.ndarray,
        compute_errors: bool
    ) -> jnp.ndarray:
      del iteration, compute_errors
      return solver.solve(b)

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

  @property
  def kernel_matrix(self) -> jnp.ndarray:

    def body_fn(carry: None, ix: int) -> Tuple[None, jnp.ndarray]:
      vec = jnp.zeros(n).at[ix].set(1.)
      return carry, self.apply_kernel(vec)

    # TODO(michalk8): consider disallowing instantiating the kernel?/
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
  def solver(self) -> decomposition.CholeskySolver:
    """Cholesky solver."""
    if self._solver is None:
      # key/beta only used for sparse solver
      self._solver = decomposition.CholeskySolver.create(
          self._M, beta=1.0, key=hash(self)
      )
    return self._solver

  @property
  def _M(self) -> Union[jnp.ndarray, Sparse_t]:
    n, _ = self.shape
    if self.is_sparse:
      # CHOLMOD supports solving `A + beta * I`, we set `beta=1.0`
      # when instantiating the solver
      return _scale_sparse(self._scale, self.laplacian)
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
    """Whether the graph Laplacian is sparse."""
    return isinstance(self.laplacian, Sparse_t.__args__)

  @property
  def laplacian(self) -> Union[jnp.ndarray, Sparse_t]:
    """The graph Laplacian."""
    return self._laplacian

  # TODO(michalk8): disallow for more, test transport output
  def apply_transport_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    """Not implemented."""
    raise ValueError("Not implemented.")

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


def _scale_sparse(scale: float, mat: Sparse_t) -> Sparse_t:
  if isinstance(mat, jesp.BCOO):
    # most feature complete, defer to original impl.
    return scale * mat
  (data, *children), aux_data = mat.tree_flatten()
  return type(mat).tree_unflatten(aux_data, [scale * data] + children)
