from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import decomposition, fixed_point_loop
from ott.geometry import geometry

__all__ = ["Graph"]

Sparse_t = Union[jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]


@jax.tree_util.register_pytree_node_class
class Graph(geometry.Geometry):
  r"""Graph distance approximation using heat kernel :cite:`heitz:21,crane:13`.

  Approximates the heat kernel for large ``n_steps``, which for small
  :math:`\epsilon` approximates the geodesic exponential kernel
  :math:`e^{\frac{-d(x, y)^2}{\epsilon}}`.

  Args:
    graph: Graph represented as an adjacency matrix of shape ``[n, n]``.
      If ``None``, the symmetric graph Laplacian has to be specified.
    laplacian: Symmetric graph Laplacian. The check for symmetry is **NOT**
      performed. If ``None``, the graph has to be specified instead.
    epsilon: Epsilon regularizer.
    n_steps: Number of steps used for the heat diffusion.
    numerical_scheme: Numerical scheme used to solve the heat diffusion.
    directed: Whether the ``graph`` is directed. If not, it will be made
      undirected as :math:`G + G^T`. This parameter is ignored when  directly
      passing the Laplacian, which is assumed to be symmetric.
    kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      graph: Optional[Union[jnp.ndarray, jesp.BCOO]] = None,
      laplacian: Optional[Union[jnp.ndarray, Sparse_t]] = None,
      epsilon: float = 1e-2,
      n_steps: int = 100,
      numerical_scheme: Literal["backward_euler",
                                "crank_nicolson"] = "backward_euler",
      directed: bool = False,
      **kwargs: Any
  ):
    # TODO(michak8): disallow epsilon scheduler
    assert ((graph is None and laplacian is not None) or
            (laplacian is None and graph is not None)), \
           "Please provide the graph or the symmetric graph Laplacian."
    super().__init__(epsilon=epsilon, **kwargs)
    if graph is not None and not isinstance(graph, (jnp.ndarray, jesp.BCOO)):
      raise NotImplementedError(
          f"Graph must be in `BCOO` format, found `{type(graph).__name__}`."
      )
    self._graph = graph
    self._laplacian = laplacian
    self._solver: Optional[decomposition.CholeskySolver] = None

    self.n_steps = n_steps
    self.numerical_scheme = numerical_scheme
    self.directed = directed

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:

    def body_fn(
        iteration: int, solver_lap: Tuple[decomposition.CholeskySolver,
                                          Optional[Union[jnp.ndarray,
                                                         Sparse_t]]],
        b: jnp.ndarray, compute_errors: bool
    ) -> jnp.ndarray:
      del iteration, compute_errors
      solver, scaled_lap = solver_lap
      if self.numerical_scheme == "crank_nicolson":
        # below is a preferred way of specifying the update (albeit more FLOPS),
        # as CSR/CSC/COO matrices don't support adding a diagonal matrix now:
        # b' = (2 * I - M) @ b = (2 * I - (I + c * L)) @ b = (I - c * L) @ b
        b = b - scaled_lap @ b
      return solver.solve(b)

    # eps we cannot use since it would require a re-solve
    # axis we can ignore since the matrix is symmetric
    del eps, axis

    if self.numerical_scheme == "crank_nicolson":
      constants = self.solver, self._scaled_laplacian
    else:
      constants = self.solver, None

    return fixed_point_loop.fixpoint_iter(
        cond_fn=lambda *_, **__: True,
        body_fn=body_fn,
        min_iterations=self.n_steps,
        max_iterations=self.n_steps,
        inner_iterations=1,
        constants=constants,
        state=scaling,
    )

  @property
  def kernel_matrix(self) -> jnp.ndarray:

    def body_fn(carry: None, ix: int) -> Tuple[None, jnp.ndarray]:
      vec = jnp.zeros(n).at[ix].set(1.)
      return carry, self.apply_kernel(vec)

    # TODO(michalk8): consider disallowing instantiating the kernel?
    # TODO(michalk8): enable batching once sparse solver (as primitive) is added
    # batching rules are not implemented for `hcb.call`
    # return jax.vmap(self.apply_kernel)(jnp.eye(self.shape[0]))
    n, _ = self.shape
    _, kernel = jax.lax.scan(body_fn, None, jnp.arange(n))
    return kernel

  @property
  def laplacian(self) -> Union[jnp.ndarray, Sparse_t]:
    """The graph Laplacian."""
    if self._laplacian is not None:
      return self._laplacian

    if self.is_sparse:
      n, _ = self.shape
      D, ixs = self.graph.sum(1).todense(), jnp.arange(n)
      D = jesp.BCOO((D, jnp.c_[ixs, ixs]), shape=(n, n))
    else:
      D = jnp.diag(self.graph.sum(1))

    # TODO(michalk8): test directed under JIT for the sparse case
    A = (self.graph + self.graph.T) if self.directed else self.graph

    # in the sparse case, we don't sum duplicates here because
    # we would to know `nnz` a priori for JIT (could be exposed in `__init__`)
    # instead, `ott.core.decomposition._jax_sparse_to_scipy` handles it on host
    return D - A

  @property
  def _scale(self) -> float:
    if self.numerical_scheme == "backward_euler":
      # TODO(michalk8): check the constants 4 and 2
      return self.epsilon / (4 * self.n_steps)
    if self.numerical_scheme == "crank_nicolson":
      return self.epsilon / (2 * self.n_steps)
    raise NotImplementedError(
        f"Numerical scheme `{self.numerical_scheme}` is not implemented."
    )

  @property
  def _scaled_laplacian(self) -> Union[float, jnp.ndarray, Sparse_t]:
    """Laplacian scaled by a constant, depending on the numerical scheme."""
    if self.is_sparse:
      return _scale_sparse(self._scale, self.laplacian)
    return self._scale * self.laplacian

  @property
  def _M(self) -> Union[jnp.ndarray, Sparse_t]:
    n, _ = self.shape
    scaled_lap = self._scaled_laplacian
    # CHOLMOD supports solving `A + beta * I`, we set `beta = 1.0`
    # when instantiating the solver
    return scaled_lap if self.is_sparse else scaled_lap + jnp.eye(n)

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
  def shape(self) -> Tuple[int, int]:
    arr = self.graph if self.graph is not None else self._laplacian
    return arr.shape

  @property
  def is_sparse(self) -> bool:
    """Whether :attr:`graph` or :attr:`laplacian` is sparse."""
    if self._laplacian is not None:
      return isinstance(self.laplacian, Sparse_t.__args__)
    if isinstance(self.graph, (jesp.CSR, jesp.CSC, jesp.COO)):
      raise NotImplementedError("Graph must be specified in `BCOO` format.")
    return isinstance(self.graph, jesp.BCOO)

  @property
  def graph(self) -> Optional[Union[jnp.ndarray, jesp.BCOO]]:
    """The underlying undirected graph, if provided."""
    return self._graph

  @property
  def is_symmetric(self) -> bool:
    return True

  # TODO(michalk8): disallow more functions, test transport output
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
    return [self.graph, self._laplacian, self._solver], {
        "epsilon": self.epsilon,
        "n_steps": self.n_steps,
        "numerical_scheme": self.numerical_scheme,
        "directed": self.directed,
        **self._kwargs,
    }

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "Graph":
    graph, laplacian, solver = children
    obj = cls(graph=graph, laplacian=laplacian, **aux_data)
    obj._solver = solver
    return obj


def _scale_sparse(c: float, mat: Sparse_t) -> Sparse_t:
  """Scale a sparse matrix by a constant."""
  if isinstance(mat, jesp.BCOO):
    # most feature complete, defer to original impl.
    return c * mat
  (data, *children), aux_data = mat.tree_flatten()
  return type(mat).tree_unflatten(aux_data, [c * data] + children)
