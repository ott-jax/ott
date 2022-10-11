from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import _math_utils as mu
from ott.core import decomposition, fixed_point_loop
from ott.geometry import geometry

__all__ = ["Graph"]

Sparse_t = Union[jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]


@jax.tree_util.register_pytree_node_class
class Graph(geometry.Geometry):
  r"""Graph distance approximation using heat kernel :cite:`heitz:21,crane:13`.

  Approximates the heat kernel for large ``n_steps``, which for small ``t``
  approximates the geodesic exponential kernel :math:`e^{\frac{-d(x, y)^2}{t}}`.

  For sparse graphs, :mod:`sksparse.cholmod` is required to compute the Cholesky
  decomposition.
  Differentiating w.r.t. the edge weights is currently possible only when the
  graph is represented as a dense adjacency matrix.

  Args:
    graph: Graph represented as an adjacency matrix of shape ``[n, n]``.
      If `None`, the symmetric graph Laplacian has to be specified.
    laplacian: Symmetric graph Laplacian. The check for symmetry is **NOT**
      performed. If `None`, the graph has to be specified instead.
    t: Constant used when approximating the geodesic exponential kernel.
      If `None`, use :math:`\frac{1}{|E|} \sum_{(u, v) \in E} weight(u, v)`
      :cite:`crane:13`. In this case, the ``graph`` must be specified
      and the edge weights are all assumed to be positive.
    n_steps: Maximum number of steps used to approximate the heat kernel.
    numerical_scheme: Numerical scheme used to solve the heat diffusion.
    directed: Whether the ``graph`` is directed. If not, it will be made
      undirected as :math:`G + G^T`. This parameter is ignored when  directly
      passing the Laplacian, which is assumed to be symmetric.
    tol: Relative tolerance with respect to the Hilbert metric, see
      :cite:`peyre:19`, Remark 4.12. Used when iteratively updating scalings.
      If negative, this option is ignored and only ``n_steps`` is used.
    kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      graph: Optional[Union[jnp.ndarray, jesp.BCOO]] = None,
      laplacian: Optional[Union[jnp.ndarray, Sparse_t]] = None,
      t: Optional[float] = 1e-3,
      n_steps: int = 100,
      numerical_scheme: Literal["backward_euler",
                                "crank_nicolson"] = "backward_euler",
      directed: bool = False,
      tol: float = -1.,
      **kwargs: Any
  ):
    assert ((graph is None and laplacian is not None) or
            (laplacian is None and graph is not None)), \
           "Please provide a graph or a symmetric graph Laplacian."
    # arbitrary epsilon; can't use `None` as `mean_cost_matrix` would be used
    super().__init__(epsilon=1., **kwargs)
    self._graph = graph
    self._laplacian = laplacian
    self._solver: Optional[decomposition.CholeskySolver] = None

    self._t = t
    self.n_steps = n_steps
    self.numerical_scheme = numerical_scheme
    self.directed = directed
    self._tol = tol

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:

    def conf_fn(
        iteration: int, solver_lap: Tuple[decomposition.CholeskySolver,
                                          Optional[Union[jnp.ndarray,
                                                         Sparse_t]]],
        old_new: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> bool:
      del iteration, solver_lap

      x_old, x_new = old_new
      x_old, x_new = mu.safe_log(x_old), mu.safe_log(x_new)
      # center
      x_old, x_new = x_old - jnp.nanmax(x_old), x_new - jnp.nanmax(x_new)
      # Hilbert metric, see Remark 4.12 in `Computational Optimal Transport`
      f = x_new - x_old
      return (jnp.nanmax(f) - jnp.nanmin(f)) > self._tol

    def body_fn(
        iteration: int, solver_lap: Tuple[decomposition.CholeskySolver,
                                          Optional[Union[jnp.ndarray,
                                                         Sparse_t]]],
        old_new: Tuple[jnp.ndarray, jnp.ndarray], compute_errors: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
      del iteration, compute_errors

      solver, scaled_lap = solver_lap
      _, b = old_new

      if self.numerical_scheme == "crank_nicolson":
        # below is a preferred way of specifying the update (albeit more FLOPS),
        # as CSR/CSC/COO matrices don't support adding a diagonal matrix now:
        # b' = (2 * I - M) @ b = (2 * I - (I + c * L)) @ b = (I - c * L) @ b
        b = b - scaled_lap @ b
      return b, solver.solve(b)

    # eps we cannot use since it would require a re-solve
    # axis we can ignore since the matrix is symmetric
    del eps, axis

    force_scan = self._tol < 0.
    fixpoint_fn = (
        fixed_point_loop.fixpoint_iter
        if force_scan else fixed_point_loop.fixpoint_iter_backprop
    )

    state = (jnp.full_like(scaling, jnp.nan), scaling)
    if self.numerical_scheme == "crank_nicolson":
      constants = self.solver, self._scaled_laplacian
    else:
      constants = self.solver, None

    return fixpoint_fn(
        cond_fn=(lambda *_, **__: True) if force_scan else conf_fn,
        body_fn=body_fn,
        min_iterations=self.n_steps if force_scan else 1,
        max_iterations=self.n_steps,
        inner_iterations=1,
        constants=constants,
        state=state,
    )[1]

  def apply_transport_from_scalings(
      self,
      u: jnp.ndarray,
      v: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:

    def body_fn(carry: None, vec: jnp.ndarray) -> jnp.ndarray:
      if axis == 1:
        return carry, u * self.apply_kernel(v * vec, axis=axis)
      return carry, v * self.apply_kernel(u * vec, axis=axis)

    if not self.is_sparse:
      return super().apply_transport_from_scalings(u, v, vec, axis=axis)

    # we solve the triangular system's on host, but
    # batching rules are implemented only for `id_tap`, not for `call`
    if vec.ndim == 1:
      _, res = jax.lax.scan(body_fn, None, vec[None, :])
      return res[0, :]

    _, res = jax.lax.scan(body_fn, None, vec)
    return res

  @property
  def kernel_matrix(self) -> jnp.ndarray:
    n, _ = self.shape
    kernel = self.apply_kernel(jnp.eye(n))
    # force symmetry because of numerical imprecisions
    # happens when `numerical_scheme='backward_euler'` and small `t`
    return (kernel + kernel.T) * .5

  @property
  def cost_matrix(self) -> jnp.ndarray:
    return -self.t * mu.safe_log(self.kernel_matrix)

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

    # in the sparse case, we don't sum duplicates here because
    # we need to know `nnz` a priori for JIT (could be exposed in `__init__`)
    # instead, `ott.core.decomposition._jax_sparse_to_scipy` handles it on host
    return D - self.graph

  @property
  def t(self) -> float:
    """Constant used when approximating the geodesic exponential kernel."""
    if self._t is None:
      graph = self.graph
      assert graph is not None, "No graph was specified."
      if self.is_sparse:
        return jnp.mean(graph.data) ** 2
      return (jnp.sum(graph) / jnp.sum(graph > 0.)) ** 2
    return self._t

  @property
  def _scale(self) -> float:
    """Constant to scale the Laplacian with."""
    if self.numerical_scheme == "backward_euler":
      return self.t / (4. * self.n_steps)
    if self.numerical_scheme == "crank_nicolson":
      return self.t / (2. * self.n_steps)
    raise NotImplementedError(
        f"Numerical scheme `{self.numerical_scheme}` is not implemented."
    )

  @property
  def _scaled_laplacian(self) -> Union[float, jnp.ndarray, Sparse_t]:
    """Laplacian scaled by a constant, depending on the numerical scheme."""
    if self.is_sparse:
      return mu.sparse_scale(self._scale, self.laplacian)
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
    """Instantiate the Cholesky solver and compute the factorization."""
    if self._solver is None:
      # key/beta only used for sparse solver
      self._solver = decomposition.CholeskySolver.create(
          self._M, beta=1., key=hash(self)
      )
      # compute the factorization to avoid tracer leaks in `apply_kernel`
      # due to the scan/while loop
      _ = self._solver.L
    return self._solver

  @property
  def shape(self) -> Tuple[int, int]:
    arr = self._graph if self._graph is not None else self._laplacian
    return arr.shape

  @property
  def is_sparse(self) -> bool:
    """Whether :attr:`graph` or :attr:`laplacian` is sparse."""
    if self._laplacian is not None:
      return isinstance(self.laplacian, Sparse_t.__args__)
    if isinstance(self._graph, (jesp.CSR, jesp.CSC, jesp.COO)):
      raise NotImplementedError("Graph must be specified in `BCOO` format.")
    return isinstance(self._graph, jesp.BCOO)

  @property
  def graph(self) -> Optional[Union[jnp.ndarray, jesp.BCOO]]:
    """The underlying undirected graph as an adjacency matrix, if provided."""
    if self._graph is None:
      return None
    return (self._graph + self._graph.T) if self.directed else self._graph

  @property
  def is_symmetric(self) -> bool:
    # there are some numerical imprecisions, but it should be symmetric
    return True

  @property
  def dtype(self) -> jnp.dtype:
    return self._graph.dtype

  # TODO(michalk8): in future, use mixins for lse/kernel mode
  def transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray
  ) -> jnp.ndarray:
    """Not implemented."""
    raise ValueError("Not implemented.")

  def apply_transport_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    """Not implemented."""
    raise ValueError("Not implemented.")

  def marginal_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Not implemented."""
    raise ValueError("Not implemented.")

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self._graph, self._laplacian, self.solver], {
        "t": self._t,
        "n_steps": self.n_steps,
        "numerical_scheme": self.numerical_scheme,
        "directed": self.directed,
        "tol": self._tol,
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
