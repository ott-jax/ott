from typing import Optional, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from networkx.algorithms import shortest_paths
from networkx.generators import random_graphs
from typing_extensions import Literal

from ott.core import decomposition
from ott.core import implicit_differentiation as implicit_lib
from ott.core import linear_problems, sinkhorn
from ott.geometry import geometry, graph

sksparse = pytest.importorskip("sksparse")

# TODO(michalk8): mark tests as fast


def random_graph(
    n: int,
    p: float = 0.3,
    seed: Optional[int] = 0,
    *,
    return_laplacian: bool = False,
    directed: bool = False,
    fmt: Optional[Literal["csr", "csc", "coo"]] = None
) -> Union[jnp.ndarray, jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]:
  G = random_graphs.fast_gnp_random_graph(n, p, seed=seed, directed=directed)
  if not directed:
    assert nx.is_connected(G), "Generated graph is not connected."

  rng = np.random.RandomState(seed)
  for _, _, w in G.edges(data=True):
    w["weight"] = rng.uniform(0, 10)

  G = nx.linalg.laplacian_matrix(
      G
  ) if return_laplacian else nx.linalg.adjacency_matrix(G)

  if fmt is None:
    return jnp.asarray(G.A)

  G = getattr(G, f"to{fmt}")()
  return decomposition._scipy_sparse_to_jax(G)


def gt_geometry(
    G: Union[jnp.ndarray, jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO, nx.Graph],
    *,
    epsilon: float = 1e-2
) -> geometry.Geometry:
  if not isinstance(G, nx.Graph):
    if isinstance(G, (jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO)):
      G = G.todense()
    G = nx.from_numpy_array(np.asarray(G))

  n = len(G)
  cost = np.zeros((n, n), dtype=float)

  path = dict(
      shortest_paths.all_pairs_bellman_ford_path_length(G, weight="weight")
  )
  for i, src in enumerate(G.nodes):
    for j, tgt in enumerate(G.nodes):
      cost[i, j] = path[src][tgt] ** 2

  cost = jnp.asarray(cost)
  kernel = jnp.asarray(np.exp(-cost / epsilon))
  return geometry.Geometry(
      cost_matrix=cost, kernel_matrix=kernel, epsilon=epsilon
  )


class TestGraph:

  @pytest.mark.parametrize("empty", [False, True])
  def test_invalid_initialization(self, empty):
    with pytest.raises(AssertionError, match="Please provide"):
      if empty:
        _ = graph.Graph(graph=None, laplacian=None)
      else:
        G = random_graph(100)
        _ = graph.Graph(graph=G, laplacian=G)

  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_init_graph(self, fmt: Optional[str]):
    n = 100
    G = random_graph(n, fmt=fmt)

    geom = graph.Graph(G)

    assert geom.shape == (n, n)
    assert geom.graph is G
    assert geom._laplacian is None
    # compute the laplacian on the fly
    assert isinstance(geom.laplacian, type(geom.graph))

  @pytest.mark.parametrize("fmt", [None, "csr", "csc", "coo"])
  def test_init_laplacian(self, fmt: Optional[str]):
    n = 39
    L = random_graph(n, return_laplacian=True, fmt=fmt)

    geom = graph.Graph(laplacian=L)

    assert geom.shape == (n, n)
    assert geom.laplacian is L
    assert geom.graph is None

  @pytest.mark.parametrize("as_laplacian", [False, True])
  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_pytree(self, fmt: Optional[str], as_laplacian: bool):
    G = random_graph(25, fmt=fmt, return_laplacian=as_laplacian)

    geom1 = graph.Graph(laplacian=G) if as_laplacian else graph.Graph(graph=G)
    children, aux_data = geom1.tree_flatten()
    geom2 = graph.Graph.tree_unflatten(aux_data, children)

    assert geom1.graph is geom2.graph
    if as_laplacian:
      assert geom1.laplacian is geom2.laplacian
    elif fmt is None:
      np.testing.assert_array_equal(geom1.laplacian, geom2.laplacian)
    else:
      lap1, lap2 = geom1.laplacian, geom2.laplacian
      np.testing.assert_array_equal(lap1.data, lap2.data)
      np.testing.assert_array_equal(lap1.indices, lap2.indices)

    assert geom1.epsilon == geom2.epsilon
    assert geom1.n_steps == geom2.n_steps
    assert geom1.numerical_scheme == geom2.numerical_scheme
    assert geom1.directed == geom2.directed
    assert geom1._solver == geom2._solver

  @pytest.mark.parametrize("fmt", [None, "csr", "csc", "coo"])
  def test_solver(self, fmt: Optional[str]):
    n = 27
    G = random_graph(n, fmt=fmt, return_laplacian=True)
    geom1 = graph.Graph(laplacian=G)

    assert geom1._solver is None
    solver = geom1.solver
    assert geom1.solver is solver  # cached

    if geom1.is_sparse:
      assert isinstance(solver, decomposition.SparseCholeskySolver)
      L = solver.L  # trigger the computation
      assert L is None  # we're interested in the side-effect below
      assert hash(geom1) in decomposition.SparseCholeskySolver._FACTOR_CACHE
    else:
      assert isinstance(solver, decomposition.DenseCholeskySolver)
      np.testing.assert_array_equal(solver.A, geom1._M)
      assert isinstance(solver.L, jnp.ndarray)
      assert solver.L.shape == (n, n)

  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_kernel_is_symmetric_positive_definite(self, fmt: Optional[str]):
    geom = graph.Graph(graph=random_graph(65, fmt=fmt), epsilon=1e-3)

    tol = 1e-4 if geom.is_sparse else 5e-3
    kernel = geom.kernel_matrix

    np.testing.assert_allclose(kernel, kernel.T, rtol=tol, atol=tol)
    np.testing.assert_array_equal(jnp.linalg.eigvals(kernel) > 0., True)

  @pytest.mark.parametrize("as_laplacian", [False])
  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_automatic_t(self, fmt: Optional[str], as_laplacian: bool):
    G = random_graph(38, fmt=fmt, return_laplacian=as_laplacian)
    if as_laplacian:
      geom = graph.Graph(laplacian=G, epsilon=None)
    else:
      geom = graph.Graph(graph=G, epsilon=None)

    if fmt is None:
      expected = (jnp.sum(jnp.abs(G)) / jnp.sum(jnp.abs(G) > 0.)) ** 2
    else:
      expected = jnp.mean(G.data) ** 2
    actual = geom._t

    np.testing.assert_equal(actual, expected)

  @pytest.mark.parametrize("fmt", [None, "coo"])
  @pytest.mark.parametrize(
      "numerical_scheme", ["backward_euler", "crank_nicolson"]
  )
  def test_approximates_ground_truth_distances(
      self, rng: jnp.ndarray, numerical_scheme: str, fmt: Optional[str]
  ):
    eps, n_steps = 1e-4, 20
    G = random_graph(27, p=0.5, fmt=fmt)
    x = jax.random.normal(rng, (G.shape[0],))

    gt_geom = gt_geometry(G, epsilon=eps)
    graph_geom = graph.Graph(
        G, epsilon=eps, n_steps=n_steps, numerical_scheme=numerical_scheme
    )

    np.testing.assert_allclose(
        gt_geom.kernel_matrix, graph_geom.kernel_matrix, rtol=1e-2, atol=1e-2
    )
    for axis in [0, 1]:
      np.testing.assert_allclose(
          gt_geom.apply_kernel(x, axis=axis),
          graph_geom.apply_kernel(x, axis=axis),
          rtol=1e-2,
          atol=1e-2
      )

  @pytest.mark.parametrize("eps", [1e-4, 1e-3])
  def test_crank_nicolson_sparse_matches_dense(self, eps: float):
    G = random_graph(51, p=0.4, fmt=None)
    G_sp = jesp.BCOO.fromdense(G)

    dense_geom = graph.Graph(G, epsilon=eps, numerical_scheme="crank_nicolson")
    sparse_geom = graph.Graph(
        G_sp, epsilon=eps, numerical_scheme="crank_nicolson"
    )

    assert not dense_geom.is_sparse
    assert sparse_geom.is_sparse

    np.testing.assert_allclose(
        sparse_geom.kernel_matrix,
        dense_geom.kernel_matrix,
        rtol=eps * 1e2,
        atol=eps * 1e2,
    )

  @pytest.mark.parametrize("jit", [False, True])
  def test_directed_graph(self, jit: bool):

    def callback(geom: graph.Graph,
                 laplacian: bool) -> Union[jnp.ndarray, jesp.BCOO]:
      return geom.laplacian if laplacian else geom.graph

    G = random_graph(16, p=0.25, directed=True)
    if jit:
      callback = jax.jit(callback, static_argnums=1)

    geom = graph.Graph(G, directed=True)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(G, G.T)

    G = callback(geom, laplacian=False)
    L = callback(geom, laplacian=True)

    np.testing.assert_allclose(G, G.T)
    np.testing.assert_allclose(L, L.T)

  def test_factor_cache(self):
    # TODO(michalk8): finish me
    pass

  # Total memory allocated: 99.1MiB
  @pytest.mark.limit_memory("200 MB")
  def test_sparse_graph_memory(self, rng: jnp.ndarray):
    # use a graph with some structure for Cholesky to be faster
    G = nx.grid_graph((200, 200))  # 40 000 nodes
    L = nx.linalg.laplacian_matrix(G).tocsc()
    L = decomposition._scipy_sparse_to_jax(L)
    x = jax.random.normal(rng, (L.shape[0],))

    geom = graph.Graph(laplacian=L, n_steps=5)
    _ = geom.apply_kernel(x)

  @pytest.mark.parametrize("jit", [False, True])
  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_graph_sinkhorn(self, fmt: Optional[str], jit: bool):

    def callback(geom: geometry.Geometry) -> sinkhorn.SinkhornOutput:
      solver = sinkhorn.Sinkhorn(lse_mode=False)
      problem = linear_problems.LinearProblem(geom)
      return solver(problem)

    rtol = atol = 1e-3
    eps = 5e-3
    G = random_graph(11, p=0.35, fmt=fmt)

    gt_geom = gt_geometry(G, epsilon=eps)
    graph_geom = graph.Graph(G, epsilon=eps)
    fn = jax.jit(callback) if jit else callback

    gt_out = fn(gt_geom)
    graph_out = fn(graph_geom)

    assert gt_out.converged
    assert graph_out.converged
    np.testing.assert_allclose(gt_out.f, graph_out.f, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gt_out.g, graph_out.g, rtol=rtol, atol=atol)

    # TODO(michalk8): test output apply/materialize

  @pytest.mark.parametrize(
      "implicit_diff", [False, True], ids=["not-implicit", "implicit"]
  )
  def test_dense_graph_differentiability(
      self, rng: jnp.ndarray, implicit_diff: bool
  ):

    def callback(
        data: jnp.ndarray, rows: jnp.ndarray, cols: jnp.ndarray,
        shape: Tuple[int, int]
    ) -> float:
      G = jesp.BCOO((data, jnp.c_[rows, cols]), shape=shape).todense()

      geom = graph.Graph(G, epsilon=1.)
      solver = sinkhorn.Sinkhorn(lse_mode=False, **kwargs)
      problem = linear_problems.LinearProblem(geom)

      return solver(problem).reg_ot_cost

    if implicit_diff:
      kwargs = {"implicit_diff": implicit_lib.ImplicitDiff()}
    else:
      kwargs = {"implicit_diff": None}

    G, eps = random_graph(20, p=0.5, fmt="coo"), 1e-3
    w, rows, cols = G.data, G.indices[:, 0], G.indices[:, 1]
    v_w = jax.random.normal(rng, shape=w.shape)
    v_w = (v_w / jnp.linalg.norm(v_w, axis=-1, keepdims=True)) * eps

    grad_w = jax.grad(callback)(w, rows, cols, shape=G.shape)

    expected = callback(w + v_w, rows, cols,
                        G.shape) - callback(w - v_w, rows, cols, G.shape)
    actual = 2 * jnp.vdot(v_w, grad_w)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
