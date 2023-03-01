# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from networkx.algorithms import shortest_paths
from networkx.generators import balanced_tree, random_graphs
from ott.geometry import geometry, graph
from ott.math import decomposition
from ott.problems.linear import linear_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn

# we mix both dense/sparse tests
_ = pytest.importorskip("sksparse", reason="Not supported for Python 3.11")


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
  return geometry.Geometry(cost_matrix=cost, kernel_matrix=kernel, epsilon=1.)


class TestGraph:

  @pytest.mark.parametrize("empty", [False, True])
  def test_invalid_initialization(self, empty):
    if empty:
      with pytest.raises(AssertionError, match="Please provide"):
        _ = graph.Graph(graph=None, laplacian=None)
    else:
      G = random_graph(100)
      L = random_graph(100, return_laplacian=True)
      with pytest.raises(AssertionError, match="Please provide"):
        _ = graph.Graph(graph=G, laplacian=L)

  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_init_graph(self, fmt: Optional[str]):
    n = 100
    G = random_graph(n, fmt=fmt)

    geom = graph.Graph(G)

    assert geom.shape == (n, n)
    assert geom.graph is G
    assert geom._lap is None
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

  @pytest.mark.fast()
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

    assert geom1.t == geom2.t
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

  @pytest.mark.fast.with_args("fmt", [None, "coo"], only_fast=0)
  def test_kernel_is_symmetric_positive_definite(
      self, rng: jax.random.PRNGKeyArray, fmt: Optional[str]
  ):
    n = 65
    x = jax.random.normal(rng, (n,))
    geom = graph.Graph(graph=random_graph(n, fmt=fmt), t=1e-3)

    tol = 1e-4 if geom.is_sparse else 5e-3
    kernel = geom.kernel_matrix

    vec0 = geom.apply_kernel(x, axis=0)
    vec1 = geom.apply_kernel(x, axis=1)
    vec_direct0 = geom.kernel_matrix.T @ x
    vec_direct1 = geom.kernel_matrix @ x

    # we symmetrize the kernel explicitly when materializing it, because
    # numerical error arise  for small `t` and `backward_euler`
    np.testing.assert_array_equal(kernel, kernel.T)
    np.testing.assert_array_equal(jnp.linalg.eigvals(kernel) > 0., True)
    # internally, the axis is ignored because the kernel is symmetric
    np.testing.assert_array_equal(vec0, vec1)
    np.testing.assert_array_equal(vec_direct0, vec_direct1)

    tol = tol if geom.is_sparse else 5 * tol
    np.testing.assert_allclose(vec0, vec_direct0, rtol=tol, atol=tol)
    np.testing.assert_allclose(vec1, vec_direct1, rtol=tol, atol=tol)

  @pytest.mark.parametrize("as_laplacian", [False])
  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_automatic_t(self, fmt: Optional[str], as_laplacian: bool):
    G = random_graph(38, fmt=fmt, return_laplacian=as_laplacian)
    if as_laplacian:
      geom = graph.Graph(laplacian=G, t=None)
    else:
      geom = graph.Graph(graph=G, t=None)

    if fmt is None:
      expected = (jnp.sum(G) / jnp.sum(G > 0.)) ** 2
    else:
      expected = jnp.mean(G.data) ** 2
    actual = geom.t

    np.testing.assert_equal(actual, expected)

  @pytest.mark.fast.with_args(
      numerical_scheme=["backward_euler", "crank_nicolson"],
      fmt=[None, "coo"],
      only_fast=0,
  )
  def test_approximates_ground_truth(
      self, rng: jax.random.PRNGKeyArray, numerical_scheme: str,
      fmt: Optional[str]
  ):
    eps, n_steps = 1e-5, 20
    G = random_graph(37, p=0.5, fmt=fmt)
    x = jax.random.normal(rng, (G.shape[0],))

    gt_geom = gt_geometry(G, epsilon=eps)
    graph_geom = graph.Graph(
        G, t=eps, n_steps=n_steps, numerical_scheme=numerical_scheme
    )

    np.testing.assert_allclose(
        gt_geom.kernel_matrix, graph_geom.kernel_matrix, rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        gt_geom.apply_kernel(x),
        graph_geom.apply_kernel(x),
        rtol=1e-2,
        atol=1e-2
    )

  @pytest.mark.fast.with_args(
      n_steps=[50, 100, 200],
      t=[1e-4, 1e-5],
      only_fast=0,
  )
  def test_crank_nicolson_more_stable(self, t: Optional[float], n_steps: int):
    tol = 5 * t
    G = nx.linalg.adjacency_matrix(balanced_tree(r=2, h=5))
    G = jnp.asarray(G.A, dtype=float)
    eye = jnp.eye(G.shape[0])

    be_geom = graph.Graph(
        G, t=t, n_steps=n_steps, numerical_scheme="backward_euler"
    )
    cn_geom = graph.Graph(
        G, t=t, n_steps=n_steps, numerical_scheme="crank_nicolson"
    )
    eps = jnp.finfo(eye.dtype).tiny

    be_cost = -t * jnp.log(be_geom.apply_kernel(eye) + eps)
    cn_cost = -t * jnp.log(cn_geom.apply_kernel(eye) + eps)

    np.testing.assert_allclose(cn_cost, cn_cost.T, rtol=tol, atol=tol)
    with pytest.raises(AssertionError):
      np.testing.assert_allclose(be_cost, be_cost.T, rtol=tol, atol=tol)

  @pytest.mark.parametrize("eps", [1e-4, 1e-3])
  def test_crank_nicolson_sparse_matches_dense(self, eps: float):
    G = random_graph(51, p=0.4, fmt=None)
    G_sp = jesp.BCOO.fromdense(G)

    dense_geom = graph.Graph(G, t=eps, numerical_scheme="crank_nicolson")
    sparse_geom = graph.Graph(G_sp, t=eps, numerical_scheme="crank_nicolson")

    assert not dense_geom.is_sparse
    assert sparse_geom.is_sparse

    np.testing.assert_allclose(
        sparse_geom.kernel_matrix,
        dense_geom.kernel_matrix,
        rtol=eps * 1e2,
        atol=eps * 1e2,
    )

  @pytest.mark.parametrize(("jit", "normalize"), [(False, True), (True, False)])
  def test_directed_graph(self, jit: bool, normalize: bool):

    def callback(geom: graph.Graph,
                 laplacian: bool) -> Union[jnp.ndarray, jesp.BCOO]:
      return geom.laplacian if laplacian else geom.graph

    G = random_graph(16, p=0.25, directed=True)
    fn = jax.jit(callback, static_argnums=1) if jit else callback

    geom = graph.Graph(G, directed=True, normalize=normalize)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(G, G.T)

    G = fn(geom, laplacian=False)
    L = fn(geom, laplacian=True)

    np.testing.assert_allclose(G, G.T, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(L, L.T, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("fmt", [None, "coo"])
  @pytest.mark.parametrize("normalize", [False, True])
  def test_normalize_laplacian(self, fmt: Optional[str], normalize: bool):

    def laplacian(geom: graph.Graph) -> jnp.ndarray:
      graph = geom.graph.todense() if geom.is_sparse else geom.graph
      data = G.sum(1)
      deg = jnp.diag(data)
      lap = deg - graph
      if not normalize:
        return lap
      inv_sqrt_deg = jnp.diag(jnp.where(data > 0., 1. / jnp.sqrt(data), 0.))
      return inv_sqrt_deg @ lap @ inv_sqrt_deg

    directed = False
    G = random_graph(51, p=0.35, directed=directed)
    geom = graph.Graph(G, directed=directed, normalize=normalize)

    expected = laplacian(geom)
    actual = geom.laplacian

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @pytest.mark.fast()
  def test_factor_cache_works(self, rng: jax.random.PRNGKeyArray):

    def timeit(fn: Callable[[Any], Any]) -> Callable[[Any], float]:

      def decorator(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        _ = fn(*args, **kwargs)
        return time.perf_counter() - start

      return decorator

    @timeit
    def callback(g: graph.Graph, x: jnp.ndarray) -> jnp.ndarray:
      return g.apply_kernel(x)

    n = 256
    G = random_graph(n, p=0.27, fmt="coo")
    x = jax.random.normal(rng, (n,))
    geom = graph.Graph(G)
    key = hash(geom)

    assert key not in decomposition.SparseCholeskySolver._FACTOR_CACHE

    time_non_cached = callback(geom, x)
    assert key in decomposition.SparseCholeskySolver._FACTOR_CACHE
    time_cached = callback(geom, x)

    assert time_cached < time_non_cached

  @pytest.mark.parametrize("jit", [False, True])
  @pytest.mark.skip(reason="Buggy")
  def test_factor_cache_unique(self, jit: bool):

    def callback(g: graph.Graph) -> decomposition.CholeskySolver:
      # run the decomposition
      return g.solver

    G1 = random_graph(12, p=0.7, fmt="coo")
    G2 = random_graph(13, p=0.6, fmt="coo")
    geom1 = graph.Graph(G1)
    geom2 = graph.Graph(G2)
    key1, key2 = hash(geom1), hash(geom2)
    fn = jax.jit(callback) if jit else callback

    assert key1 not in decomposition.SparseCholeskySolver._FACTOR_CACHE
    assert key2 not in decomposition.SparseCholeskySolver._FACTOR_CACHE

    _ = fn(geom1)
    _ = fn(geom2)

    assert key1 in decomposition.SparseCholeskySolver._FACTOR_CACHE
    assert key2 in decomposition.SparseCholeskySolver._FACTOR_CACHE

  # Total memory allocated: 99.1MiB
  @pytest.mark.fast()
  @pytest.mark.limit_memory("200 MB")
  def test_sparse_graph_memory(self, rng: jax.random.PRNGKeyArray):
    # use a graph with some structure for Cholesky to be faster
    G = nx.grid_graph((200, 200))  # 40 000 nodes
    L = nx.linalg.laplacian_matrix(G).tocsc()
    L = decomposition._scipy_sparse_to_jax(L)
    x = jax.random.normal(rng, (L.shape[0],))

    geom = graph.Graph(laplacian=L, n_steps=5)
    res = geom.apply_kernel(x)

    assert res.shape == x.shape

  @pytest.mark.fast.with_args(
      jit=[False, True],
      fmt=[None, "coo"],
      ids=["nojit-dense", "nojit-sparse", "jit-dense", "jit-sparse"],
      only_fast=0,
  )
  def test_graph_sinkhorn(
      self, rng: jax.random.PRNGKeyArray, fmt: Optional[str], jit: bool
  ):

    def callback(geom: geometry.Geometry) -> sinkhorn.SinkhornOutput:
      solver = sinkhorn.Sinkhorn(lse_mode=False)
      problem = linear_problem.LinearProblem(geom)
      return solver(problem)

    n, eps, tol = 11, 1e-5, 1e-3
    G = random_graph(n, p=0.35, fmt=fmt)
    x = jax.random.normal(rng, (n,))

    gt_geom = gt_geometry(G, epsilon=eps)
    graph_geom = graph.Graph(G, t=eps)

    fn = jax.jit(callback) if jit else callback

    gt_out = fn(gt_geom)
    graph_out = fn(graph_geom)

    assert gt_out.converged
    assert graph_out.converged
    np.testing.assert_allclose(
        graph_out.reg_ot_cost, gt_out.reg_ot_cost, rtol=tol, atol=tol
    )
    np.testing.assert_allclose(graph_out.f, gt_out.f, rtol=tol, atol=tol)
    np.testing.assert_allclose(graph_out.g, gt_out.g, rtol=tol, atol=tol)

    for axis in [0, 1]:
      y_gt = gt_out.apply(x, axis=axis)
      y_out = graph_out.apply(x, axis=axis)
      # note the high tolerance
      np.testing.assert_allclose(y_gt, y_out, rtol=5e-1, atol=5e-1)

    np.testing.assert_allclose(
        gt_out.matrix, graph_out.matrix, rtol=1e-1, atol=1e-1
    )

  @pytest.mark.parametrize(
      "implicit_diff",
      [False, True],
      ids=["not-implicit", "implicit"],
  )
  def test_dense_graph_differentiability(
      self, rng: jax.random.PRNGKeyArray, implicit_diff: bool
  ):

    def callback(
        data: jnp.ndarray, rows: jnp.ndarray, cols: jnp.ndarray,
        shape: Tuple[int, int]
    ) -> float:
      G = jesp.BCOO((data, jnp.c_[rows, cols]), shape=shape).todense()

      geom = graph.Graph(G, t=1.)
      solver = sinkhorn.Sinkhorn(lse_mode=False, **kwargs)
      problem = linear_problem.LinearProblem(geom)

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

  def test_tolerance_hilbert_metric(self, rng: jax.random.PRNGKeyArray):
    n, n_steps, t, tol = 256, 1000, 1e-4, 3e-4
    G = random_graph(n, p=0.15)
    x = jnp.abs(jax.random.normal(rng, (n,)))

    graph_geom_no_tol = graph.Graph(G, t=t, n_steps=n_steps, tol=-1)
    graph_geom_low_tol = graph.Graph(G, t=t, n_steps=n_steps, tol=2.5e-4)
    graph_geom_high_tol = graph.Graph(G, t=t, n_steps=n_steps, tol=1e-1)

    app_no_tol = graph_geom_no_tol.apply_kernel(x)
    app_low_tol = graph_geom_low_tol.apply_kernel(x)  # does 1 iteration
    app_high_tol = graph_geom_high_tol.apply_kernel(x)  # does 961 iterations

    np.testing.assert_allclose(app_no_tol, app_low_tol, rtol=tol, atol=tol)
    np.testing.assert_allclose(app_no_tol, app_high_tol, rtol=5e-2, atol=5e-2)
    with pytest.raises(AssertionError):
      np.testing.assert_allclose(app_no_tol, app_high_tol, rtol=tol, atol=tol)
