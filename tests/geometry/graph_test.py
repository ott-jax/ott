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
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from jax.experimental import sparse
from networkx.algorithms import shortest_paths
from networkx.generators import balanced_tree, random_graphs
from ott.geometry import geometry, graph
from ott.problems.linear import linear_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn


def random_graph(
    n: int,
    p: float = 0.3,
    seed: Optional[int] = 0,
    *,
    return_laplacian: bool = False,
    directed: bool = False,
) -> jnp.ndarray:
  G = random_graphs.fast_gnp_random_graph(n, p, seed=seed, directed=directed)
  if not directed:
    assert nx.is_connected(G), "Generated graph is not connected."

  rng = np.random.RandomState(seed)
  for _, _, w in G.edges(data=True):
    w["weight"] = rng.uniform(0, 10)

  G = nx.linalg.laplacian_matrix(
      G
  ) if return_laplacian else nx.linalg.adjacency_matrix(G)

  return jnp.asarray(G.toarray())


def gt_geometry(G: jnp.ndarray, *, epsilon: float = 1e-2) -> geometry.Geometry:
  if not isinstance(G, nx.Graph):
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

  def test_kernel_is_symmetric_positive_definite(
      self, rng: jax.random.PRNGKeyArray
  ):
    n, tol = 65, 0.02
    x = jax.random.normal(rng, (n,))
    geom = graph.Graph.from_graph(random_graph(n), t=1e-3)

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

    np.testing.assert_allclose(vec0, vec_direct0, rtol=tol, atol=tol)
    np.testing.assert_allclose(vec1, vec_direct1, rtol=tol, atol=tol)

  def test_automatic_t(self):
    G = random_graph(38, return_laplacian=False)
    geom = graph.Graph.from_graph(G, t=None)

    expected = (jnp.sum(G) / jnp.sum(G > 0.)) ** 2
    actual = geom.t
    np.testing.assert_equal(actual, expected)

  @pytest.mark.fast.with_args(
      numerical_scheme=["backward_euler", "crank_nicolson"],
      only_fast=0,
  )
  def test_approximates_ground_truth(
      self,
      rng: jax.random.PRNGKeyArray,
      numerical_scheme: Literal["backward_euler", "crank_nicolson"],
  ):
    eps, n_steps = 1e-5, 20
    G = random_graph(37, p=0.5)
    x = jax.random.normal(rng, (G.shape[0],))

    gt_geom = gt_geometry(G, epsilon=eps)
    graph_geom = graph.Graph.from_graph(
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
    G = jnp.asarray(G.toarray(), dtype=float)
    eye = jnp.eye(G.shape[0])

    be_geom = graph.Graph.from_graph(
        G, t=t, n_steps=n_steps, numerical_scheme="backward_euler"
    )
    cn_geom = graph.Graph.from_graph(
        G, t=t, n_steps=n_steps, numerical_scheme="crank_nicolson"
    )
    eps = jnp.finfo(eye.dtype).tiny

    be_cost = -t * jnp.log(be_geom.apply_kernel(eye) + eps)
    cn_cost = -t * jnp.log(cn_geom.apply_kernel(eye) + eps)

    np.testing.assert_allclose(cn_cost, cn_cost.T, rtol=tol, atol=tol)
    with pytest.raises(AssertionError):
      np.testing.assert_allclose(be_cost, be_cost.T, rtol=tol, atol=tol)

  @pytest.mark.parametrize(("jit", "normalize"), [(False, True), (True, False)])
  def test_directed_graph(self, jit: bool, normalize: bool):

    def create_graph(G: jnp.ndarray) -> graph.Graph:
      return graph.Graph.from_graph(G, directed=True, normalize=normalize)

    G = random_graph(16, p=0.25, directed=True)
    create_fn = jax.jit(create_graph) if jit else create_graph
    geom = create_fn(G)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(G, G.T)

    L = geom.laplacian

    with pytest.raises(AssertionError):
      # make sure that original graph was directed
      np.testing.assert_allclose(G, G.T, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(L, L.T, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("directed", [False, True])
  @pytest.mark.parametrize("normalize", [False, True])
  def test_normalize_laplacian(self, directed: bool, normalize: bool):

    def laplacian(G: jnp.ndarray) -> jnp.ndarray:
      if directed:
        G = G + G.T

      data = jnp.sum(G, axis=1)
      lap = jnp.diag(data) - G
      if normalize:
        inv_sqrt_deg = jnp.diag(
            jnp.where(data > 0.0, 1.0 / jnp.sqrt(data), 0.0)
        )
        return inv_sqrt_deg @ lap @ inv_sqrt_deg
      return lap

    G = random_graph(51, p=0.35, directed=directed)
    geom = graph.Graph.from_graph(G, directed=directed, normalize=normalize)

    expected = laplacian(G)
    actual = geom.laplacian

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @pytest.mark.fast.with_args(jit=[False, True], only_fast=0)
  def test_graph_sinkhorn(self, rng: jax.random.PRNGKeyArray, jit: bool):

    def callback(geom: geometry.Geometry) -> sinkhorn.SinkhornOutput:
      solver = sinkhorn.Sinkhorn(lse_mode=False)
      problem = linear_problem.LinearProblem(geom)
      return solver(problem)

    n, eps, tol = 11, 1e-5, 1e-3
    G = random_graph(n, p=0.35)
    x = jax.random.normal(rng, (n,))

    gt_geom = gt_geometry(G, epsilon=eps)
    graph_geom = graph.Graph.from_graph(G, t=eps)

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
      G = sparse.BCOO((data, jnp.c_[rows, cols]), shape=shape).todense()

      geom = graph.Graph.from_graph(G, t=1.)
      solver = sinkhorn.Sinkhorn(lse_mode=False, **kwargs)
      problem = linear_problem.LinearProblem(geom)

      return solver(problem).reg_ot_cost

    if implicit_diff:
      kwargs = {"implicit_diff": implicit_lib.ImplicitDiff()}
    else:
      kwargs = {"implicit_diff": None}

    eps = 1e-3
    G = random_graph(20, p=0.5)
    G = sparse.BCOO.fromdense(G)

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

    graph_no_tol = graph.Graph.from_graph(G, t=t, n_steps=n_steps, tol=-1)
    graph_low_tol = graph.Graph.from_graph(G, t=t, n_steps=n_steps, tol=2.5e-4)
    graph_high_tol = graph.Graph.from_graph(G, t=t, n_steps=n_steps, tol=1e-1)

    app_no_tol = graph_no_tol.apply_kernel(x)
    app_low_tol = graph_low_tol.apply_kernel(x)  # does 1 iteration
    app_high_tol = graph_high_tol.apply_kernel(x)  # does 961 iterations

    np.testing.assert_allclose(app_no_tol, app_low_tol, rtol=tol, atol=tol)
    np.testing.assert_allclose(app_no_tol, app_high_tol, rtol=5e-2, atol=5e-2)
    with pytest.raises(AssertionError):
      np.testing.assert_allclose(app_no_tol, app_high_tol, rtol=tol, atol=tol)
