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
from typing import Optional, Union

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from networkx.algorithms import shortest_paths
from networkx.generators import balanced_tree, random_graphs
from ott.geometry import geodesic, geometry, graph
from ott.problems.linear import linear_problem
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


def gt_geometry(
    G: Union[jnp.ndarray, nx.Graph],
    *,
    epsilon: float = 1e-2
) -> geometry.Geometry:
  if not isinstance(G, nx.Graph):
    G = nx.from_numpy_array(np.asarray(G))

  n = len(G)
  cost = np.zeros((n, n))

  path = dict(
      shortest_paths.all_pairs_bellman_ford_path_length(G, weight="weight")
  )
  for i, src in enumerate(G.nodes):
    for j, tgt in enumerate(G.nodes):
      cost[i, j] = path[src][tgt] ** 2

  cost = jnp.asarray(cost)
  kernel = jnp.asarray(np.exp(-cost / epsilon))
  return geometry.Geometry(cost_matrix=cost, kernel_matrix=kernel, epsilon=1.0)


def exact_heat_kernel(G: jnp.ndarray, normalize: bool = False, t: float = 10):
  degree = jnp.sum(G, axis=1)
  L = jnp.diag(degree) - G
  if normalize:
    inv_sqrt_deg = jnp.diag(
        jnp.where(degree > 0.0, 1.0 / jnp.sqrt(degree), 0.0)
    )
    L = inv_sqrt_deg @ L @ inv_sqrt_deg

  e, v = jnp.linalg.eigh(L)
  e = jnp.clip(e, 0)

  return v @ jnp.diag(jnp.exp(-t * e)) @ v.T


class TestGeodesic:

  def test_kernel_is_symmetric_positive_definite(
      self,
      rng: jax.Array,
  ):
    n, tol = 100, 0.02
    t = 1
    order = 50
    x = jax.random.normal(rng, (n,))
    G = random_graph(n)
    geom = geodesic.Geodesic.from_graph(G, t=t, order=order)

    kernel = geom.kernel_matrix

    vec0 = geom.apply_kernel(x, axis=0)
    vec1 = geom.apply_kernel(x, axis=1)
    vec_direct0 = geom.kernel_matrix.T @ x
    vec_direct1 = geom.kernel_matrix @ x

    # we symmetrize the kernel explicitly when materializing it, because
    # numerical errors can make it non-symmetric.
    np.testing.assert_allclose(kernel, kernel.T, rtol=tol, atol=tol)
    eigenvalues = jnp.linalg.eigvals(kernel)
    neg_eigenvalues = eigenvalues[eigenvalues < 0]
    # check that the negative eigenvalues are all very small
    np.testing.assert_array_less(jnp.abs(neg_eigenvalues), 1e-3)
    # internally, the axis is ignored because the kernel is symmetric
    np.testing.assert_allclose(vec0, vec1, rtol=tol, atol=tol)
    np.testing.assert_allclose(vec_direct0, vec_direct1, rtol=tol, atol=tol)

    np.testing.assert_allclose(vec0, vec_direct0, rtol=tol, atol=tol)
    np.testing.assert_allclose(vec1, vec_direct1, rtol=tol, atol=tol)

    cost_matrix = geom.cost_matrix
    np.testing.assert_allclose(cost_matrix, cost_matrix.T, rtol=tol, atol=tol)
    np.testing.assert_array_less(0, cost_matrix)

  @pytest.mark.fast.with_args(
      order=[50, 100, 200],
      t=[1e-4, 1e-5],
  )
  def test_approximates_ground_truth(self, t: Optional[float], order: int):
    tol = 1e-2
    G = nx.linalg.adjacency_matrix(balanced_tree(r=2, h=5))
    G = jnp.asarray(G.toarray().astype(float))
    eye = jnp.eye(G.shape[0])
    eps = jnp.finfo(eye.dtype).tiny

    gt_geom = gt_geometry(G, epsilon=eps)

    geo = geodesic.Geodesic.from_graph(G, t=t, order=order)

    np.testing.assert_allclose(
        gt_geom.kernel_matrix, geo.kernel_matrix, rtol=tol, atol=tol
    )

  @pytest.mark.parametrize(("jit", "normalize"), [(False, True), (True, False)])
  def test_directed_graph(self, jit: bool, normalize: bool):

    def create_graph(G: jnp.ndarray) -> graph.Graph:
      return geodesic.Geodesic.from_graph(G, directed=True, normalize=normalize)

    G = random_graph(16, p=0.25, directed=True)
    create_fn = jax.jit(create_graph) if jit else create_graph
    geom = create_fn(G)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(G, G.T)

    L = geom.scaled_laplacian

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
    geom = geodesic.Geodesic.from_graph(
        G, directed=directed, normalize=normalize
    )

    expected = laplacian(G)
    eigenvalues = jnp.linalg.eigvals(expected)
    eigval = jnp.max(eigenvalues)
    #rescale the laplacian
    expected = 2 * expected / eigval if eigval > 2 else expected

    actual = geom.scaled_laplacian

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @pytest.mark.fast.with_args(jit=[False, True], only_fast=0)
  def test_geo_sinkhorn(self, rng: jax.Array, jit: bool):

    def callback(geom: geometry.Geometry) -> sinkhorn.SinkhornOutput:
      solver = sinkhorn.Sinkhorn(lse_mode=False)
      problem = linear_problem.LinearProblem(geom)
      return solver(problem)

    n, eps, tol = 11, 1e-5, 1e-3
    G = random_graph(n, p=0.35)
    x = jax.random.normal(rng, (n,))

    gt_geom = gt_geometry(G, epsilon=eps)
    graph_geom = geodesic.Geodesic.from_graph(G, t=eps / 4.0)

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
      y_gt = gt_out.apply(x, axis=axis, lse_mode=False)
      y_out = graph_out.apply(x, axis=axis, lse_mode=False)
      # note the high tolerance
      np.testing.assert_allclose(y_gt, y_out, rtol=5e-1, atol=5e-1)

    np.testing.assert_allclose(
        gt_out.matrix, graph_out.matrix, rtol=1e-1, atol=1e-1
    )

  def test_geometry_differentiability(self, rng: jax.Array):

    def callback(geom: geodesic.Geodesic) -> float:
      solver = sinkhorn.Sinkhorn(lse_mode=False)
      problem = linear_problem.LinearProblem(geom)
      return solver(problem).reg_ot_cost

    eps = 1e-3
    G = random_graph(20, p=0.5)
    geom = geodesic.Geodesic.from_graph(G, t=1.0)

    v_w = jax.random.normal(rng, shape=G.shape)
    v_w = (v_w / jnp.linalg.norm(v_w, axis=-1, keepdims=True)) * eps

    grad_sl = jax.grad(callback)(geom).scaled_laplacian
    geom__finite_right = geodesic.Geodesic.from_graph(G + v_w, t=1.0)
    geom__finite_left = geodesic.Geodesic.from_graph(G - v_w, t=1.0)

    expected = callback(geom__finite_right) - callback(geom__finite_left)
    actual = 2 * jnp.vdot(v_w, grad_sl)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("normalize", [False, True])
  @pytest.mark.parametrize("t", [5, 10, 50])
  @pytest.mark.parametrize("order", [20, 30, 40])
  def test_heat_approx(self, normalize: bool, t: float, order: int):
    G = random_graph(20, p=0.5)
    exact = exact_heat_kernel(G, normalize=normalize, t=t)
    geom = geodesic.Geodesic.from_graph(
        G, t=t, order=order, normalize=normalize
    )
    approx = geom.apply_kernel(jnp.eye(G.shape[0]))
    np.testing.assert_allclose(exact, approx, rtol=1e-1, atol=1e-1)
