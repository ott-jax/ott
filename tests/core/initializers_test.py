# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for Sinkhorn initializers."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import gromov_wasserstein
from ott.core import initializers as init_lib
from ott.core import (
    initializers_lr,
    linear_problems,
    quad_initializers,
    quad_problems,
    sinkhorn,
    sinkhorn_lr,
)
from ott.geometry import geometry, low_rank, pointcloud


def create_sorting_problem(rng, n, epsilon=0.01, online=False):
  # define ot problem
  x_init = jnp.array([-1., 0, .22])
  y_init = jnp.array([0., 0, 1.1])
  x_rng, y_rng = jax.random.split(rng)

  x = jnp.concatenate([x_init, 10 + jnp.abs(jax.random.normal(x_rng, (n,)))])
  y = jnp.concatenate([y_init, 10 + jnp.abs(jax.random.normal(y_rng, (n,)))])

  x = jnp.sort(x)
  y = jnp.sort(y)

  n = len(x)
  m = len(y)
  a = jnp.ones(n) / n
  b = jnp.ones(m) / m

  batch_size = 3 if online else None
  geom = pointcloud.PointCloud(
      x.reshape(-1, 1),
      y.reshape(-1, 1),
      epsilon=epsilon,
      batch_size=batch_size
  )
  ot_problem = linear_problems.LinearProblem(geom=geom, a=a, b=b)

  return ot_problem


def create_ot_problem(rng, n, m, d, epsilon=0.01, online=False):
  # define ot problem
  x_rng, y_rng = jax.random.split(rng)

  mu_a = jnp.array([-1, 1]) * 5
  mu_b = jnp.array([0, 0])

  x = jax.random.normal(x_rng, (n, d)) + mu_a
  y = jax.random.normal(y_rng, (m, d)) + mu_b

  a = jnp.ones(n) / n
  b = jnp.ones(m) / m

  batch_size = 3 if online else None
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon, batch_size=batch_size)

  ot_problem = linear_problems.LinearProblem(geom=geom, a=a, b=b)
  return ot_problem


# define sinkhorn functions
@functools.partial(jax.jit, static_argnames=['lse_mode', 'vector_min'])
def run_sinkhorn_sort_init(
    x, y, a=None, b=None, epsilon=0.01, vector_min=True, lse_mode=True
):
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  sort_init = init_lib.SortingInitializer(vectorized_update=vector_min)
  out = sinkhorn.sinkhorn(
      geom,
      a=a,
      b=b,
      jit=True,
      potential_initializer=sort_init,
      lse_mode=lse_mode
  )
  return out


@functools.partial(jax.jit, static_argnames=['lse_mode'])
def run_sinkhorn(x, y, a=None, b=None, epsilon=0.01, lse_mode=True):
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  out = sinkhorn.sinkhorn(geom, a=a, b=b, jit=True, lse_mode=lse_mode)
  return out


@functools.partial(jax.jit, static_argnames=['lse_mode'])
def run_sinkhorn_gaus_init(x, y, a=None, b=None, epsilon=0.01, lse_mode=True):
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  out = sinkhorn.sinkhorn(
      geom,
      a=a,
      b=b,
      jit=True,
      potential_initializer=init_lib.GaussianInitializer(),
      lse_mode=lse_mode
  )
  return out


@pytest.mark.fast
class TestSinkhornInitializers:

  def test_init_pytree(self):

    @jax.jit
    def init_sort():
      init = init_lib.SortingInitializer()
      return init

    @jax.jit
    def init_gaus():
      init = init_lib.GaussianInitializer()
      return init

    _ = init_gaus()
    _ = init_sort()

  @pytest.mark.parametrize(
      "vector_min, lse_mode", [(True, True), (True, False), (False, True)]
  )
  def test_sorting_init(self, vector_min: bool, lse_mode: bool):
    """Tests sorting dual initializer."""
    rng = jax.random.PRNGKey(42)
    n = 500
    epsilon = 0.01

    ot_problem = create_sorting_problem(
        rng=rng, n=n, epsilon=epsilon, online=False
    )
    # run sinkhorn
    sink_out_base = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon
    )
    base_num_iter = jnp.sum(sink_out_base.errors > -1)

    sink_out_init = run_sinkhorn_sort_init(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        vector_min=vector_min,
        lse_mode=lse_mode
    )
    sort_num_iter = jnp.sum(sink_out_init.errors > -1)

    # check initializer is better or equal
    if lse_mode:
      assert base_num_iter >= sort_num_iter

  def test_sorting_init_online(self, rng: jnp.ndarray):
    n = 100
    epsilon = 0.01

    ot_problem = create_sorting_problem(
        rng=rng, n=n, epsilon=epsilon, online=True
    )
    sort_init = init_lib.SortingInitializer(vectorized_update=True)
    with pytest.raises(AssertionError, match=r"online"):
      sort_init.init_dual_a(ot_problem=ot_problem, lse_mode=True)

  def test_sorting_init_square_cost(self, rng: jnp.ndarray):
    n = 100
    m = 150
    d = 1
    epsilon = 0.01

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, online=False)
    sort_init = init_lib.SortingInitializer(vectorized_update=True)
    with pytest.raises(AssertionError, match=r"square"):
      sort_init.init_dual_a(ot_problem=ot_problem, lse_mode=True)

  def test_default_initializer(self, rng: jnp.ndarray):
    """Tests default initializer"""
    n = 200
    m = 200
    d = 2
    epsilon = 0.01

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, online=False)

    default_potential_a = init_lib.DefaultInitializer().init_dual_a(
        ot_problem=ot_problem, lse_mode=True
    )
    default_potential_b = init_lib.DefaultInitializer().init_dual_b(
        ot_problem=ot_problem, lse_mode=True
    )

    # check default is 0
    np.testing.assert_array_equal(0., default_potential_a)
    np.testing.assert_array_equal(0., default_potential_b)

  def test_gauss_pointcloud_geom(self, rng: jnp.ndarray):
    n = 200
    m = 200
    d = 2
    epsilon = 0.01

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, online=False)

    gaus_init = init_lib.GaussianInitializer()
    new_geom = geometry.Geometry(
        cost_matrix=ot_problem.geom.cost_matrix, epsilon=epsilon
    )
    ot_problem = linear_problems.LinearProblem(
        geom=new_geom, a=ot_problem.a, b=ot_problem.b
    )

    with pytest.raises(AssertionError, match=r"point cloud"):
      gaus_init.init_dual_a(ot_problem=ot_problem, lse_mode=True)

  @pytest.mark.parametrize('lse_mode', [True, False])
  def test_gauss_initializer(self, lse_mode, rng: jnp.ndarray):
    """Tests Gaussian initializer"""
    # define OT problem
    n = 200
    m = 200
    d = 2
    epsilon = 0.01

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, online=False)

    # run sinkhorn
    sink_out = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        lse_mode=lse_mode
    )
    base_num_iter = jnp.sum(sink_out.errors > -1)
    sink_out = run_sinkhorn_gaus_init(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        lse_mode=lse_mode
    )
    gaus_num_iter = jnp.sum(sink_out.errors > -1)

    # check initializer is better
    if lse_mode:
      assert base_num_iter >= gaus_num_iter


# TODO(michalk8): mark tests as fast
class TestLRInitializers:

  @pytest.mark.parametrize("kind", ["pc", "lrc", "geom"])
  def test_create_default_initializer(self, rng: jnp.ndarray, kind: str):
    n, d, rank = 110, 2, 3
    x = jax.random.normal(rng, (n, d))
    geom = pointcloud.PointCloud(x)

    if kind == "pc":
      pass
    elif kind == "lrc":
      geom = geom.to_LRCGeometry()
      assert isinstance(geom, low_rank.LRCGeometry)
    elif kind == "geom":
      geom = geometry.Geometry(geom.cost_matrix)
    else:
      raise NotImplementedError(geom)
    prob = linear_problems.LinearProblem(geom)

    solver = sinkhorn_lr.LRSinkhorn(rank=rank, initializer=None)
    initializer = solver.create_initializer(prob)

    assert initializer.rank == rank
    if kind in ("pc", "lrc"):
      assert isinstance(initializer, initializers_lr.KMeansInitializer)
    else:
      assert isinstance(initializer, initializers_lr.RandomInitializer)

    q, r, g = initializer(prob)

    assert q.shape == (n, rank)
    assert r.shape == (n, rank)
    assert g.shape == (rank,)

  def test_explicitly_passing_initializer(self):
    rank = 2
    initializer = initializers_lr.RandomInitializer(rank=rank)
    solver = sinkhorn_lr.LRSinkhorn(rank=rank, initializer=initializer)

    assert solver.create_initializer(prob="not used") is initializer

  @pytest.mark.parametrize(
      "initializer", ["random", "rank2", "k-means", "generalized-k-means"]
  )
  @pytest.mark.parametrize("partial_init", ["q", "r", "g"])
  def test_partial_initialization(
      self, rng: jnp.ndarray, initializer: str, partial_init: str
  ):
    n, d, rank = 100, 10, 6
    key1, key2, key3, key4 = jax.random.split(rng, 4)
    x = jax.random.normal(key1, (n, d))
    pc = pointcloud.PointCloud(x, epsilon=5e-1)
    prob = linear_problems.LinearProblem(pc)
    q_init = jax.random.normal(key2, (n, rank))
    r_init = jax.random.normal(key2, (n, rank))
    g_init = jax.random.normal(key2, (rank,))

    solver = sinkhorn_lr.LRSinkhorn(rank=rank, initializer=initializer)
    initializer = solver.create_initializer(prob)

    if partial_init == "q":
      q, _, _ = initializer(prob, q=q_init)
      np.testing.assert_array_equal(q, q_init)
    elif partial_init == "r":
      _, r, _ = initializer(prob, r=r_init)
      np.testing.assert_array_equal(r, r_init)
    elif partial_init == "g":
      _, _, g = initializer(prob, g=g_init)
      np.testing.assert_array_equal(g, g_init)
    else:
      raise NotImplementedError(partial_init)

  @pytest.mark.parametrize("rank", [2, 4, 10, 13])
  def test_generalized_k_means_has_correct_rank(
      self, rng: jnp.ndarray, rank: int
  ):
    n, d = 100, 10
    x = jax.random.normal(rng, (n, d))
    pc = pointcloud.PointCloud(x, epsilon=5e-1)
    prob = linear_problems.LinearProblem(pc)

    solver = sinkhorn_lr.LRSinkhorn(
        rank=rank, initializer="generalized-k-means"
    )
    initializer = solver.create_initializer(prob)

    q, r, g = initializer(prob)

    assert jnp.linalg.matrix_rank(q) == rank
    assert jnp.linalg.matrix_rank(r) == rank

  def test_generalized_k_means_matches_k_means(self, rng: jnp.ndarray):
    n, d, rank = 120, 15, 5
    eps = 1e-1
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, (n, d))
    y = jax.random.normal(key1, (n, d))

    pc = pointcloud.PointCloud(x, y, epsilon=eps)
    geom = geometry.Geometry(cost_matrix=pc.cost_matrix, epsilon=eps)
    pc_problem = linear_problems.LinearProblem(pc)
    geom_problem = linear_problems.LinearProblem(geom)

    solver = sinkhorn_lr.LRSinkhorn(
        rank=rank, initializer="k-means", max_iterations=5000
    )
    pc_out = solver(pc_problem)

    solver = sinkhorn_lr.LRSinkhorn(
        rank=rank, initializer="generalized-k-means", max_iterations=5000
    )
    geom_out = solver(geom_problem)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(pc_out.errors, geom_out.errors)

    np.testing.assert_allclose(
        pc_out.reg_ot_cost, geom_out.reg_ot_cost, atol=0.5, rtol=0.02
    )

  @pytest.mark.parametrize("epsilon", [0., 1e-1])
  def test_better_initialization_helps(self, rng: jnp.ndarray, epsilon: float):
    n, d, rank = 81, 13, 3
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, (n, d))
    y = jax.random.normal(key2, (n, d))
    pc = pointcloud.PointCloud(x, y, epsilon=5e-1)
    prob = linear_problems.LinearProblem(pc)

    solver_random = sinkhorn_lr.LRSinkhorn(
        rank=rank, epsilon=epsilon, initializer="random", max_iterations=5000
    )
    solver_init = sinkhorn_lr.LRSinkhorn(
        rank=rank, epsilon=epsilon, initializer="k-means", max_iterations=5000
    )

    out_random = solver_random(prob)
    out_init = solver_init(prob)

    assert out_random.converged
    assert out_init.converged
    # converged earlier
    assert (out_init.errors > -1).sum() < (out_random.errors > -1).sum()
    # converged to a better solution
    assert out_init.reg_ot_cost < out_random.reg_ot_cost


class TestQuadraticInitializers:

  @pytest.mark.parametrize("kind", ["pc", "lrc", "geom"])
  def test_create_default_lr_initializer(self, rng: jnp.ndarray, kind: str):
    n, d1, d2, rank = 150, 2, 3, 5
    eps = 1e-1
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, (n, d1))
    y = jax.random.normal(key1, (n, d2))
    kwargs_init = {"foo": "bar"}

    geom_x = pointcloud.PointCloud(x, epsilon=eps)
    geom_y = pointcloud.PointCloud(y, epsilon=eps)
    if kind == "pc":
      pass
    elif kind == "lrc":
      geom_x = geom_x.to_LRCGeometry()
      geom_y = geom_y.to_LRCGeometry()
    elif kind == "geom":
      geom_x = geometry.Geometry(geom_x.cost_matrix, epsilon=eps)
      geom_y = geometry.Geometry(geom_y.cost_matrix, epsilon=eps)
    else:
      raise NotImplementedError(kind)
    prob = quad_problems.QuadraticProblem(geom_x, geom_y)

    solver = gromov_wasserstein.GromovWasserstein(
        rank=rank, quad_initializer=None, kwargs_init=kwargs_init
    )
    initializer = solver.create_initializer(prob)

    assert isinstance(initializer, quad_initializers.LRQuadraticInitializer)
    assert initializer.rank == rank
    linear_init = initializer._linear_lr_initializer
    if kind in ("pc", "lrc"):
      assert isinstance(linear_init, initializers_lr.KMeansInitializer)
    else:
      assert isinstance(linear_init, initializers_lr.RandomInitializer)
    assert linear_init._kwargs == kwargs_init

  def test_non_lr_initializer(self):
    solver = gromov_wasserstein.GromovWasserstein(
        rank=-1, quad_initializer="not used"
    )
    initializer = solver.create_initializer(prob="not used")
    assert isinstance(initializer, quad_initializers.QuadraticInitializer)

  @pytest.mark.parametrize("rank", [-1, 2])
  def test_explicitly_passing_initializer(self, rank: int):
    if rank == -1:
      linear_init = init_lib.SortingInitializer()
      quad_init = quad_initializers.QuadraticInitializer()
    else:
      linear_init = initializers_lr.Rank2Initializer(rank)
      quad_init = quad_initializers.LRQuadraticInitializer(linear_init)

    solver = gromov_wasserstein.GromovWasserstein(
        # TODO(michalk8): rename to initializer
        potential_initializer=linear_init,
        quad_initializer=quad_init,
    )

    assert solver.linear_ot_solver.potential_initializer is linear_init
    assert solver.quad_initializer is quad_init
    if solver.is_low_rank:
      assert solver.quad_initializer.rank == rank

  @pytest.mark.parametrize("eps", [0., 1e-2])
  def test_gw_better_initialization_helps(self, rng: jnp.ndarray, eps: float):
    n, m, d1, d2, rank = 123, 124, 12, 10, 5
    key1, key2, key3, key4 = jax.random.split(rng, 4)

    geom_x = pointcloud.PointCloud(
        jax.random.normal(key1, (n, d1)),
        jax.random.normal(key2, (n, d1)),
        epsilon=eps,
        scale_cost=1.,
    )
    geom_y = pointcloud.PointCloud(
        jax.random.normal(key3, (m, d2)),
        jax.random.normal(key4, (m, d2)),
        epsilon=eps,
        scale_cost=1.,
    )
    problem = quad_problems.QuadraticProblem(geom_x, geom_y)
    solver_random = gromov_wasserstein.GromovWasserstein(
        rank=rank,
        initializer="random",
        quad_initializer="random",
        epsilon=eps,
        store_inner_errors=True,
    )
    solver_kmeans = gromov_wasserstein.GromovWasserstein(
        rank=rank,
        initializer="k-means",
        quad_initializer="k-means",
        epsilon=eps,
        store_inner_errors=True
    )

    out_random = solver_random(problem)
    out_kmeans = solver_kmeans(problem)

    assert out_random.reg_gw_cost - out_kmeans.reg_gw_cost >= 15.
    assert out_random.errors[0, 0] > out_kmeans.errors[0, 0]
