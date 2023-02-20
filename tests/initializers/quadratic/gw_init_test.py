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
"""Tests for Gromov-Wasserstein initializers."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import geometry, pointcloud
from ott.initializers.linear import initializers as lin_init
from ott.initializers.linear import initializers_lr
from ott.initializers.quadratic import initializers as quad_init
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein


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
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y)

    solver = gromov_wasserstein.GromovWasserstein(
        rank=rank, quad_initializer=None, kwargs_init=kwargs_init
    )
    initializer = solver.create_initializer(prob)

    assert isinstance(initializer, quad_init.LRQuadraticInitializer)
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
    assert isinstance(initializer, quad_init.QuadraticInitializer)

  @pytest.mark.parametrize("rank", [-1, 2])
  def test_explicitly_passing_initializer(self, rank: int):
    if rank == -1:
      linear_init = lin_init.SortingInitializer()
      q_init = quad_init.QuadraticInitializer()
    else:
      linear_init = initializers_lr.Rank2Initializer(rank)
      q_init = quad_init.LRQuadraticInitializer(linear_init)

    solver = gromov_wasserstein.GromovWasserstein(
        initializer=linear_init,
        quad_initializer=q_init,
    )

    assert solver.linear_ot_solver.initializer is linear_init
    assert solver.quad_initializer is q_init
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
    )
    geom_y = pointcloud.PointCloud(
        jax.random.normal(key3, (m, d2)),
        jax.random.normal(key4, (m, d2)),
        epsilon=eps,
    )
    problem = quadratic_problem.QuadraticProblem(geom_x, geom_y)
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

    random_cost = out_random.reg_gw_cost
    random_errors = out_random.errors[out_random.errors > -1]
    kmeans_cost = out_kmeans.reg_gw_cost
    kmeans_errors = out_kmeans.errors[out_kmeans.errors > -1]

    assert random_cost > kmeans_cost
    np.testing.assert_array_equal(random_errors >= 0., True)
    np.testing.assert_array_equal(kmeans_errors >= 0., True)
