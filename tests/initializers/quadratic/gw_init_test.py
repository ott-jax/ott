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
import jax
import numpy as np
import pytest
from ott.geometry import pointcloud
from ott.initializers.linear import initializers as lin_init
from ott.initializers.linear import initializers_lr
from ott.initializers.quadratic import initializers as quad_init
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr


class TestQuadraticInitializers:

  def test_explicit_initializer(self):
    linear_init = lin_init.SortingInitializer()
    q_init = quad_init.QuadraticInitializer()
    solver = gromov_wasserstein.GromovWasserstein(
        initializer=linear_init,
        quad_initializer=q_init,
    )

    assert solver.create_initializer("not used") is q_init
    assert solver.linear_ot_solver.initializer is linear_init
    assert solver.quad_initializer is q_init

  def test_explicit_initializer_lr(self):
    rank = 10
    q_init = initializers_lr.Rank2Initializer(rank)
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=rank, initializer=q_init
    )

    assert solver.create_initializer("not used") is q_init
    assert solver.initializer is q_init
    assert solver.initializer.rank == rank

  @pytest.mark.parametrize("eps", [0., 1e-2])
  def test_gw_better_initialization_helps(
      self, rng: jax.random.PRNGKeyArray, eps: float
  ):
    n, m, d1, d2, rank = 83, 84, 8, 6, 4
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    geom_x = pointcloud.PointCloud(
        jax.random.normal(rng1, (n, d1)),
        jax.random.normal(rng2, (n, d1)),
        epsilon=eps,
    )
    geom_y = pointcloud.PointCloud(
        jax.random.normal(rng3, (m, d2)),
        jax.random.normal(rng4, (m, d2)),
        epsilon=eps,
    )
    problem = quadratic_problem.QuadraticProblem(geom_x, geom_y)
    solver_random = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=rank,
        initializer="random",
        epsilon=eps,
    )
    solver_kmeans = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=rank,
        initializer="k-means",
        epsilon=eps,
    )

    out_random = solver_random(problem)
    out_kmeans = solver_kmeans(problem)

    random_cost = out_random.reg_gw_cost
    random_errors = out_random.errors[out_random.errors > -1]
    kmeans_cost = out_kmeans.reg_gw_cost
    kmeans_errors = out_kmeans.errors[out_kmeans.errors > -1]

    np.testing.assert_array_less(kmeans_cost, random_cost)
    np.testing.assert_array_equal(random_errors >= 0., True)
    np.testing.assert_array_equal(kmeans_errors >= 0., True)
