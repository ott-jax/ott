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
import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import geometry, pointcloud
from ott.initializers.linear import initializers_lr
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr


class TestLRInitializers:

  def test_explicit_initializer(self):
    rank = 2
    initializer = initializers_lr.RandomInitializer(rank=rank)
    solver = sinkhorn_lr.LRSinkhorn(rank=rank, initializer=initializer)

    assert solver.create_initializer(prob="not used") is initializer

  @pytest.mark.parametrize(
      "initializer", ["random", "rank2", "k-means", "generalized-k-means"]
  )
  @pytest.mark.parametrize("partial_init", ["q", "r", "g"])
  def test_partial_initialization(
      self, rng: jax.Array, initializer: str, partial_init: str
  ):
    n, d, rank = 27, 5, 6
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    x = jax.random.normal(rng1, (n, d))
    pc = pointcloud.PointCloud(x, epsilon=5e-1)
    prob = linear_problem.LinearProblem(pc)
    q_init = jax.random.normal(rng2, (n, rank))
    r_init = jax.random.normal(rng2, (n, rank))
    g_init = jax.random.normal(rng2, (rank,))

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

  @pytest.mark.fast.with_args("rank", [2, 4, 10, 13], only_fast=True)
  def test_generalized_k_means_has_correct_rank(
      self, rng: jax.Array, rank: int
  ):
    n, d = 27, 5
    x = jax.random.normal(rng, (n, d))
    pc = pointcloud.PointCloud(x, epsilon=0.5)
    prob = linear_problem.LinearProblem(pc)

    solver = sinkhorn_lr.LRSinkhorn(
        rank=rank, initializer="generalized-k-means"
    )
    initializer = solver.create_initializer(prob)

    q, r, g = initializer(prob)

    assert jnp.linalg.matrix_rank(q) == rank
    assert jnp.linalg.matrix_rank(r) == rank

  def test_generalized_k_means_matches_k_means(self, rng: jax.Array):
    n, d, rank = 27, 7, 5
    eps = 1e-1
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (n, d))
    y = jax.random.normal(rng1, (n, d))

    pc = pointcloud.PointCloud(x, y, epsilon=eps)
    geom = geometry.Geometry(cost_matrix=pc.cost_matrix, epsilon=eps)
    pc_problem = linear_problem.LinearProblem(pc)
    geom_problem = linear_problem.LinearProblem(geom)

    solver = sinkhorn_lr.LRSinkhorn(
        rank=rank, initializer="k-means", max_iterations=5000
    )
    pc_out = solver(pc_problem)

    solver = sinkhorn_lr.LRSinkhorn(
        rank=rank, initializer="generalized-k-means", max_iterations=5000
    )
    geom_out = solver(geom_problem)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(pc_out.costs, geom_out.costs)

    np.testing.assert_allclose(
        pc_out.reg_ot_cost, geom_out.reg_ot_cost, atol=0.5, rtol=0.02
    )

  @pytest.mark.parametrize("epsilon", [0.0, 1e-1])
  def test_better_initialization_helps(self, rng: jax.Array, epsilon: float):
    n, d, rank = 81, 13, 3
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (n, d))
    y = jax.random.normal(rng2, (n, d))
    pc = pointcloud.PointCloud(x, y, epsilon=5e-1)
    prob = linear_problem.LinearProblem(pc)

    solver_random = sinkhorn_lr.LRSinkhorn(
        rank=rank, epsilon=epsilon, initializer="random", max_iterations=10000
    )
    solver_init = sinkhorn_lr.LRSinkhorn(
        rank=rank, epsilon=epsilon, initializer="k-means", max_iterations=10000
    )

    out_random = solver_random(prob)
    out_init = solver_init(prob)

    assert out_random.converged
    assert out_init.converged
    # converged earlier
    np.testing.assert_array_less((out_init.errors > -1).sum(),
                                 (out_random.errors > -1).sum())
    # converged to a better solution
    np.testing.assert_array_less(out_init.reg_ot_cost, out_random.reg_ot_cost)
