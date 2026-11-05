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
import jax.random as jr
import numpy as np

from ott.geometry import pointcloud
from ott.initializers.linear import initializers_lr
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr
from tests import _utils


class TestLRInitializers:

  @pytest.mark.fast.with_args("rank", [2, 4, 10, 13], only_fast=True)
  def test_generalized_k_means_has_correct_rank(
      self, rng: jax.Array, rank: int
  ):
    n, d = 27, 5
    x = jr.normal(rng, (n, d))
    pc = pointcloud.PointCloud(x, epsilon=0.5)
    prob = linear_problem.LinearProblem(pc)

    initializer = initializers_lr.GeneralizedKMeansInitializer(rank)
    q, r, _ = initializer(prob)

    assert jnp.linalg.matrix_rank(q) == rank
    assert jnp.linalg.matrix_rank(r) == rank

  @pytest.mark.fast()
  def test_rank2_with_zero_weights(self, rng: jax.Array):
    # zero weights must not collapse the factors to the product coupling
    rank = 6
    clouds = _utils.random_clouds(rng, zero_a=(0,), zero_b=(3,))
    prob = clouds.problem()

    q, r, g = initializers_lr.Rank2Initializer(rank)(prob)

    for factor, marginal in ((q, clouds.a), (r, clouds.b)):
      assert jnp.linalg.matrix_rank(factor) == 2
      assert jnp.all(factor >= 0.0)
      np.testing.assert_allclose(factor.sum(1), marginal, rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(factor.sum(0), g, rtol=1e-6, atol=1e-6)

  def test_better_initialization_helps(self, rng: jax.Array):
    n, d, rank = 81, 15, 3
    epsilon = 1e-1
    rng1, rng2, rng3 = jr.split(rng, 3)
    x = jr.normal(rng1, (n, d))
    y = jr.normal(rng2, (n, d))
    pc = pointcloud.PointCloud(x, y, epsilon=5e-1)
    prob = linear_problem.LinearProblem(pc)

    solver_random = sinkhorn_lr.LRSinkhorn(
        rank=rank,
        epsilon=epsilon,
        initializer=initializers_lr.RandomInitializer(rank),
        max_iterations=10000,
    )
    solver_init = sinkhorn_lr.LRSinkhorn(
        rank=rank,
        epsilon=epsilon,
        initializer=initializers_lr.KMeansInitializer(rank),
        max_iterations=10000
    )

    out_random = solver_random(prob, rng=rng3)
    out_init = solver_init(prob)

    assert out_random.converged
    assert out_init.converged
    # converged earlier
    assert out_init.n_iters <= out_random.n_iters
    # converged to a better solution
    assert out_init.reg_ot_cost <= out_random.reg_ot_cost
