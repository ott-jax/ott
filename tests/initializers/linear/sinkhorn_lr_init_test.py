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

from ott.geometry import pointcloud
from ott.initializers.linear import initializers_lr
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr


class TestLRInitializers:

  @pytest.mark.fast.with_args("rank", [2, 4, 10, 13], only_fast=True)
  def test_generalized_k_means_has_correct_rank(
      self, rng: jax.Array, rank: int
  ):
    n, d = 27, 5
    x = jax.random.normal(rng, (n, d))
    pc = pointcloud.PointCloud(x, epsilon=0.5)
    prob = linear_problem.LinearProblem(pc)

    initializer = initializers_lr.GeneralizedKMeansInitializer(rank)
    q, r, g = initializer(prob)

    assert jnp.linalg.matrix_rank(q) == rank
    assert jnp.linalg.matrix_rank(r) == rank

  @pytest.mark.parametrize("epsilon", [0.0, 1e-1])
  def test_better_initialization_helps(self, rng: jax.Array, epsilon: float):
    n, d, rank = 81, 13, 3
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (n, d))
    y = jax.random.normal(rng2, (n, d))
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

    out_random = solver_random(prob)
    out_init = solver_init(prob)

    assert out_random.converged
    assert out_init.converged
    # converged earlier
    np.testing.assert_array_less((out_init.errors > -1).sum(),
                                 (out_random.errors > -1).sum())
    # converged to a better solution
    np.testing.assert_array_less(out_init.reg_ot_cost, out_random.reg_ot_cost)
