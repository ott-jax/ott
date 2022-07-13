# Copyright 2022 Google LLC.
#
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
"""Tests Sinkhorn Low-Rank solver with various initializations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import linear_problems, sinkhorn_lr
from ott.geometry import low_rank, pointcloud


class TestLRSinkhorn:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    self.dim = 4
    self.n = 29
    self.m = 27
    self.rng, *rngs = jax.random.split(rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))

    # #  adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @pytest.mark.fast.with_args(
      use_lrcgeom=[True, False],
      init_type=["rank_2", "random"],
      only_fast=0,
  )
  def test_euclidean_point_cloud(self, use_lrcgeom: bool, init_type: str):
    """Two point clouds, tested with 2 different initializations."""
    threshold = 1e-6
    geom = pointcloud.PointCloud(self.x, self.y)
    # This test to check LR can work both with LRCGeometries and regular ones
    if use_lrcgeom:
      geom = geom.to_LRCGeometry()
      assert isinstance(geom, low_rank.LRCGeometry)
    ot_prob = linear_problems.LinearProblem(geom, self.a, self.b)

    # Start with a low rank parameter
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=10,
        epsilon=0.0,
        init_type=init_type,
    )
    solved = solver(ot_prob)
    costs = solved.costs
    costs = costs[costs > -1]

    # Check convergence
    assert solved.converged
    assert jnp.isclose(costs[-2], costs[-1], rtol=threshold)

    # Store cost value.
    cost_1 = costs[-1]

    # Try with higher rank
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=14,
        epsilon=0.0,
        init_type=init_type,
    )
    out = solver(ot_prob)
    costs = out.costs
    cost_2 = costs[costs > -1][-1]
    # Ensure solution with more rank budget has lower cost (not guaranteed)
    assert cost_1 > cost_2

    # Ensure cost can still be computed on different geometry.
    other_geom = pointcloud.PointCloud(self.x, self.y + 0.3)
    cost_other = out.cost_at_geom(other_geom)
    assert cost_other > 0.0

    # Ensure cost is higher when using high entropy.
    # (Note that for small entropy regularizers, this can be the opposite
    # due to non-convexity of problem and benefit of adding regularizer.
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=14,
        epsilon=1e-1,
        init_type=init_type,
    )
    out = solver(ot_prob)
    costs = out.costs
    cost_3 = costs[costs > -1][-1]
    assert cost_3 > cost_2

  @pytest.mark.parametrize("axis", [0, 1])
  def test_output_apply_batch_size(self, axis: int):
    n_stack = 3
    threshold = 1e-6
    data = self.a if axis == 0 else self.b

    geom = pointcloud.PointCloud(self.x, self.y)
    ot_prob = linear_problems.LinearProblem(geom, self.a, self.b)
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=10,
        epsilon=0.0,
    )
    out = solver(ot_prob)

    gt = out.apply(data, axis=axis)
    pred = out.apply(jnp.stack([data] * n_stack), axis=axis)

    np.testing.assert_array_equal(gt.shape, (geom.shape[1 - axis],))
    np.testing.assert_array_equal(pred.shape, (n_stack, geom.shape[1 - axis]))
    np.testing.assert_allclose(
        pred, jnp.stack([gt] * n_stack), rtol=1e-6, atol=1e-6
    )
