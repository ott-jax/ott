# coding=utf-8
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
"""Tests for the Policy."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from ott.core import problems
from ott.core import sinkhorn_lr
from ott.geometry import pointcloud


class SinkhornLRTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 2
    self.n = 19
    self.m = 17
    self.rng, *rngs = jax.random.split(self.rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))

    # #  adding zero weights to test proper handling
    # a = a.at[0].set(0)
    # b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @parameterized.parameters([True], [False])
  def test_euclidean_point_cloud(self, use_lrcgeom):
    """Two point clouds, tested with various parameters."""
    init_type_arr = ["rank_2", "random"]
    for init_type in init_type_arr:
      threshold = 1e-9
      gamma = 100
      geom = pointcloud.PointCloud(self.x, self.y)
      if use_lrcgeom:
        geom = geom.to_LRCGeometry()
      ot_prob = problems.LinearProblem(geom, self.a, self.b)
      solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        gamma=gamma,
        rank=2,
        epsilon=0.0,
        init_type=init_type,
      )
      costs = solver(ot_prob).costs
      self.assertTrue(jnp.isclose(costs[-2], costs[-1], rtol=threshold))
      cost_1 = costs[costs > -1][-1]

      solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        gamma=gamma,
        rank=10,
        epsilon=0.0,
        init_type=init_type,
      )
      out = solver(ot_prob)
      costs = out.costs
      cost_2 = costs[costs > -1][-1]
      self.assertGreater(cost_1, cost_2)

      other_geom = pointcloud.PointCloud(self.x, self.y + 0.3)
      cost_other = out.cost_at_geom(other_geom)
      self.assertGreater(cost_other, 0.0)

      solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        gamma=gamma,
        rank=14,
        epsilon=1e-1,
        init_type=init_type,
      )
      out = solver(ot_prob)
      costs = out.costs
      cost_3 = costs[costs > -1][-1]

      solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        gamma=gamma,
        rank=14,
        epsilon=1e-3,
        init_type=init_type,
      )
      out = solver(ot_prob)
      costs = out.costs
      cost_4 = costs[costs > -1][-1]
      self.assertGreater(cost_3, cost_4)

if __name__ == "__main__":
  absltest.main()
