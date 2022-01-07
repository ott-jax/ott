# coding=utf-8
# Copyright 2021 Google LLC.
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
import jax
import jax.numpy as jnp
import jax.test_util
from ott.core import problems
from ott.core import sinkhorn_lr
from ott.geometry import pointcloud


class SinkhornLRTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 4
    self.n = 68
    self.m = 123
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

  def test_euclidean_point_cloud(self):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y)
    ot_prob = problems.LinearProblem(geom, self.a, self.b)
    solver = sinkhorn_lr.LRSinkhorn(threshold=threshold, rank=10)
    costs = solver(ot_prob).costs
    self.assertTrue(jnp.isclose(costs[-2], costs[-1], rtol=threshold))
    cost_1 = costs[costs > -1][-1]

    solver = sinkhorn_lr.LRSinkhorn(threshold=threshold, rank=20)
    costs = solver(ot_prob).costs
    cost_2 = costs[costs > -1][-1]
    self.assertGreater(cost_1, cost_2)

if __name__ == '__main__':
  absltest.main()
