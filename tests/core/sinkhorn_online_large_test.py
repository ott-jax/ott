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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
from ott.core import sinkhorn
from ott.geometry import pointcloud


class SinkhornOnlineTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 4
    self.n = 20000
    self.m = 20002
    self.rng, *rngs = jax.random.split(self.rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))
    #  adding zero weights to test proper handling
    a = jax.ops.index_update(a, 0, 0)
    b = jax.ops.index_update(b, 3, 0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @parameterized.parameters([True], [False])
  def test_euclidean_point_cloud(self, lse_mode):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-1
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=1, online=True)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode,
        implicit_differentiation=True).errors
    err = errors[errors > -1][-1]
    self.assertGreater(threshold, err)


if __name__ == '__main__':
  absltest.main()
