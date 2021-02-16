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
import jax.numpy as np
import jax.test_util
from ott.core import sinkhorn
from ott.core.geometry import costs
from ott.core.geometry import pointcloud


class SinkhornTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.eps = 1.0
    self.n = 27
    self.m = 13
    self.dim = 7
    self.rngs = jax.random.split(jax.random.PRNGKey(0), 6)

    x = jax.random.normal(self.rngs[0], (self.n, self.dim, self.dim))
    y = jax.random.normal(self.rngs[1], (self.m, self.dim, self.dim))

    sig_x = np.matmul(x, np.transpose(x, (0, 2, 1))) / 100
    sig_y = np.matmul(y, np.transpose(y, (0, 2, 1))) / 100

    m_x = jax.random.uniform(self.rngs[2], (self.n, self.dim))
    m_y = jax.random.uniform(self.rngs[3], (self.m, self.dim))

    self.x = np.concatenate((m_x.reshape((self.n, -1)),
                             sig_x.reshape((self.n, -1))), axis=1)
    self.y = np.concatenate((m_y.reshape((self.m, -1)),
                             sig_y.reshape((self.m, -1))), axis=1)
    a = jax.random.uniform(self.rngs[4], (self.n,)) + .1
    b = jax.random.uniform(self.rngs[5], (self.m,)) + .1
    self.a = a / np.sum(a)
    self.b = b / np.sum(b)

  @parameterized.named_parameters(
      dict(testcase_name='ker-batch', lse_mode=False, online=False))
  def test_bures_point_cloud(self, lse_mode, online):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(
        self.x, self.y,
        cost_fn=costs.Bures(dimension=self.dim, regularization=1e-4),
        online=online,
        epsilon=self.eps)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        lse_mode=lse_mode).errors
    err = errors[errors > -1][-1]
    self.assertGreater(threshold, err)

if __name__ == '__main__':
  absltest.main()
