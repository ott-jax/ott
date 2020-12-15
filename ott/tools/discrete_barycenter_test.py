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

from ott.core.ground_geometry import grid
from ott.tools import discrete_barycenter as db


class SinkhornTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters([True], [False])
  def test_discrete_barycenter(self, debiased):
    """Tests the discrete barycenters on a 5x5x5 grid.
    Puts two masses on opposing ends of the hypercube with small noise in
    between. Check that their W barycenter sits (mostly) at the middle of the
    hypercube (e.g. index (5x5x5-1)/2)

    Args:
      debiased: bool, use (or not) debiasing as proposed in
      https://arxiv.org/abs/2006.02575
    """
    size = np.array([5, 5, 5])
    grid_3d = grid.Grid(grid_size=size, epsilon=0.01)
    a = np.ones(size)
    b = np.ones(size)
    a = a.ravel()
    b = b.ravel()
    a = jax.ops.index_update(a, 0, 10000)
    b = jax.ops.index_update(b, -1, 10000)
    a = a / np.sum(a)
    b = b / np.sum(b)

    bar = db.discrete_barycenter(grid_3d, a=np.stack((a, b))).histogram
    self.assertGreater(bar[(np.prod(size) - 1) // 2], 0.95)
    self.assertGreater(1, bar[(np.prod(size) - 1) // 2])

if __name__ == '__main__':
  absltest.main()
