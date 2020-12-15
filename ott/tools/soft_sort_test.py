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
"""Tests for the soft sort tools."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as np
import jax.test_util

from ott.tools import soft_sort


class SoftSortTest(jax.test_util.JaxTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self._dim = 4
    self._num_points = 20

  @parameterized.parameters([(20,)], [(20, 1)])
  def test_sort_one_array(self, shape):
    x = jax.random.uniform(self.rng, shape)
    xs = soft_sort.softsort(x, axis=0)
    self.assertEqual(x.shape, xs.shape)
    self.assertTrue(np.alltrue(np.diff(xs, axis=0) >= 0.0))

  def test_sort_batch(self):
    x = jax.random.uniform(self.rng, (32, 20, 12, 8))
    xs = soft_sort.softsort(x, axis=1)
    self.assertEqual(x.shape, xs.shape)
    self.assertTrue(np.alltrue(np.diff(xs, axis=1) >= 0.0))

  def test_rank_one_array(self):
    x = jax.random.uniform(self.rng, (20,))
    ranks = soft_sort.softranks(x, epsilon=0.005)
    self.assertEqual(x.shape, ranks.shape)
    expected_ranks = np.argsort(np.argsort(x, axis=0), axis=0).astype(float)
    self.assertAllClose(ranks, expected_ranks, atol=0.9, rtol=0.1)

  @parameterized.parameters([0.2, 0.5, 0.9])
  def test_quantile(self, level):
    x = np.linspace(0.0, 1.0, 100)
    q = soft_sort.softquantile(
        x,
        level=level,
        weight=0.05,
        epsilon=1e-3,
        sinkhorn_kw={'lse_mode': True})
    self.assertAlmostEqual(q, level, places=1)

if __name__ == '__main__':
  absltest.main()
