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
"""Tests for apply_cost and apply_kernel."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import geometry
from ott.geometry import pointcloud
from ott.geometry.costs import Cosine

class ApplyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def test_general_cost_fn(self):
    """Test non-vec cost apply to vec."""
    n, m, p, b = 5, 8, 10, 7
    keys = jax.random.split(self.rng, 5)
    x = jax.random.normal(keys[0], (n, p))
    y = jax.random.normal(keys[1], (m, p)) + 1
    vec0 = jax.random.normal(keys[2], (n, b))
    vec1 = jax.random.normal(keys[3], (m, b))

    geom = pointcloud.PointCloud(x, y, cost_fn=Cosine(), online=False)
    cost = geom.cost_matrix
    prod0 = geom.apply_cost(vec0, axis=0)
    prod1 = geom.apply_cost(vec1, axis=1)
    
    geom = geometry.Geometry(cost)
    prod0_geom = geom.apply_cost(vec0, axis=0)
    prod1_geom = geom.apply_cost(vec1, axis=1)

    np.testing.assert_allclose(prod0_geom, prod0, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(prod1_geom, prod1, rtol=1e-03, atol=1e-02)


if __name__ == '__main__':
  absltest.main()
