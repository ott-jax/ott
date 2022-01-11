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
"""Test Low-Rank Geometry."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.test_util
from ott.geometry import geometry
from ott.geometry import geometry_lr


class LRGeometryTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def test_apply(self):
    """Test application of cost to vec or matrix."""
    n, m, r = 17, 11, 7
    keys = jax.random.split(self.rng, 5)
    c1 = jax.random.normal(keys[0], (n, r))
    c2 = jax.random.normal(keys[1], (m, r))
    c = jnp.matmul(c1, c2.T)
    geom = geometry.Geometry(c)
    geom_lr = geometry_lr.LRCGeometry(c1, c2)
    for dim, axis in ((m, 1), (n, 0)):
      mat = jax.random.normal(keys[1], (dim, 2))
      self.assertAllClose(
          geom.apply_cost(mat, axis=axis),
          geom_lr.apply_cost(mat, axis=axis),
          rtol=1e-4)
      vec = jax.random.normal(keys[1], (dim,))
      self.assertAllClose(
          geom.apply_cost(vec, axis=axis),
          geom_lr.apply_cost(vec, axis=axis),
          rtol=1e-4)

  def test_add_lr_geoms(self):
    """Test application of cost to vec or matrix."""
    n, m, r, q = 17, 11, 7, 2
    keys = jax.random.split(self.rng, 5)
    c1 = jax.random.normal(keys[0], (n, r))
    c2 = jax.random.normal(keys[1], (m, r))
    d1 = jax.random.normal(keys[0], (n, q))
    d2 = jax.random.normal(keys[1], (m, q))

    c = jnp.matmul(c1, c2.T)
    d = jnp.matmul(d1, d2.T)
    geom = geometry.Geometry(c + d)

    geom_lr_c = geometry_lr.LRCGeometry(c1, c2)
    geom_lr_d = geometry_lr.LRCGeometry(d1, d2)
    geom_lr = geometry_lr.add_lrc_geom(geom_lr_c, geom_lr_d)

    for dim, axis in ((m, 1), (n, 0)):
      mat = jax.random.normal(keys[1], (dim, 2))
      self.assertAllClose(
          geom.apply_cost(mat, axis=axis),
          geom_lr.apply_cost(mat, axis=axis),
          rtol=1e-4)
      vec = jax.random.normal(keys[1], (dim,))
      self.assertAllClose(
          geom.apply_cost(vec, axis=axis),
          geom_lr.apply_cost(vec, axis=axis),
          rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
