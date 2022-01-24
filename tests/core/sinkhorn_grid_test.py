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
"""Tests for Sinkhorn when applied on a grid."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np

from ott.core import sinkhorn
from ott.geometry import grid
from ott.geometry import pointcloud


class SinkhornGridTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters([True], [False])
  def test_separable_grid(self, lse_mode):
    """Two histograms in a grid of size 5 x 6 x 7  in the hypercube^3."""
    grid_size = (5, 6, 7)
    keys = jax.random.split(self.rng, 2)
    a = jax.random.uniform(keys[0], grid_size)
    b = jax.random.uniform(keys[1], grid_size)
    #  adding zero weights  to test proper handling, then ravel.
    a = a.at[0].set(0).ravel()
    a = a / jnp.sum(a)
    b = b.at[3].set(0).ravel()
    b = b / jnp.sum(b)

    threshold = 0.01
    geom = grid.Grid(grid_size=grid_size, epsilon=0.1)
    errors = sinkhorn.sinkhorn(
        geom, a=a, b=b, threshold=threshold, lse_mode=lse_mode).errors
    err = errors[jnp.isfinite(errors)][-1]
    self.assertGreater(threshold, err)

  @parameterized.parameters([True], [False])
  def test_grid_vs_euclidean(self, lse_mode):
    grid_size = (5, 6, 7)
    keys = jax.random.split(self.rng, 2)
    a = jax.random.uniform(keys[0], grid_size)
    b = jax.random.uniform(keys[1], grid_size)
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)
    epsilon = 0.1
    geometry_grid = grid.Grid(grid_size=grid_size, epsilon=epsilon)
    x, y, z = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = jnp.stack([
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geometry_mat = pointcloud.PointCloud(xyz, xyz, epsilon=epsilon)
    out_mat = sinkhorn.sinkhorn(geometry_mat, a=a, b=b, lse_mode=lse_mode)
    out_grid = sinkhorn.sinkhorn(geometry_grid, a=a, b=b, lse_mode=lse_mode)
    self.assertAllClose(out_mat.reg_ot_cost, out_grid.reg_ot_cost)

  @parameterized.parameters([True], [False])
  def test_apply_transport_grid(self, lse_mode):
    grid_size = (5, 6, 7)
    keys = jax.random.split(self.rng, 3)
    a = jax.random.uniform(keys[0], grid_size)
    b = jax.random.uniform(keys[1], grid_size)
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)
    geom_grid = grid.Grid(grid_size=grid_size, epsilon=0.1)
    x, y, z = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = jnp.stack([
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geom_mat = pointcloud.PointCloud(xyz, xyz, epsilon=0.1)
    sink_mat = sinkhorn.sinkhorn(geom_mat, a=a, b=b, lse_mode=lse_mode)
    sink_grid = sinkhorn.sinkhorn(geom_grid, a=a, b=b, lse_mode=lse_mode)

    batch_a = 3
    batch_b = 4
    vec_a = jax.random.normal(keys[4], [batch_a,
                                        np.prod(np.array(grid_size))])
    vec_b = jax.random.normal(keys[4], [batch_b,
                                        np.prod(grid_size)])

    vec_a = vec_a / jnp.sum(vec_a, axis=1)[:, jnp.newaxis]
    vec_b = vec_b / jnp.sum(vec_b, axis=1)[:, jnp.newaxis]

    mat_transport_t_vec_a = geom_mat.apply_transport_from_potentials(
        sink_mat.f, sink_mat.g, vec_a, axis=0)
    mat_transport_vec_b = geom_mat.apply_transport_from_potentials(
        sink_mat.f, sink_mat.g, vec_b, axis=1)

    grid_transport_t_vec_a = geom_grid.apply_transport_from_potentials(
        sink_grid.f, sink_grid.g, vec_a, axis=0)
    grid_transport_vec_b = geom_grid.apply_transport_from_potentials(
        sink_grid.f, sink_grid.g, vec_b, axis=1)

    self.assertAllClose(mat_transport_t_vec_a, grid_transport_t_vec_a)
    self.assertAllClose(mat_transport_vec_b, grid_transport_vec_b)
    self.assertIsNot(jnp.any(jnp.isnan(mat_transport_t_vec_a)), True)

  def test_apply_cost(self):
    grid_size = (5, 6, 7)

    geom_grid = grid.Grid(grid_size=grid_size, epsilon=0.1)
    x, y, z = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = jnp.stack([
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geom_mat = pointcloud.PointCloud(xyz, xyz, epsilon=0.1)

    vec = jax.random.uniform(self.rng, grid_size).ravel()
    self.assertAllClose(geom_mat.apply_cost(vec),
                        geom_grid.apply_cost(vec))

    self.assertAllClose(
        geom_grid.apply_cost(vec)[:, 0], np.dot(geom_mat.cost_matrix.T, vec))


if __name__ == '__main__':
  absltest.main()
