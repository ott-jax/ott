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
from ott.core.ground_geometry import geometry
from ott.core.ground_geometry import pointcloud


class SinkhornTest(jax.test_util.JaxTestCase):

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
    self.a = a / np.sum(a)
    self.b = b / np.sum(b)

  @parameterized.named_parameters(
      dict(
          testcase_name='lse-Leh-mom',
          lse_mode=True,
          momentum_strategy='Lehmann',
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='lse-no-mom',
          lse_mode=True,
          momentum_strategy=1.0,
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='lse-high-mom',
          lse_mode=True,
          momentum_strategy=1.5,
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='scal-Leh-mom',
          lse_mode=False,
          momentum_strategy='Lehmann',
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='scal-no-mom',
          lse_mode=False,
          momentum_strategy=1.0,
          inner_iterations=10,
          norm_error=1,
      ),
      dict(
          testcase_name='scal-high-mom',
          lse_mode=False,
          momentum_strategy=1.5,
          inner_iterations=10,
          norm_error=1,
      ),
      dict(
          testcase_name='lse-Leh-1',
          lse_mode=True,
          momentum_strategy='Lehmann',
          inner_iterations=1,
          norm_error=2),
      dict(
          testcase_name='lse-Leh-13',
          lse_mode=True,
          momentum_strategy='Lehmann',
          inner_iterations=13,
          norm_error=3,
      ),
      dict(
          testcase_name='lse-Leh-24',
          lse_mode=True,
          momentum_strategy='Lehmann',
          inner_iterations=24,
          norm_error=4,
      ))
  def test_euclidean_point_cloud(self, lse_mode, momentum_strategy,
                                 inner_iterations, norm_error):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloudGeometry(self.x, self.y, epsilon=0.1)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        momentum_strategy=momentum_strategy,
        inner_iterations=inner_iterations,
        norm_error=norm_error,
        lse_mode=lse_mode).errors
    err = errors[errors > -1][-1]
    self.assertGreater(threshold, err)

  def test_euclidean_point_cloud_min_iter(self):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloudGeometry(self.x, self.y, epsilon=0.1)
    errors = sinkhorn.sinkhorn(
        geom, a=self.a, b=self.b, threshold=threshold, min_iterations=34).errors
    err = errors[np.logical_and(errors > -1, np.isfinite(errors))][-1]
    self.assertGreater(threshold, err)
    self.assertEqual(np.inf, errors[0])
    self.assertEqual(np.inf, errors[1])
    self.assertEqual(np.inf, errors[2])
    self.assertGreater(errors[3], 0)

  def test_geom_vs_point_cloud(self):
    """Two point clouds vs. simple cost_matrix execution of sinkorn."""
    geom = pointcloud.PointCloudGeometry(self.x, self.y)
    geom_2 = geometry.Geometry(geom.cost_matrix)
    f = sinkhorn.sinkhorn(geom, a=self.a, b=self.b).f
    f_2 = sinkhorn.sinkhorn(geom_2, a=self.a, b=self.b).f
    self.assertAllClose(f, f_2)

  @parameterized.parameters([True], [False])
  def test_euclidean_point_cloud_parallel_weights(self, lse_mode):
    """Two point clouds, parallel execution for batched histograms."""
    self.rng, *rngs = jax.random.split(self.rng, 2)
    batch = 4
    a = jax.random.uniform(rngs[0], (batch, self.n))
    b = jax.random.uniform(rngs[0], (batch, self.m))
    a = a / np.sum(a, axis=1)[:, np.newaxis]
    b = b / np.sum(b, axis=1)[:, np.newaxis]
    threshold = 1e-3
    geom = pointcloud.PointCloudGeometry(
        self.x, self.y, epsilon=0.1, online=True)
    errors = sinkhorn.sinkhorn(
        geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode).errors
    err = errors[errors > -1][-1]
    self.assertGreater(np.min(threshold - err), 0)

  @parameterized.parameters([True], [False])
  def test_online_euclidean_point_cloud(self, lse_mode):
    """Testing the online way to handle geometry."""
    threshold = 1e-3
    geom = pointcloud.PointCloudGeometry(
        self.x, self.y, epsilon=0.1, online=True)
    errors = sinkhorn.sinkhorn(
        geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode).errors
    err = errors[errors > -1][-1]
    self.assertGreater(threshold, err)

  @parameterized.parameters([True], [False])
  def test_online_vs_batch_euclidean_point_cloud(self, lse_mode):
    """Comparing online vs batch geometry."""
    threshold = 1e-3
    eps = 0.1
    online_geom = pointcloud.PointCloudGeometry(
        self.x, self.y, epsilon=eps, online=True)
    batch_geom = pointcloud.PointCloudGeometry(self.x, self.y, epsilon=eps)
    out_online = sinkhorn.sinkhorn(
        online_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode)
    out_batch = sinkhorn.sinkhorn(
        batch_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode)

    # Checks regularized transport costs match.
    self.assertAllClose(out_online.reg_ot_cost, out_batch.reg_ot_cost)
    # check regularized transport matrices match
    self.assertAllClose(
        online_geom.transport_from_potentials(out_online.f, out_online.g),
        batch_geom.transport_from_potentials(out_batch.f, out_batch.g))

  def test_apply_transport_geometry(self):
    """Applying transport matrix P on vector without instantiating P."""
    n, m, d = 160, 230, 6
    keys = jax.random.split(self.rng, 6)
    x = jax.random.uniform(keys[0], (n, d))
    y = jax.random.uniform(keys[1], (m, d))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / np.sum(a)
    b = b / np.sum(b)
    transport_t_vec_a = [None, None, None, None]
    transport_vec_b = [None, None, None, None]

    batch_b = 8

    vec_a = jax.random.normal(keys[4], (n,))
    vec_b = jax.random.normal(keys[5], (batch_b, m))

    # test with lse_mode and online = True / False
    for j, lse_mode in enumerate([True, False]):
      for i, online in enumerate([True, False]):
        geom = pointcloud.PointCloudGeometry(x, y, online=online, epsilon=0.2)
        sink = sinkhorn.sinkhorn(geom, a, b, lse_mode=lse_mode)

        transport_t_vec_a[i + 2 * j] = geom.apply_transport_from_potentials(
            sink.f, sink.g, vec_a, axis=0)
        transport_vec_b[i + 2 * j] = geom.apply_transport_from_potentials(
            sink.f, sink.g, vec_b, axis=1)

        transport = geom.transport_from_potentials(sink.f, sink.g)

        self.assertAllClose(
            transport_t_vec_a[i + 2 * j],
            np.dot(transport.T, vec_a).T,
            rtol=1e-3,
            atol=1e-3)
        self.assertAllClose(
            transport_vec_b[i + 2 * j],
            np.dot(transport, vec_b.T).T,
            rtol=1e-3,
            atol=1e-3)

    for i in range(4):
      self.assertAllClose(
          transport_vec_b[i], transport_vec_b[0], rtol=1e-3, atol=1e-3)
      self.assertAllClose(
          transport_t_vec_a[i], transport_t_vec_a[0], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
