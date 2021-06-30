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
from ott.geometry import costs
from ott.geometry import geometry
from ott.geometry import pointcloud


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

    #  adding zero weights to test proper handling
    a = jax.ops.index_update(a, 0, 0)
    b = jax.ops.index_update(b, 3, 0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @parameterized.named_parameters(
      dict(
          testcase_name='lse-Leh-mom',
          lse_mode=True,
          momentum=1.0,
          chg_momentum_from=29,
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='lse-small-mom',
          lse_mode=True,
          momentum=1.01,
          chg_momentum_from=0,
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='lse-high-mom',
          lse_mode=True,
          momentum=1.5,
          chg_momentum_from=0,
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='scal-Leh-mom',
          lse_mode=False,
          momentum=1.01,
          chg_momentum_from=30,
          inner_iterations=10,
          norm_error=1),
      dict(
          testcase_name='scal-no-mom',
          lse_mode=False,
          momentum=1.0,
          chg_momentum_from=0,
          inner_iterations=10,
          norm_error=1,
      ),
      dict(
          testcase_name='scal-high-mom',
          lse_mode=False,
          momentum=1.5,
          chg_momentum_from=0,
          inner_iterations=10,
          norm_error=1,
      ),
      dict(
          testcase_name='lse-Leh-1',
          lse_mode=True,
          momentum=1.0,
          chg_momentum_from=60,
          inner_iterations=1,
          norm_error=2),
      dict(
          testcase_name='lse-Leh-13',
          lse_mode=True,
          momentum=1.0,
          chg_momentum_from=40,
          inner_iterations=13,
          norm_error=3,
      ),
      dict(
          testcase_name='lse-Leh-24',
          lse_mode=True,
          momentum=1.0,
          chg_momentum_from=12,
          inner_iterations=24,
          norm_error=4,
      ))
  def test_euclidean_point_cloud(self, lse_mode, momentum, chg_momentum_from,
                                 inner_iterations, norm_error):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        momentum=momentum,
        chg_momentum_from=chg_momentum_from,
        inner_iterations=inner_iterations,
        norm_error=norm_error,
        lse_mode=lse_mode).errors
    err = errors[errors > -1][-1]
    self.assertGreater(threshold, err)

  def test_autoepsilon(self):
    """Check that with auto-epsilon, dual potentials scale."""
    scale = 13
    geom_1 = pointcloud.PointCloud(self.x, self.y)
    f_1 = sinkhorn.sinkhorn(geom_1, a=self.a, b=self.b).f
    geom_2 = pointcloud.PointCloud(scale * self.x, scale * self.y)
    f_2 = sinkhorn.sinkhorn(geom_2, a=self.a, b=self.b).f
    # recentering to remove ambiguity on equality up to additive constant.
    f_1 -= jnp.mean(f_1[jnp.isfinite(f_1)])
    f_2 -= jnp.mean(f_2[jnp.isfinite(f_2)])

    self.assertAllClose(f_1 * scale**2, f_2, rtol=1e-3, atol=1e-3)

  def test_euclidean_point_cloud_min_iter(self):
    """Testing the min_iterations parameter."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1)
    errors = sinkhorn.sinkhorn(
        geom, a=self.a, b=self.b, threshold=threshold, min_iterations=34,
        implicit_differentiation=False).errors
    err = errors[jnp.logical_and(errors > -1, jnp.isfinite(errors))][-1]
    self.assertGreater(threshold, err)
    self.assertEqual(jnp.inf, errors[0])
    self.assertEqual(jnp.inf, errors[1])
    self.assertEqual(jnp.inf, errors[2])
    self.assertGreater(errors[3], 0)

  def test_geom_vs_point_cloud(self):
    """Two point clouds vs. simple cost_matrix execution of sinkorn."""
    geom_1 = pointcloud.PointCloud(self.x, self.y)
    geom_2 = geometry.Geometry(geom_1.cost_matrix)

    f_1 = sinkhorn.sinkhorn(geom_1, a=self.a, b=self.b).f
    f_2 = sinkhorn.sinkhorn(geom_2, a=self.a, b=self.b).f
    # recentering to remove ambiguity on equality up to additive constant.
    f_1 -= jnp.mean(f_1[jnp.isfinite(f_1)])
    f_2 -= jnp.mean(f_2[jnp.isfinite(f_2)])

    self.assertAllClose(f_1, f_2)

  @parameterized.parameters([True], [False])
  def test_euclidean_point_cloud_parallel_weights(self, lse_mode):
    """Two point clouds, parallel execution for batched histograms."""
    self.rng, *rngs = jax.random.split(self.rng, 2)
    batch = 4
    a = jax.random.uniform(rngs[0], (batch, self.n))
    b = jax.random.uniform(rngs[0], (batch, self.m))
    a = a / jnp.sum(a, axis=1)[:, jnp.newaxis]
    b = b / jnp.sum(b, axis=1)[:, jnp.newaxis]
    threshold = 1e-3
    geom = pointcloud.PointCloud(
        self.x, self.y, epsilon=0.1, online=True)
    errors = sinkhorn.sinkhorn(
        geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode).errors
    err = errors[errors > -1][-1]
    self.assertGreater(jnp.min(threshold - err), 0)

  @parameterized.parameters([True], [False])
  def test_online_euclidean_point_cloud(self, lse_mode):
    """Testing the online way to handle geometry."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(
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
    online_geom = pointcloud.PointCloud(
        self.x, self.y, epsilon=eps, online=True)
    online_geom_euc = pointcloud.PointCloud(
        self.x,
        self.y,
        cost_fn=costs.Euclidean(),
        epsilon=eps,
        online=True)

    batch_geom = pointcloud.PointCloud(self.x, self.y, epsilon=eps)
    batch_geom_euc = pointcloud.PointCloud(
        self.x,
        self.y,
        cost_fn=costs.Euclidean(),
        epsilon=eps)

    out_online = sinkhorn.sinkhorn(
        online_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode)
    out_batch = sinkhorn.sinkhorn(
        batch_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode)
    out_online_euc = sinkhorn.sinkhorn(
        online_geom_euc,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode)
    out_batch_euc = sinkhorn.sinkhorn(
        batch_geom_euc,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode)

    # Checks regularized transport costs match.
    self.assertAllClose(out_online.reg_ot_cost, out_batch.reg_ot_cost)
    # check regularized transport matrices match
    self.assertAllClose(
        online_geom.transport_from_potentials(out_online.f, out_online.g),
        batch_geom.transport_from_potentials(out_batch.f, out_batch.g))

    self.assertAllClose(
        online_geom_euc.transport_from_potentials(out_online_euc.f,
                                                  out_online_euc.g),
        batch_geom_euc.transport_from_potentials(out_batch_euc.f,
                                                 out_batch_euc.g))

    self.assertAllClose(
        batch_geom.transport_from_potentials(out_batch.f, out_batch.g),
        batch_geom_euc.transport_from_potentials(out_batch_euc.f,
                                                 out_batch_euc.g))

  def test_apply_transport_geometry_from_potentials(self):
    """Applying transport matrix P on vector without instantiating P."""
    n, m, d = 160, 230, 6
    keys = jax.random.split(self.rng, 6)
    x = jax.random.uniform(keys[0], (n, d))
    y = jax.random.uniform(keys[1], (m, d))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    transport_t_vec_a = [None, None, None, None]
    transport_vec_b = [None, None, None, None]

    batch_b = 8

    vec_a = jax.random.normal(keys[4], (n,))
    vec_b = jax.random.normal(keys[5], (batch_b, m))

    # test with lse_mode and online = True / False
    for j, lse_mode in enumerate([True, False]):
      for i, online in enumerate([True, False]):
        geom = pointcloud.PointCloud(x, y, online=online, epsilon=0.2)
        sink = sinkhorn.sinkhorn(geom, a, b, lse_mode=lse_mode)

        transport_t_vec_a[i + 2 * j] = geom.apply_transport_from_potentials(
            sink.f, sink.g, vec_a, axis=0)
        transport_vec_b[i + 2 * j] = geom.apply_transport_from_potentials(
            sink.f, sink.g, vec_b, axis=1)

        transport = geom.transport_from_potentials(sink.f, sink.g)

        self.assertAllClose(
            transport_t_vec_a[i + 2 * j],
            jnp.dot(transport.T, vec_a).T,
            rtol=1e-3,
            atol=1e-3)
        self.assertAllClose(
            transport_vec_b[i + 2 * j],
            jnp.dot(transport, vec_b.T).T,
            rtol=1e-3,
            atol=1e-3)

    for i in range(4):
      self.assertAllClose(
          transport_vec_b[i], transport_vec_b[0], rtol=1e-3, atol=1e-3)
      self.assertAllClose(
          transport_t_vec_a[i], transport_t_vec_a[0], rtol=1e-3, atol=1e-3)

  def test_apply_transport_geometry_from_scalings(self):
    """Applying transport matrix P on vector without instantiating P."""
    n, m, d = 160, 230, 6
    keys = jax.random.split(self.rng, 6)
    x = jax.random.uniform(keys[0], (n, d))
    y = jax.random.uniform(keys[1], (m, d))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    transport_t_vec_a = [None, None, None, None]
    transport_vec_b = [None, None, None, None]

    batch_b = 8

    vec_a = jax.random.normal(keys[4], (n,))
    vec_b = jax.random.normal(keys[5], (batch_b, m))

    # test with lse_mode and online = True / False
    for j, lse_mode in enumerate([True, False]):
      for i, online in enumerate([True, False]):
        geom = pointcloud.PointCloud(x, y, online=online, epsilon=0.2)
        sink = sinkhorn.sinkhorn(geom, a, b, lse_mode=lse_mode)

        u = geom.scaling_from_potential(sink.f)
        v = geom.scaling_from_potential(sink.g)

        transport_t_vec_a[i + 2 * j] = geom.apply_transport_from_scalings(
            u, v, vec_a, axis=0)
        transport_vec_b[i + 2 * j] = geom.apply_transport_from_scalings(
            u, v, vec_b, axis=1)

        transport = geom.transport_from_scalings(u, v)

        self.assertAllClose(
            transport_t_vec_a[i + 2 * j],
            jnp.dot(transport.T, vec_a).T,
            rtol=1e-3,
            atol=1e-3)
        self.assertAllClose(
            transport_vec_b[i + 2 * j],
            jnp.dot(transport, vec_b.T).T,
            rtol=1e-3,
            atol=1e-3)
        self.assertIsNot(jnp.any(jnp.isnan(transport_t_vec_a[i + 2 * j])),
                         True)
    for i in range(4):
      self.assertAllClose(
          transport_vec_b[i], transport_vec_b[0], rtol=1e-3, atol=1e-3)
      self.assertAllClose(
          transport_t_vec_a[i], transport_t_vec_a[0], rtol=1e-3, atol=1e-3)

  @parameterized.parameters([True], [False])
  def test_restart(self, lse_mode):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-4
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.01)
    out = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode,
        inner_iterations=1)
    errors = out.errors
    err = errors[errors > -1][-1]
    self.assertGreater(threshold, err)

    # recover solution from previous and ensure faster convergence.
    if lse_mode:
      init_dual_a, init_dual_b = out.f, out.g
    else:
      init_dual_a, init_dual_b = (geom.scaling_from_potential(out.f),
                                  geom.scaling_from_potential(out.g))
    out_restarted = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode,
        init_dual_a=init_dual_a,
        init_dual_b=init_dual_b,
        inner_iterations=1)
    errors_restarted = out_restarted.errors
    err_restarted = errors_restarted[errors_restarted > -1][-1]
    self.assertGreater(threshold, err_restarted)

    num_iter_restarted = jnp.sum(errors_restarted > -1)
    # check we can only improve on error
    self.assertGreater(err, err_restarted)
    # check first error in restart does at least as well as previous best
    self.assertGreater(err, errors_restarted[0])
    # check only one iteration suffices when restarting with same data.
    self.assertEqual(num_iter_restarted, 1)


if __name__ == '__main__':
  absltest.main()
