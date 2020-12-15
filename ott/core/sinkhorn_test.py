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
import numpy as onp

from ott.core import sinkhorn
from ott.core.ground_geometry import geometry
from ott.core.ground_geometry import grid
from ott.core.ground_geometry import pointcloud


class SinkhornTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 10
    self.n = 100
    self.m = 150
    self.rng, *rngs = jax.random.split(self.rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))
    self.a = a / np.sum(a)
    self.b = b / np.sum(b)

  @parameterized.parameters([True], [False])
  def test_euclidean_point_cloud(self, lse_mode):
    """Two point clouds in dimension 10."""
    threshold = 1e-3
    geom = pointcloud.PointCloudGeometry(self.x, self.y, epsilon=0.1)
    err = sinkhorn.sinkhorn(geom, a=self.a, b=self.b,
                            threshold=threshold, lse_mode=lse_mode).err
    self.assertGreater(threshold, err)

  @parameterized.parameters([True], [False])
  def test_euclidean_point_cloud_parallel_weights(self, lse_mode):
    """Two point clouds, parallel execution on different histograms."""
    self.rng, *rngs = jax.random.split(self.rng, 2)
    batch = 4
    a = jax.random.uniform(rngs[0], (batch, self.n))
    b = jax.random.uniform(rngs[0], (batch, self.m))
    a = a / np.sum(a, axis=1)[:, np.newaxis]
    b = b / np.sum(b, axis=1)[:, np.newaxis]
    threshold = 1e-3
    geom = pointcloud.PointCloudGeometry(
        self.x, self.y, epsilon=0.1, online=True)
    err = sinkhorn.sinkhorn(geom, a=self.a, b=self.b,
                            threshold=threshold, lse_mode=lse_mode).err
    self.assertGreater(np.min(threshold - err), 0)

  @parameterized.parameters([True], [False])
  def test_online_euclidean_point_cloud(self, lse_mode):
    """Testing the on the online geometry."""
    threshold = 1e-3
    geom = pointcloud.PointCloudGeometry(
        self.x, self.y, epsilon=0.1, online=True)
    err = sinkhorn.sinkhorn(geom, a=self.a, b=self.b,
                            threshold=threshold, lse_mode=lse_mode).err
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
        online_geom, a=self.a, b=self.b, threshold=threshold,
        lse_mode=lse_mode)
    out_batch = sinkhorn.sinkhorn(
        batch_geom, a=self.a, b=self.b, threshold=threshold,
        lse_mode=lse_mode)

    # Checks regularized transport costs match.
    self.assertAllClose(out_online.reg_ot_cost, out_batch.reg_ot_cost)
    # check regularized transport matrices match
    self.assertAllClose(
        online_geom.transport_from_potentials(out_online.f, out_online.g),
        batch_geom.transport_from_potentials(out_batch.f, out_batch.g))

  @parameterized.parameters([True], [False])
  def test_separable_grid(self, lse_mode):
    """Two histograms in a grid of size 5 x 6 x 7  in the hypercube^3."""
    grid_size = np.array([5, 6, 7])
    keys = jax.random.split(self.rng, 2)
    a = jax.random.uniform(keys[0], grid_size)
    b = jax.random.uniform(keys[1], grid_size)
    a = a.ravel() / np.sum(a)
    b = b.ravel() / np.sum(b)
    threshold = 1e-3
    geom = grid.Grid(grid_size=grid_size, epsilon=0.1)
    err = sinkhorn.sinkhorn(geom, a=a, b=b,
                            threshold=threshold, lse_mode=lse_mode).err
    self.assertGreater(threshold, err)

  @parameterized.parameters([True], [False])
  def test_grid_vs_euclidean(self, lse_mode):
    grid_size = np.array([4, 3, 8])
    keys = jax.random.split(self.rng, 2)
    a = jax.random.uniform(keys[0], grid_size)
    b = jax.random.uniform(keys[1], grid_size)
    a = a.ravel() / np.sum(a)
    b = b.ravel() / np.sum(b)
    epsilon = 0.1
    geometry_grid = grid.Grid(grid_size=grid_size, epsilon=epsilon)
    x, y, z = onp.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = np.stack([
        np.array(x.ravel()) / np.maximum(1, grid_size[0] - 1),
        np.array(y.ravel()) / np.maximum(1, grid_size[1] - 1),
        np.array(z.ravel()) / np.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geometry_mat = pointcloud.PointCloudGeometry(xyz, xyz, epsilon=epsilon)
    out_mat = sinkhorn.sinkhorn(geometry_mat, a=a, b=b, lse_mode=lse_mode)
    out_grid = sinkhorn.sinkhorn(geometry_grid, a=a, b=b, lse_mode=lse_mode)
    self.assertAllClose(out_mat.reg_ot_cost, out_grid.reg_ot_cost)

  @parameterized.parameters([True], [False])
  def test_autograd_sinkhorn(self, lse_mode):
    d = 3
    n, m = 10, 15
    eps = 1e-3  # perturbation magnitude
    keys = jax.random.split(self.rng, 5)
    x = jax.random.uniform(keys[0], (n, d))
    y = jax.random.uniform(keys[1], (m, d))
    a = jax.random.uniform(keys[2], (n,)) + eps
    b = jax.random.uniform(keys[3], (m,)) + eps
    a = a / np.sum(a)
    b = b / np.sum(b)
    geom = pointcloud.PointCloudGeometry(x, y, epsilon=0.1)

    def reg_ot(a, b):
      return sinkhorn.sinkhorn(geom, a=a, b=b, lse_mode=lse_mode).reg_ot_cost

    reg_ot_and_grad = jax.jit(jax.value_and_grad(reg_ot))
    _, grad_reg_ot = reg_ot_and_grad(a, b)
    delta = jax.random.uniform(keys[4], (n,))
    delta = delta - np.mean(delta)  # center perturbation
    reg_ot_delta_plus = reg_ot(a + eps * delta, b)
    reg_ot_delta_minus = reg_ot(a - eps * delta, b)
    delta_dot_grad = np.sum(delta * grad_reg_ot)
    self.assertAllClose(delta_dot_grad,
                        (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
                        rtol=1e-03, atol=1e-02)

  @parameterized.parameters([True], [False])
  def test_gradient_sinkhorn_geometry(self, lse_mode):
    n = 10
    m = 15
    keys = jax.random.split(self.rng, 2)
    cost_matrix = np.abs(jax.random.normal(keys[0], (n, m)))
    delta = jax.random.normal(keys[1], (n, m))
    delta = delta / np.sqrt(np.vdot(delta, delta))
    eps = 1e-3  # perturbation magnitude

    def loss_fn(cm):
      a = np.ones(cm.shape[0]) / cm.shape[0]
      b = np.ones(cm.shape[1]) / cm.shape[1]
      geom = geometry.Geometry(cm, epsilon=0.5)
      f, g, regularized_transport_cost, _ = sinkhorn.sinkhorn(
          geom, a, b, lse_mode=lse_mode)
      return regularized_transport_cost, (geom, f, g)

    # first calculation of gradient
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    (loss_value, aux), grad_loss = loss_and_grad(cost_matrix)
    custom_grad = np.sum(delta * grad_loss)

    self.assertIsNot(loss_value, np.nan)
    self.assertEqual(grad_loss.shape, cost_matrix.shape)
    self.assertFalse(np.any(np.isnan(grad_loss)))

    # second calculation of gradient
    transport_matrix = aux[0].transport_from_potentials(aux[1], aux[2])
    grad_x = transport_matrix
    other_grad = np.sum(delta * grad_x)

    # third calculation of gradient
    loss_delta_plus, _ = loss_fn(cost_matrix + eps * delta)
    loss_delta_minus, _ = loss_fn(cost_matrix - eps * delta)
    finite_diff_grad = (loss_delta_plus - loss_delta_minus) / (2 * eps)

    self.assertAllClose(custom_grad, other_grad, rtol=1e-02, atol=1e-02)
    self.assertAllClose(custom_grad, finite_diff_grad, rtol=1e-02, atol=1e-02)
    self.assertAllClose(other_grad, finite_diff_grad, rtol=1e-02, atol=1e-02)

  @parameterized.parameters([True], [False])
  def test_gradient_sinkhorn_euclidean(self, lse_mode):
    for lse_mode in [True, False]:
      d = 3
      n = 10
      m = 15
      keys = jax.random.split(self.rng, 2)
      x = jax.random.normal(keys[0], (n, d)) / 10
      y = jax.random.normal(keys[1], (m, d)) / 10

      def loss_fn(x, y):
        a = np.ones(x.shape[0]) / x.shape[0]
        b = np.ones(y.shape[0]) / y.shape[0]
        geom = pointcloud.PointCloudGeometry(x, y, epsilon=0.2)
        f, g, regularized_transport_cost, _ = sinkhorn.sinkhorn(
            geom, a, b, lse_mode=lse_mode)
        return regularized_transport_cost, (geom, f, g)

      delta = jax.random.normal(keys[0], (n, d))
      delta = delta / np.sqrt(np.vdot(delta, delta))
      eps = 1e-5  # perturbation magnitude

      # first calculation of gradient
      loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
      (loss_value, aux), grad_loss = loss_and_grad(x, y)
      custom_grad = np.sum(delta * grad_loss)

      self.assertIsNot(loss_value, np.nan)
      self.assertEqual(grad_loss.shape, x.shape)
      self.assertFalse(np.any(np.isnan(grad_loss)))

      # second calculation of gradient
      tm = aux[0].transport_from_potentials(aux[1], aux[2])
      tmp = 2 * tm[:, :, None] * (x[:, None, :] - y[None, :, :])
      grad_x = np.sum(tmp, 1)
      other_grad = np.sum(delta * grad_x)

      # third calculation of gradient
      loss_delta_plus, _ = loss_fn(x + eps * delta, y)
      loss_delta_minus, _ = loss_fn(x - eps * delta, y)
      finite_diff_grad = (loss_delta_plus - loss_delta_minus) / (2 * eps)

      self.assertAllClose(custom_grad, other_grad, rtol=1e-02, atol=1e-02)
      self.assertAllClose(custom_grad, finite_diff_grad, rtol=1e-02, atol=1e-02)
      self.assertAllClose(other_grad, finite_diff_grad, rtol=1e-02, atol=1e-02)

  def test_apply_transport_geometry(self):
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

    batch_a = 5
    batch_b = 8

    vec_a = jax.random.normal(keys[4], (batch_a, n))
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
            np.dot(transport.T, vec_a.T).T,
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

  @parameterized.parameters([True], [False])
  def test_apply_transport_grid(self, lse_mode):
    grid_size = np.array([3, 5, 4])
    keys = jax.random.split(self.rng, 3)
    a = jax.random.uniform(keys[0], grid_size)
    b = jax.random.uniform(keys[1], grid_size)
    a = a.ravel() / np.sum(a)
    b = b.ravel() / np.sum(b)
    geom_grid = grid.Grid(grid_size=grid_size, epsilon=0.1)
    x, y, z = onp.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = np.stack([
        np.array(x.ravel()) / np.maximum(1, grid_size[0] - 1),
        np.array(y.ravel()) / np.maximum(1, grid_size[1] - 1),
        np.array(z.ravel()) / np.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geom_mat = pointcloud.PointCloudGeometry(xyz, xyz, epsilon=0.1)
    sink_mat = sinkhorn.sinkhorn(geom_mat, a=a, b=b, lse_mode=lse_mode)
    sink_grid = sinkhorn.sinkhorn(geom_grid, a=a, b=b, lse_mode=lse_mode)

    batch_a = 3
    batch_b = 4
    vec_a = jax.random.normal(keys[4], [batch_a, np.prod(grid_size)])
    vec_b = jax.random.normal(keys[4], [batch_b, np.prod(grid_size)])

    vec_a = vec_a / np.sum(vec_a, axis=1)[:, np.newaxis]
    vec_b = vec_b / np.sum(vec_b, axis=1)[:, np.newaxis]

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

if __name__ == '__main__':
  absltest.main()
