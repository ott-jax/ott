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
"""Tests for the Gromov Wasserstein."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
from ott.core import gromov_wasserstein
from ott.geometry import geometry
from ott.geometry import pointcloud


class GromovWassersteinGradTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters([True], [False])
  def test_gradient_marginals_gromov_wasserstein(self, jit):
    """Test gradient w.r.t. probability weights."""
    d_x = 2
    d_y = 3
    n, m = 5, 6
    keys = jax.random.split(self.rng, 5)
    x = jax.random.uniform(keys[0], (n, d_x))
    y = jax.random.uniform(keys[1], (m, d_y))
    a = jax.random.uniform(keys[2], (n,)) + 0.1
    b = jax.random.uniform(keys[3], (m,)) + 0.1
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    geom_x = pointcloud.PointCloud(x, x)
    geom_y = pointcloud.PointCloud(y, y)

    def reg_gw(a, b, implicit):
      sinkhorn_kwargs = {'implicit_differentiation': implicit,
                         'max_iterations': 1001}
      out = gromov_wasserstein.gromov_wasserstein(
          geom_x, geom_y, a=a, b=b, epsilon=1.0,
          loss=gromov_wasserstein.GWSqEuclLoss(),
          max_iterations=10, jit=jit,
          sinkhorn_kwargs=sinkhorn_kwargs)
      return out.reg_gw_cost, (out.f, out.g)

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(reg_gw, has_aux=True,
                                           argnums=(0, 1,))
      (_, aux), grad_reg_gw = reg_gw_and_grad(a, b, implicit)
      grad_matrices[i] = grad_reg_gw
      grad_manual_a = aux[0] - jnp.log(a)
      grad_manual_b = aux[1] - jnp.log(b)
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[0])), True)
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[1])), True)
      self.assertAllClose(grad_manual_a, grad_reg_gw[0], rtol=1e-2, atol=1e-2)
      self.assertAllClose(grad_manual_b, grad_reg_gw[1], rtol=1e-2, atol=1e-2)
    self.assertAllClose(grad_matrices[0][0], grad_matrices[1][0],
                        rtol=1e-02, atol=1e-02)
    self.assertAllClose(grad_matrices[0][1], grad_matrices[1][1],
                        rtol=1e-02, atol=1e-02)

  @parameterized.parameters([True], [False])
  def test_gradient_gromov_wasserstein_pointcloud(self, lse_mode):
    """Test gradient w.r.t. pointclouds."""
    d_x = 2
    d_y = 3
    n, m = 5, 6
    keys = jax.random.split(self.rng, 5)
    x = jax.random.uniform(keys[0], (n, d_x))
    y = jax.random.uniform(keys[1], (m, d_y))
    a = jax.random.uniform(keys[2], (n,)) + 0.1
    b = jax.random.uniform(keys[3], (m,)) + 0.1
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    def reg_gw(x, y, a, b, implicit):
      geom_x = pointcloud.PointCloud(x, x)
      geom_y = pointcloud.PointCloud(y, y)
      sinkhorn_kwargs = {'implicit_differentiation': implicit,
                         'max_iterations': 1001, 'lse_mode': lse_mode}
      return gromov_wasserstein.gromov_wasserstein(
          geom_x, geom_y, a=a, b=b, epsilon=1.0, max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(reg_gw, argnums=(0, 1,))
      _, grad_reg_gw = reg_gw_and_grad(x, y, a, b, implicit)
      grad_matrices[i] = grad_reg_gw
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[0])), True)
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[1])), True)
    self.assertAllClose(grad_matrices[0][0], grad_matrices[1][0],
                        rtol=1e-02, atol=1e-02)
    self.assertAllClose(grad_matrices[0][1], grad_matrices[1][1],
                        rtol=1e-02, atol=1e-02)

  @parameterized.parameters([True], [False])
  def test_gradient_gromov_wasserstein_geometry(self, lse_mode):
    """Test gradient w.r.t. cost matrices."""
    n, m = 5, 6
    keys = jax.random.split(self.rng, 5)
    cx = jax.random.uniform(keys[0], (n, n))
    cy = jax.random.uniform(keys[1], (m, m))
    a = jax.random.uniform(keys[2], (n,)) + 0.1
    b = jax.random.uniform(keys[3], (m,)) + 0.1
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    def reg_gw(cx, cy, a, b, implicit):
      geom_x = geometry.Geometry(cost_matrix=cx)
      geom_y = geometry.Geometry(cost_matrix=cy)
      sinkhorn_kwargs = {'implicit_differentiation': implicit,
                         'max_iterations': 1001, 'lse_mode': lse_mode}
      return gromov_wasserstein.gromov_wasserstein(
          geom_x, geom_y, a=a, b=b, epsilon=1.0, max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(reg_gw, argnums=(0, 1,))
      _, grad_reg_gw = reg_gw_and_grad(cx, cy, a, b, implicit)
      grad_matrices[i] = grad_reg_gw
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[0])), True)
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[1])), True)
    self.assertAllClose(grad_matrices[0][0], grad_matrices[1][0],
                        rtol=1e-02, atol=1e-02)
    self.assertAllClose(grad_matrices[0][1], grad_matrices[1][1],
                        rtol=1e-02, atol=1e-02)

if __name__ == '__main__':
  absltest.main()
