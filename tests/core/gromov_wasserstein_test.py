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
  def test_autograd_gromov_wasserstein(self, jit):
    """Test gradient w.r.t. probability weights."""
    d_x = 2
    d_y = 3
    n, m = 5, 6
    eps = 1e-3  # perturbation magnitude
    keys = jax.random.split(self.rng, 5)
    x = jax.random.uniform(keys[0], (n, d_x))
    y = jax.random.uniform(keys[1], (m, d_y))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    geom_x = pointcloud.PointCloud(x, x, epsilon=0.1)
    geom_y = pointcloud.PointCloud(y, y, epsilon=0.1)

    def reg_gw(a, b):
      sinkhorn_kwargs = {'implicit_differentiation': False,
                         'max_iterations': 101}
      return gromov_wasserstein.gromov_wasserstein(
          geom_x, geom_y, a=a, b=b, loss=gromov_wasserstein.GWSqEuclLoss(),
          max_iterations_gromov=5, jit_gromov=jit,
          sinkhorn_kwargs=sinkhorn_kwargs).reg_gw_cost

    reg_gw_and_grad = jax.value_and_grad(reg_gw)
    _, grad_reg_gw = reg_gw_and_grad(a, b)
    delta = jax.random.uniform(keys[4], (n,))
    reg_gw_delta_plus = reg_gw(a + eps * delta, b)
    reg_gw_delta_minus = reg_gw(a - eps * delta, b)
    delta_dot_grad = jnp.nansum(delta * grad_reg_gw)
    self.assertIsNot(jnp.any(jnp.isnan(delta_dot_grad)), True)
    self.assertAllClose(delta_dot_grad,
                        (reg_gw_delta_plus - reg_gw_delta_minus) / (2 * eps),
                        rtol=1e-02, atol=1e-02)

  @parameterized.parameters([True], [False])
  def test_gradient_gromov_wasserstein_pointcloud(self, lse_mode):
    """Test gradient w.r.t. pointclouds."""
    d_x = 2
    d_y = 3
    n, m = 5, 6
    eps = 1e-3  # perturbation magnitude
    keys = jax.random.split(self.rng, 5)
    x = jax.random.uniform(keys[0], (n, d_x))
    y = jax.random.uniform(keys[1], (m, d_y))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    def reg_gw(x, y, a, b):
      geom_x = pointcloud.PointCloud(x, x, epsilon=0.1)
      geom_y = pointcloud.PointCloud(y, y, epsilon=0.1)
      sinkhorn_kwargs = {'implicit_differentiation': False,
                         'max_iterations': 101}
      return gromov_wasserstein.gromov_wasserstein(
          geom_x, geom_y, a=a, b=b, max_iterations_gromov=5,
          sinkhorn_kwargs=sinkhorn_kwargs).reg_gw_cost

    reg_gw_and_grad = jax.value_and_grad(reg_gw, argnums=(0, 1,))
    _, grad_reg_gw = reg_gw_and_grad(x, y, a, b)
    delta_x = jax.random.uniform(keys[4], (n, d_x))
    delta_y = jax.random.uniform(keys[4], (m, d_y))
    reg_gw_delta_plus_x = reg_gw(x + eps * delta_x, y, a, b)
    reg_gw_delta_minus_x = reg_gw(x - eps * delta_x, y, a, b)
    reg_gw_delta_plus_y = reg_gw(x, y + eps * delta_y, a, b)
    reg_gw_delta_minus_y = reg_gw(x, y - eps * delta_y, a, b)
    delta_dot_grad_x = jnp.sum(delta_x * grad_reg_gw[0])
    delta_dot_grad_y = jnp.sum(delta_y * grad_reg_gw[1])
    self.assertIsNot(jnp.any(jnp.isnan(delta_dot_grad_x)), True)
    self.assertIsNot(jnp.any(jnp.isnan(delta_dot_grad_y)), True)
    self.assertAllClose(delta_dot_grad_x,
                        (reg_gw_delta_plus_x
                         - reg_gw_delta_minus_x) / (2 * eps),
                        rtol=1e-02, atol=1e-02)
    self.assertAllClose(delta_dot_grad_y,
                        (reg_gw_delta_plus_y
                         - reg_gw_delta_minus_y) / (2 * eps),
                        rtol=1e-02, atol=1e-02)

  @parameterized.parameters([True], [False])
  def test_gradient_gromov_wasserstein_geometry(self, lse_mode):
    """Test gradient w.r.t. cost matrices."""
    n, m = 5, 6
    eps = 1e-3  # perturbation magnitude
    keys = jax.random.split(self.rng, 5)
    cx = jax.random.uniform(keys[0], (n, n))
    cy = jax.random.uniform(keys[1], (m, m))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    def reg_gw(cx, cy, a, b):
      geom_x = geometry.Geometry(cost_matrix=cx, epsilon=0.1)
      geom_y = geometry.Geometry(cost_matrix=cy, epsilon=0.1)
      sinkhorn_kwargs = {'implicit_differentiation': False,
                         'max_iterations': 101}
      return gromov_wasserstein.gromov_wasserstein(
          geom_x, geom_y, a=a, b=b, max_iterations_gromov=5,
          sinkhorn_kwargs=sinkhorn_kwargs).reg_gw_cost

    reg_gw_and_grad = jax.value_and_grad(reg_gw, argnums=(0, 1,))
    _, grad_reg_gw = reg_gw_and_grad(cx, cy, a, b)
    delta_x = jax.random.uniform(keys[4], (n, n))
    delta_y = jax.random.uniform(keys[4], (m, m))
    reg_gw_delta_plus_x = reg_gw(cx + eps * delta_x, cy, a, b)
    reg_gw_delta_minus_x = reg_gw(cx - eps * delta_x, cy, a, b)
    reg_gw_delta_plus_y = reg_gw(cx, cy + eps * delta_y, a, b)
    reg_gw_delta_minus_y = reg_gw(cx, cy - eps * delta_y, a, b)
    delta_dot_grad_x = jnp.sum(delta_x * grad_reg_gw[0])
    delta_dot_grad_y = jnp.sum(delta_y * grad_reg_gw[1])
    self.assertIsNot(jnp.any(jnp.isnan(delta_dot_grad_x)), True)
    self.assertIsNot(jnp.any(jnp.isnan(delta_dot_grad_y)), True)
    self.assertAllClose(delta_dot_grad_x,
                        (reg_gw_delta_plus_x
                         - reg_gw_delta_minus_x) / (2 * eps),
                        rtol=1e-02, atol=1e-02)
    self.assertAllClose(delta_dot_grad_y,
                        (reg_gw_delta_plus_y
                         - reg_gw_delta_minus_y) / (2 * eps),
                        rtol=1e-02, atol=1e-02)

if __name__ == '__main__':
  absltest.main()
