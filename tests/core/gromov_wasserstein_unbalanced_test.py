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
"""Tests for the Gromov Wasserstein."""

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
from absl.testing import absltest, parameterized

from ott.core import gromov_wasserstein
from ott.geometry import geometry, pointcloud


class GromovWassersteinUnbalancedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    d_x = 2
    d_y = 3
    self.n, self.m = 5, 6
    keys = jax.random.split(self.rng, 7)
    self.x = jax.random.uniform(keys[0], (self.n, d_x))
    self.y = jax.random.uniform(keys[1], (self.m, d_y))
    a = jax.random.uniform(keys[2], (self.n,)) + 0.1
    b = jax.random.uniform(keys[3], (self.m,)) + 0.1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(keys[4], (self.n, self.n))
    self.cy = jax.random.uniform(keys[5], (self.m, self.m))
    self.tau_a = 0.8
    self.tau_b = 0.9

  def test_gromov_wasserstein_pointcloud(self):
    """Test basic computations pointclouds."""

    def reg_gw(x, y, a, b):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      return gromov_wasserstein.gromov_wasserstein(
          geom_x,
          geom_y,
          a=a,
          b=b,
          tau_a=self.tau_a,
          tau_b=self.tau_b,
          epsilon=1.0,
          max_iterations=10
      ).reg_gw_cost

    self.assertIsNot(jnp.isnan(reg_gw(self.x, self.y, self.a, self.b)), True)

  @parameterized.parameters([True], [False])
  def test_gradient_gromov_wasserstein_pointcloud(
      self, gw_unbalanced_correction
  ):
    """Test gradient w.r.t. pointclouds."""

    def reg_gw(x, y, a, b, implicit):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001
      }
      return gromov_wasserstein.gromov_wasserstein(
          geom_x,
          geom_y,
          a=a,
          b=b,
          tau_a=self.tau_a,
          tau_b=self.tau_b,
          gw_unbalanced_correction=gw_unbalanced_correction,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      ).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(
          reg_gw, argnums=(
              0,
              1,
          )
      )
      _, grad_reg_gw = reg_gw_and_grad(self.x, self.y, self.a, self.b, implicit)
      grad_matrices[i] = grad_reg_gw
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[0])), True)
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[1])), True)
    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-02, atol=1e-02
    )

  @parameterized.parameters([True], [False])
  def test_gradient_gromov_wasserstein_geometry(self, gw_unbalanced_correction):
    """Test gradient w.r.t. cost matrices."""

    def reg_gw(cx, cy, a, b, implicit):
      geom_x = geometry.Geometry(cost_matrix=cx)
      geom_y = geometry.Geometry(cost_matrix=cy)
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001
      }
      return gromov_wasserstein.gromov_wasserstein(
          geom_x,
          geom_y,
          a=a,
          b=b,
          tau_a=self.tau_a,
          tau_b=self.tau_b,
          gw_unbalanced_correction=gw_unbalanced_correction,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      ).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(
          reg_gw, argnums=(
              0,
              1,
          )
      )
      _, grad_reg_gw = reg_gw_and_grad(
          self.cx, self.cy, self.a, self.b, implicit
      )
      grad_matrices[i] = grad_reg_gw
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[0])), True)
      self.assertIsNot(jnp.any(jnp.isnan(grad_reg_gw[1])), True)
    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-02, atol=1e-02
    )


if __name__ == '__main__':
  absltest.main()
