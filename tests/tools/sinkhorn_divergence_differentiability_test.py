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
"""Tests for the Sinkhorn divergence."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.test_util
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence


class SinkhornDivergenceGradTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self._dim = 3
    self._num_points = 13, 12
    self.rng, *rngs = jax.random.split(self.rng, 3)
    a = jax.random.uniform(rngs[0], (self._num_points[0],))
    b = jax.random.uniform(rngs[1], (self._num_points[1],))
    self._a = a / jnp.sum(a)
    self._b = b / jnp.sum(b)

  def test_gradient_generic_point_cloud_wrapper(self):
    rngs = jax.random.split(self.rng, 3)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))

    def loss_fn(cloud_a, cloud_b):
      div = sinkhorn_divergence.sinkhorn_divergence(
          pointcloud.PointCloud,
          cloud_a, cloud_b, epsilon=1.0,
          a=self._a, b=self._b,
          sinkhorn_kwargs=dict(threshold=0.05))
      return div.divergence

    delta = jax.random.normal(rngs[2], x.shape)
    eps = 1e-3  # perturbation magnitude

    # first calculation of gradient
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    loss_value, grad_loss = loss_and_grad(x, y)
    custom_grad = jnp.sum(delta * grad_loss)

    self.assertIsNot(loss_value, jnp.nan)
    self.assertEqual(grad_loss.shape, x.shape)
    self.assertFalse(jnp.any(jnp.isnan(grad_loss)))

    # second calculation of gradient
    loss_delta_plus = loss_fn(x + eps * delta, y)
    loss_delta_minus = loss_fn(x - eps * delta, y)
    finite_diff_grad = (loss_delta_plus - loss_delta_minus) / (2 * eps)

    self.assertAllClose(custom_grad, finite_diff_grad, rtol=1e-02, atol=1e-02)

if __name__ == '__main__':
  absltest.main()
