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
"""Tests for the Policy."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from ott.core import sinkhorn
from ott.geometry import geometry, pointcloud


class SinkhornTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 3
    self.n = 38
    self.m = 73
    self.rng, *rngs = jax.random.split(self.rng, 10)
    self.rngs = rngs
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,)) + .1
    b = jax.random.uniform(rngs[3], (self.m,)) + .1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @parameterized.parameters([False])
  def test_implicit_differentiation_versus_autodiff(self, lse_mode):
    epsilon = 0.05

    def loss_g(a, x, implicit=True):
      out = sinkhorn.sinkhorn(
          geometry.Geometry(
              cost_matrix=jnp.sum(x ** 2, axis=1)[:, jnp.newaxis] +
              jnp.sum(self.y ** 2, axis=1)[jnp.newaxis, :] -
              2 * jnp.dot(x, self.y.T),
              epsilon=epsilon
          ),
          a=a,
          b=self.b,
          tau_a=0.9,
          tau_b=0.87,
          threshold=1e-6,
          lse_mode=lse_mode,
          implicit_differentiation=implicit
      )
      return out.reg_ot_cost

    def loss_pcg(a, x, implicit=True):
      out = sinkhorn.sinkhorn(
          pointcloud.PointCloud(x, self.y, epsilon=epsilon),
          a=a,
          b=self.b,
          tau_a=1.0,
          tau_b=0.95,
          threshold=1e-6,
          lse_mode=lse_mode,
          implicit_differentiation=implicit
      )
      return out.reg_ot_cost

    for loss in [loss_g, loss_pcg]:
      loss_and_grad_imp = jax.jit(
          jax.value_and_grad(lambda a, x: loss(a, x, True), argnums=(0, 1))
      )
      loss_and_grad_auto = jax.jit(
          jax.value_and_grad(lambda a, x: loss(a, x, False), argnums=(0, 1))
      )

      loss_value_imp, grad_loss_imp = loss_and_grad_imp(self.a, self.x)
      loss_value_auto, grad_loss_auto = loss_and_grad_auto(self.a, self.x)

      np.testing.assert_allclose(loss_value_imp, loss_value_auto)
      eps = 1e-3

      # test gradient w.r.t. a works and gradient implicit ~= gradient autodiff
      delta = jax.random.uniform(self.rngs[4], (self.n,)) / 10
      delta = delta - jnp.mean(delta)  # center perturbation
      reg_ot_delta_plus = loss(self.a + eps * delta, self.x)
      reg_ot_delta_minus = loss(self.a - eps * delta, self.x)
      delta_dot_grad = jnp.sum(delta * grad_loss_imp[0])
      np.testing.assert_allclose(
          delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
          rtol=1e-02,
          atol=1e-02
      )
      # note how we removed gradients below. This is because gradients are only
      # determined up to additive constant here (the primal variable is in the
      # simplex).
      np.testing.assert_allclose(
          grad_loss_imp[0] - jnp.mean(grad_loss_imp[0]),
          grad_loss_auto[0] - jnp.mean(grad_loss_auto[0]),
          rtol=1e-02,
          atol=1e-02
      )

      # test gradient w.r.t. x works and gradient implicit ~= gradient autodiff
      delta = jax.random.uniform(self.rngs[4], (self.n, self.dim))
      reg_ot_delta_plus = loss(self.a, self.x + eps * delta)
      reg_ot_delta_minus = loss(self.a, self.x - eps * delta)
      delta_dot_grad = jnp.sum(delta * grad_loss_imp[1])
      np.testing.assert_allclose(
          delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
          rtol=1e-02,
          atol=1e-02
      )
      np.testing.assert_allclose(
          grad_loss_imp[1], grad_loss_auto[1], rtol=1e-02, atol=1e-02
      )


if __name__ == '__main__':
  absltest.main()
