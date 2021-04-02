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
from ott.geometry import geometry


class SinkhornJacobianTest(jax.test_util.JaxTestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Bal-Lse',
          lse_mode=True,
          tau_a=1.0,
          tau_b=1.0),
      dict(
          testcase_name='Bal-Scal',
          lse_mode=False,
          tau_a=1.0,
          tau_b=1.0),
      dict(
          testcase_name='Unbal-Lse',
          lse_mode=True,
          tau_a=.8,
          tau_b=.9),
      dict(
          testcase_name='Unbal-Scal',
          lse_mode=False,
          tau_a=.88,
          tau_b=.99),
      dict(
          testcase_name='SemiBal-Lse',
          lse_mode=True,
          tau_a=1.0,
          tau_b=.95),
      dict(
          testcase_name='SemiBal-Scal',
          lse_mode=False,
          tau_a=.7,
          tau_b=1.0))
  def test_implicit_jacobian_versus_autodiff(self, lse_mode, tau_a, tau_b):
    epsilon = 0.1
    dim = 4
    m = 38
    for n in (15, 73):  # try both situations where n<m and n>m
      rngs = jax.random.split(jax.random.PRNGKey(0), 10)
      x = jax.random.uniform(rngs[0], (n, dim))
      y = jax.random.uniform(rngs[1], (m, dim))
      a = jax.random.uniform(rngs[2], (n,)) + .1
      b = jax.random.uniform(rngs[3], (m,)) + .1
      a = a / jnp.sum(a)
      b = b / jnp.sum(b)
      # h, a random vector, will be summed against potential to create
      # an objective function that is not directly reg_ot_cost, to test
      # jacobian application.
      h = 5 * jax.random.uniform(rngs[5], (n,)) -2.5
      # since potentials are defined up to an additive constant, it makes
      # sense to use a function that is invariant w.r.t that.
      h = h - jnp.mean(h)

      def potential(a, x, implicit, d):
        out = sinkhorn.sinkhorn(
            geometry.Geometry(
                cost_matrix=jnp.sum(x ** 2, axis=1)[:, jnp.newaxis] +
                jnp.sum(y ** 2, axis=1)[jnp.newaxis, :] -
                2 * jnp.dot(x, y.T),
                epsilon=epsilon),
            a=a,
            b=b,
            tau_a=tau_a,
            tau_b=tau_b,
            lse_mode=lse_mode,
            threshold=1e-4,
            implicit_differentiation=implicit,
            inner_iterations=2,
            jit=False)
        return jnp.sum(out.f * d)

      loss_and_grad_imp = jax.value_and_grad(
          lambda a, x: potential(a, x, True, h), argnums=(0, 1))
      loss_and_grad_auto = jax.value_and_grad(
          lambda a, x: potential(a, x, False, h), argnums=(0, 1))

      loss_value_imp, grad_loss_imp = loss_and_grad_imp(a, x)
      loss_value_auto, grad_loss_auto = loss_and_grad_auto(a, x)

      self.assertAllClose(loss_value_imp, loss_value_auto)
      eps = 1e-3

      # test gradient w.r.t. a works and gradient implicit ~= gradient autodiff
      delta = jax.random.uniform(rngs[4], (n,)) / 10
      delta = delta - jnp.mean(delta)  # center perturbation
      reg_ot_delta_plus = potential(a + eps * delta, x, True, h)
      reg_ot_delta_minus = potential(a - eps * delta, x, True, h)
      delta_dot_grad = jnp.sum(delta * grad_loss_imp[0])

      self.assertAllClose(
          delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
          rtol=1e-02,
          atol=1e-02)
      # note how we removed gradients below. This is because gradients are only
      # determined up to additive constant here (the primal variable is in the
      # simplex).
      self.assertAllClose(
          grad_loss_imp[0] - jnp.mean(grad_loss_imp[0]),
          grad_loss_auto[0] - jnp.mean(grad_loss_auto[0]),
          rtol=5e-02)

      # test gradient w.r.t. x works and gradient implicit ~= gradient autodiff
      delta = jax.random.uniform(rngs[4], (n, dim))
      reg_ot_delta_plus = potential(a, x + eps * delta, True, h)
      reg_ot_delta_minus = potential(a, x - eps * delta, True, h)
      delta_dot_grad = jnp.sum(delta * grad_loss_imp[1])
      self.assertAllClose(
          delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
          rtol=1e-02,
          atol=1e-02)
      self.assertAllClose(
          grad_loss_imp[1], grad_loss_auto[1], rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
  absltest.main()
