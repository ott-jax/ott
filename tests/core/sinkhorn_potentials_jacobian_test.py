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
"""Tests for the Jacobian of optimal potential."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
from ott.tools import transport


class SinkhornJacobianTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.product(
      lse_mode=[True, False],
      tau_a=[1.0, .93],
      tau_b=[1.0, .91],
      shape=[(12, 15), (27, 18)],
      arg=[0, 1])
  def test_potential_jacobian_sinkhorn(self, lse_mode, tau_a, tau_b, shape,
                                       arg):
    """Test Jacobian of optimal potential w.r.t. weights and locations."""
    n, m = shape
    dim = 3
    rngs = jax.random.split(self.rng, 7)
    x = jax.random.uniform(rngs[0], (n, dim))
    y = jax.random.uniform(rngs[1], (m, dim))
    a = jax.random.uniform(rngs[2], (n,)) + .2
    b = jax.random.uniform(rngs[3], (m,)) + .2
    a = a / (0.5 * n) if tau_a < 1.0 else a / jnp.sum(a)
    b = b / (0.5 * m) if tau_b < 1.0 else b / jnp.sum(b)
    random_dir = jax.random.uniform(rngs[4], (n,)) / n
    # center projection direction so that < potential , random_dir>
    # is invariant w.r.t additive shifts.
    random_dir = random_dir - jnp.mean(random_dir)
    delta_a = jax.random.uniform(rngs[5], (n,))
    if tau_a == 1.0:
      delta_a = delta_a - jnp.mean(delta_a)
    delta_x = jax.random.uniform(rngs[6], (n, dim))

    # As expected, lse_mode False has a harder time with small epsilon when
    # differentiating.
    epsilon = 0.01 if lse_mode else 0.1

    def loss_from_potential(a, x, implicit):
      out = transport.solve(
          x, y, epsilon=epsilon, a=a, b=b, tau_a=tau_a, tau_b=tau_b,
          lse_mode=lse_mode,
          implicit_differentiation=implicit)
      return jnp.sum(random_dir * out.solver_output.f)

    # Compute implicit gradient
    loss_imp = jax.jit(
        jax.value_and_grad(
            lambda a, x: loss_from_potential(a, x, True), argnums=arg))
    _, g_imp = loss_imp(a, x)
    imp_dif = jnp.sum(g_imp * (delta_a if arg == 0 else delta_x))
    # Compute backprop (unrolling) gradient

    loss_back = jax.jit(
        jax.grad(lambda a, x: loss_from_potential(a, x, False), argnums=arg))
    g_back = loss_back(a, x)
    back_dif = jnp.sum(g_back * (delta_a if arg == 0 else delta_x))

    # Compute finite difference
    perturb_scale = 1e-4
    a_p = a + perturb_scale * delta_a if arg == 0 else a
    x_p = x if arg == 0 else x + perturb_scale * delta_x
    a_m = a - perturb_scale * delta_a if arg == 0 else a
    x_m = x if arg == 0 else x - perturb_scale * delta_x

    val_p, _ = loss_imp(a_p, x_p)
    val_m, _ = loss_imp(a_m, x_m)
    fin_dif = (val_p - val_m) / (2 * perturb_scale)

    self.assertAllClose(fin_dif, back_dif, atol=1e-2, rtol=1e-2)
    self.assertAllClose(fin_dif, imp_dif, atol=1e-2, rtol=1e-2)

    # center g_imp, g_back if balanced problem testing gradient w.r.t weights
    if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
      g_imp = g_imp - jnp.mean(g_imp)
      g_back = g_back - jnp.mean(g_back)
    self.assertAllClose(g_imp, g_back, atol=5e-2, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
