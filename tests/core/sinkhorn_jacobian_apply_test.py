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
"""Tests for the Jacobian of Apply OT."""

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
      tau_a=[1.0, .94],
      tau_b=[1.0, .92],
      shape=[(237, 153)],
      arg=[0, 1],
      axis=[0, 1])
  def test_apply_transport_jacobian(self, lse_mode, tau_a, tau_b, shape, arg,
                                    axis):
    """Tests Jacobian of application of OT to vector, w.r.t.

    a/x.

    Args:
      lse_mode: most tests here are intended to be run in lse_mode=True, kernel
        mode (False) is tested with looser convergence settings.
      tau_a: loosen up 1st marginal constraint when <1.0
      tau_b: loosen up 2nd marginal constraint when <1.0
      shape: size for point clouds n, m.
      arg: test jacobian w.r.t. either weight vectors a or locations x
      axis: test the jacobian of the application of the (right) application of
        transport to arbitrary vec (axis=0) or the left (axis=1).
    """
    n, m = shape
    dim = 4
    rngs = jax.random.split(self.rng, 9)
    x = jax.random.uniform(rngs[0], (n, dim)) / dim
    y = jax.random.uniform(rngs[1], (m, dim)) / dim
    a = jax.random.uniform(rngs[2], (n,)) + .2
    b = jax.random.uniform(rngs[3], (m,)) + .2
    a = a / (0.5 * n) if tau_a < 1.0 else a / jnp.sum(a)
    b = b / (0.5 * m) if tau_b < 1.0 else b / jnp.sum(b)
    vec = jax.random.uniform(rngs[4], (m if axis else n,)) - .5

    delta_a = jax.random.uniform(rngs[5], (n,))
    if tau_a == 1.0:
      delta_a = delta_a - jnp.mean(delta_a)
    delta_x = jax.random.uniform(rngs[6], (n, dim))

    # lse_mode=False is unstable for small epsilon when differentiating as a
    # general rule, even more so when using backprop.
    epsilon = 0.01 if lse_mode else 0.1

    def apply_ot(a, x, implicit):
      out = transport.solve(
          x, y, epsilon=epsilon, a=a, b=b, tau_a=tau_a, tau_b=tau_b,
          lse_mode=lse_mode,
          implicit_differentiation=implicit)
      return out.apply(vec, axis=axis)

    delta = delta_x if arg else delta_a
    # Compute implicit jacobian
    jac_apply_imp = jax.jit(
        jax.jacrev(lambda a, x: apply_ot(a, x, True), argnums=arg))
    j_imp = jac_apply_imp(a, x)
    # Apply jacobian to perturbation tensor (here vector or matrix)
    imp_dif = jnp.sum(
        j_imp * delta[jnp.newaxis, ...],
        axis=tuple(range(1, 1 + len(delta.shape))))
    if lse_mode:  # only check unrolling if using lse_mode, too unstable else.
      # Compute backprop (unrolling) jacobian
      jac_apply_back = jax.jit(
          jax.jacrev(lambda a, x: apply_ot(a, x, False), argnums=arg))
      j_back = jac_apply_back(a, x)
      # Apply jacobian to perturbation tensor (here vector or matrix)
      back_dif = jnp.sum(
          j_back * delta[jnp.newaxis, ...],
          axis=tuple(range(1, 1 + len(delta.shape))))

    # Compute finite difference
    perturb_scale = 1e-5
    a_p = a + perturb_scale * delta_a if arg == 0 else a
    x_p = x if arg == 0 else x + perturb_scale * delta_x
    a_m = a - perturb_scale * delta_a if arg == 0 else a
    x_m = x if arg == 0 else x - perturb_scale * delta_x

    app_p = apply_ot(a_p, x_p, False)
    app_m = apply_ot(a_m, x_m, True)
    fin_dif = (app_p - app_m) / (2 * perturb_scale)

    # Set tolerance depending on lse_mode (False is more loose)
    atol = 1e-2 if lse_mode else 1e-1
    # Check finite differences match with application of (implicit) Jacobian.
    self.assertAllClose(fin_dif, imp_dif, atol=atol, rtol=1e-1)

    # Check unrolling jacobian when using lse_mode.
    if lse_mode:
      self.assertAllClose(fin_dif, back_dif, atol=atol, rtol=1e-1)

      # Check Jacobian matrices match loosely.
      # Orthogonalize j_imp, j_back w.r.t. 1 if balanced problem,
      # and testing jacobian w.r.t weights
      if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
        j_imp = j_imp - jnp.mean(j_imp, axis=1)[:, None]
        j_back = j_back - jnp.mean(j_imp, axis=1)[:, None]
      self.assertAllClose(j_imp, j_back, atol=atol, rtol=1e-1)


if __name__ == '__main__':
  absltest.main()
