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

import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from ott.core import implicit_differentiation as implicit_lib
from ott.core import problems, sinkhorn
from ott.geometry import pointcloud


class SinkhornHessianTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.product(
      lse_mode=[True, False],
      tau_a=[1.0, .93],
      tau_b=[1.0, .91],
      shape=[(12, 15)],
      arg=[0, 1]
  )
  def test_hessian_sinkhorn(self, lse_mode, tau_a, tau_b, shape, arg):
    """Test hessian w.r.t. weights and locations."""
    n, m = shape

    dim = 3
    rngs = jax.random.split(self.rng, 6)
    x = jax.random.uniform(rngs[0], (n, dim))
    y = jax.random.uniform(rngs[1], (m, dim))
    a = jax.random.uniform(rngs[2], (n,)) + .1
    b = jax.random.uniform(rngs[3], (m,)) + .1
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    epsilon = 0.1
    ridge = 1e-5

    def loss(a, x, implicit=True):
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
      prob = problems.LinearProblem(geom, a, b, tau_a, tau_b)
      implicit_diff = (
          None if not implicit else
          implicit_lib.ImplicitDiff(ridge_kernel=ridge, ridge_identity=ridge)
      )
      solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode,
          threshold=1e-4,
          use_danskin=False,
          implicit_diff=implicit_diff
      )
      return solver(prob).reg_ot_cost

    delta_a = jax.random.uniform(rngs[4], (n,))
    delta_a = delta_a - jnp.mean(delta_a)
    delta_x = jax.random.uniform(rngs[5], (n, dim))

    # Test that Hessians produced with either backprop or implicit do match.
    hess_loss_imp = jax.jit(
        jax.hessian(lambda a, x: loss(a, x, True), argnums=arg)
    )
    hess_imp = hess_loss_imp(a, x)
    hess_loss_back = jax.jit(
        jax.hessian(lambda a, x: loss(a, x, False), argnums=arg)
    )
    hess_back = hess_loss_back(a, x)
    # In the balanced case, when studying differentiability w.r.t
    # weights, both Hessians must be the same,
    # but only need to be so on the orthogonal space to 1s.
    # For that reason we remove that contribution and check the
    # resulting matrices are equal.
    if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
      hess_imp -= jnp.mean(hess_imp, axis=1)[:, None]
      hess_back -= jnp.mean(hess_back, axis=1)[:, None]

    # Uniform equality is difficult to obtain numerically on the
    # entire matrices. We switch to relative 1-norm of difference.
    dif_norm = jnp.sum(jnp.abs(hess_imp - hess_back))
    rel_dif_norm = dif_norm / jnp.sum(jnp.abs(hess_imp))
    self.assertGreater(0.1, rel_dif_norm)

    eps = 1e-3
    for impl in [True, False]:
      grad_ = jax.jit(
          jax.grad(functools.partial(loss, implicit=impl), argnums=arg)
      )
      grad_init = grad_(a, x)

      # Depending on variable tested, perturb either a or x.
      a_p = a + eps * delta_a if arg == 0 else a
      x_p = x if arg == 0 else x + eps * delta_x

      # Perturbed gradient.
      grad_pert = grad_(a_p, x_p)
      grad_dif = (grad_pert - grad_init) / eps
      # Apply hessian to perturbation
      if arg == 0:
        hess_delta = jnp.matmul(hess_imp, delta_a)
      else:
        # Here tensordot is needed because Hessian is 4D, delta_x is 2D.
        hess_delta = jnp.tensordot(hess_imp, delta_x)

      if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
        hess_delta -= jnp.mean(hess_delta)
        grad_dif -= jnp.mean(grad_dif)

      # No rtol here because many of these values can be close to 0.
      np.testing.assert_allclose(grad_dif, hess_delta, atol=0.1, rtol=0)


if __name__ == '__main__':
  absltest.main()
