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

"""Jitting test for Sinkhorn."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.test_util
from ott.core import sinkhorn
from ott.geometry import geometry

non_jitted_sinkhorn = functools.partial(sinkhorn.sinkhorn, jit=False)


def assert_output_close(x, y):
  """Asserst SinkhornOutputs are close."""
  x = tuple(a for a in x if (a is not None and isinstance(a, jnp.ndarray)))
  y = tuple(a for a in y if (a is not None and isinstance(a, jnp.ndarray)))
  return chex.assert_tree_all_close(x, y, atol=1e-6, rtol=0)


class SinkhornTest(jax.test_util.JaxTestCase):
  """Check jitted and non jit match for Sinkhorn, and that everything jits."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 3
    self.n = 10
    self.m = 11
    self.rng, *rngs = jax.random.split(self.rng, 10)
    self.rngs = rngs
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,)) + .1
    b = jax.random.uniform(rngs[3], (self.m,)) + .1

    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.epsilon = 0.05
    self.geometry = geometry.Geometry(
        cost_matrix=(jnp.sum(self.x**2, axis=1)[:, jnp.newaxis] +
                     jnp.sum(self.y**2, axis=1)[jnp.newaxis, :] -
                     2 * jnp.dot(self.x, self.y.T)),
        epsilon=self.epsilon)

  def test_jit_vs_non_jit_fwd(self):
    jitted_result = sinkhorn.sinkhorn(self.geometry, self.a, self.b)
    non_jitted_result = non_jitted_sinkhorn(self.geometry, self.a, self.b)

    def f(g, a, b):
      return non_jitted_sinkhorn(g, a, b)

    user_jitted_result = jax.jit(f)(self.geometry, self.a, self.b)
    assert_output_close(jitted_result, non_jitted_result)
    assert_output_close(jitted_result, user_jitted_result)

  @parameterized.parameters([True, False])
  def test_jit_vs_non_jit_bwd(self, implicit):

    def loss(a, x, fun):
      out = fun(
          geometry.Geometry(
              cost_matrix=(jnp.sum(x**2, axis=1)[:, jnp.newaxis] +
                           jnp.sum(self.y**2, axis=1)[jnp.newaxis, :] -
                           2 * jnp.dot(x, self.y.T)),
              epsilon=self.epsilon),
          a=a,
          b=self.b,
          tau_a=0.8,
          tau_b=0.87,
          threshold=1e-4,
          lse_mode=True,
          implicit_differentiation=implicit)
      return out.reg_ot_cost

    def value_and_grad(a, x):
      return jax.value_and_grad(loss)(a, x, non_jitted_sinkhorn)

    jitted_loss, jitted_grad = jax.value_and_grad(loss)(
        self.a, self.x, sinkhorn.sinkhorn)
    non_jitted_loss, non_jitted_grad = jax.value_and_grad(loss)(
        self.a, self.x, non_jitted_sinkhorn)

    user_jitted_loss, user_jitted_grad = jax.jit(value_and_grad)(self.a, self.x)
    chex.assert_tree_all_close(jitted_loss, non_jitted_loss, atol=1e-6, rtol=0)
    chex.assert_tree_all_close(jitted_grad, non_jitted_grad, atol=1e-6, rtol=0)
    chex.assert_tree_all_close(user_jitted_loss, jitted_loss, atol=1e-6, rtol=0)
    chex.assert_tree_all_close(user_jitted_grad, jitted_grad, atol=1e-6, rtol=0)


if __name__ == '__main__':
  absltest.main()
