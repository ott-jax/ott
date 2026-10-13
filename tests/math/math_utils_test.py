# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.math import utils as mu


class TestNorm:

  @pytest.mark.fast()
  @pytest.mark.parametrize("kind", ["zero", "rows", "columns", "matrices"])
  @pytest.mark.parametrize("keepdims", [False, True])
  @pytest.mark.parametrize("jit", [False, True])
  def test_zero_slice_jvp(self, kind, keepdims, jit):
    x = jnp.array([[0.0, 0.0], [3.0, 4.0]])
    expected_grad = jnp.array([[0.0, 0.0], [0.6, 0.8]])
    axis = 1
    if kind == "zero":
      x, expected_grad, axis = jnp.zeros(3), jnp.zeros(3), None
    elif kind == "columns":
      x, expected_grad, axis = x.T, expected_grad.T, 0
    elif kind == "matrices":
      x = jnp.stack([jnp.zeros((2, 2)), jnp.array([[3.0, 4.0], [0.0, 0.0]])])
      expected_grad = x / 5.0
      axis = (1, 2)
    fun = lambda x: mu.norm(x, axis=axis, keepdims=keepdims)
    if jit:
      fun = jax.jit(fun)
    value, tangent = jax.jvp(fun, (x,), (jnp.ones_like(x),))
    expected_value = jnp.linalg.norm(x, axis=axis, keepdims=keepdims)
    expected_tangent = expected_grad.sum(axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(value, expected_value, rtol=1e-6)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(lambda x: fun(x).sum())(x), expected_grad, rtol=1e-6
    )

  @pytest.mark.parametrize("ord", [1.1, 2.0, jnp.inf])
  def test_norm(
      self,
      rng: jax.Array,
      ord: float,
  ):
    d = 5
    rng_x, rng_z = jr.split(rng, 2)
    # Test that mu.norm returns properly 0 gradients at 0, when comparing point
    # against itself.
    f = lambda x: mu.norm(x - x, ord=ord)
    x = jr.uniform(rng_x, (d,))
    np.testing.assert_array_equal(jax.grad(f)(x), 0.0)

    # Test same property when using an axis, on jacobians.
    ds = (4, 7, 3)
    z = jr.uniform(rng_z, ds)
    h = lambda x: mu.norm(x - x, ord=ord, axis=1)
    jac = jax.jacobian(h)(z)
    np.testing.assert_array_equal(jac, 0.0)

    # Check native jax's norm still returns NaNs when ord is float.
    if ord != jnp.inf:
      g = lambda x: jnp.linalg.norm(x - x, ord=ord)
      np.testing.assert_array_equal(jnp.isnan(jax.grad(g)(x)), True)
      h = lambda x: jnp.linalg.norm(x - x, ord=ord, axis=1)
      jac = jax.jacobian(h)(z)
      np.testing.assert_array_equal(jnp.isnan(jac), True)

    # Check both coincide outside of 0.
    f = functools.partial(mu.norm, ord=ord)
    g = functools.partial(jnp.linalg.norm, ord=ord)

    np.testing.assert_array_equal(jax.grad(f)(x), jax.grad(g)(x))
    np.testing.assert_array_equal(jax.hessian(f)(x), jax.hessian(g)(x))

    # Check both coincide outside of 0 when using axis.
    f = functools.partial(mu.norm, ord=ord, axis=1)
    g = functools.partial(jnp.linalg.norm, ord=ord, axis=1)

    np.testing.assert_array_equal(jax.jacobian(f)(z), jax.jacobian(g)(z))
