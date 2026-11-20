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
from typing import Callable

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.math import fixed_point_loop


class TestFixpointIterBackprop:

  def test_force_scan_grad(self):

    def loss(fixpoint_fn, c: jnp.ndarray, x: jnp.ndarray) -> float:
      x = fixpoint_fn(
          cond_fn=lambda *_: True,
          body_fn=lambda _, c, x, __: jnp.tanh(c * x + 1.0),
          min_iterations=6,
          max_iterations=6,
          inner_iterations=2,
          constants=c,
          state=x,
      )
      return jnp.sum(x)

    grad_fn = jax.grad(loss, argnums=(1, 2))
    c, x = jnp.array([0.7, -1.3]), jnp.array([0.2, 0.5])

    expected = grad_fn(fixed_point_loop.fixpoint_iter, c, x)
    actual = grad_fn(fixed_point_loop.fixpoint_iter_backprop, c, x)

    for a, e in zip(actual, expected):
      np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-6)

  @pytest.mark.parametrize(
      "fixpoint_fn",
      [fixed_point_loop.fixpoint_iter, fixed_point_loop.fixpoint_iter_backprop]
  )
  def test_force_scan_partial_block(self, fixpoint_fn: Callable):

    def loss(c: jnp.ndarray, x: jnp.ndarray) -> float:
      x = fixpoint_fn(
          cond_fn=lambda *_: True,
          body_fn=lambda _, c, x, __: c * x,
          min_iterations=15,
          max_iterations=15,
          inner_iterations=10,
          constants=c,
          state=x,
      )
      return jnp.sum(x)

    c, x = jnp.array([0.9, 1.1]), jnp.array([0.2, 0.5])
    value, (grad_c, grad_x) = jax.value_and_grad(loss, argnums=(0, 1))(c, x)

    # as in the `while_loop`, the last block of 10 iterations is completed
    np.testing.assert_allclose(value, jnp.sum(c ** 20 * x), rtol=1e-5)
    np.testing.assert_allclose(grad_c, 20 * c ** 19 * x, rtol=1e-5)
    np.testing.assert_allclose(grad_x, c ** 20, rtol=1e-5)
