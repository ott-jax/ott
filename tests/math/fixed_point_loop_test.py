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
import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.math import fixed_point_loop


class TestFixpointIterBackprop:

  @pytest.mark.parametrize("inner_iterations", [1, 2])
  def test_force_scan_grad(self, inner_iterations: int):

    def loop(fixpoint_fn, c: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
      return fixpoint_fn(
          cond_fn=lambda *_: True,
          body_fn=lambda _, c, x, __: jnp.tanh(c * x + 1.0),
          min_iterations=6,
          max_iterations=6,
          inner_iterations=inner_iterations,
          constants=c,
          state=x,
      )

    c, x = jnp.array([0.7, -1.3]), jnp.array([0.2, 0.5])
    expected = jax.grad(
        lambda c, x: jnp.sum(loop(fixed_point_loop.fixpoint_iter, c, x)),
        argnums=(0, 1)
    )(c, x)
    actual = jax.grad(
        lambda c, x: jnp.
        sum(loop(fixed_point_loop.fixpoint_iter_backprop, c, x)),
        argnums=(0, 1)
    )(c, x)

    for a, e in zip(actual, expected):
      np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-6)
