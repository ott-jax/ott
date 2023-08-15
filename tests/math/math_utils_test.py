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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.math import utils as mu


class TestNorm:

  @pytest.mark.parametrize("ord", [1.1, 2.0, jnp.inf])
  def test_norm(self, rng: jax.random.PRNGKeyArray, ord):
    d = 13
    f = lambda x: mu.norm(x - x, ord=ord)
    x = jax.random.uniform(rng, (d,))
    np.testing.assert_array_equal(jax.grad(f)(x), 0.0)

    if ord != jnp.inf:
      # Check native jax's norm still returns NaNs when exponent is float or 2.0
      g = lambda x: jnp.linalg.norm(x - x, ord=ord)
      assert jnp.all(jnp.isnan(jax.grad(g)(x)))

    f = functools.partial(mu.norm, ord=ord)
    g = functools.partial(jnp.linalg.norm, ord=ord)

    np.testing.assert_array_equal(jax.grad(f)(x), jax.grad(g)(x))

    np.testing.assert_array_equal(jax.hessian(f)(x), jax.hessian(g)(x))
