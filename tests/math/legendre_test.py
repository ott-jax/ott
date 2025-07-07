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

from ott import math


@pytest.mark.fast()
class TestLegendre:

  def test_legendre(self, rng: jax.Array):
    """Test Legendre by evaluating it on a quadratic function."""
    d = 5
    rngs = jax.random.split(rng, 5)
    mat = jax.random.normal(rngs[0], (d, d))
    mat = mat @ mat.T

    x = jax.random.normal(rngs[1], (d,))

    def fun(x):
      return .5 * jnp.dot(x, mat @ x)

    fun_star = math.legendre(fun)

    np.testing.assert_allclose(
        jnp.linalg.solve(mat, x), jax.grad(fun_star)(x), rtol=1e-5, atol=1e-5
    )
