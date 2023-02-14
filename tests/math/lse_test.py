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
"""Tests for the jvp of a custom implementation of lse."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.math import utils as mu


@pytest.mark.fast
class TestGeometryLse:

  def test_lse(self, rng: jnp.ndarray):
    """Test consistency of custom lse's jvp."""
    n, m = 12, 8
    keys = jax.random.split(rng, 5)
    mat = jax.random.normal(keys[0], (n, m))
    # picking potentially negative weights on purpose
    b_0 = jax.random.normal(keys[1], (m,))
    b_1 = jax.random.normal(keys[2], (n, 1))

    def lse_(x, axis, b, return_sign):
      out = mu.logsumexp(x, axis, False, b, return_sign)
      return jnp.sum(out[0] if return_sign else out)

    lse = jax.value_and_grad(lse_, argnums=(0, 2))
    for axis in (0, 1):
      _, g = lse(mat, axis, None, False)
      delta_mat = jax.random.normal(keys[3], (n, m))
      eps = 1e-3
      val_peps = lse(mat + eps * delta_mat, axis, None, False)[0]
      val_meps = lse(mat - eps * delta_mat, axis, None, False)[0]
      np.testing.assert_allclose((val_peps - val_meps) / (2 * eps),
                                 jnp.sum(delta_mat * g[0]),
                                 rtol=1e-03,
                                 atol=1e-02)
    for b, dim, axis in zip((b_0, b_1), (m, n), (1, 0)):
      delta_b = jax.random.normal(keys[4], (dim,)).reshape(b.shape)
      _, g = lse(mat, axis, b, True)
      eps = 1e-3
      val_peps = lse(mat + eps * delta_mat, axis, b + eps * delta_b, True)[0]
      val_meps = lse(mat - eps * delta_mat, axis, b - eps * delta_b, True)[0]
      np.testing.assert_allclose((val_peps - val_meps) / (2 * eps),
                                 jnp.sum(delta_mat * g[0]) +
                                 jnp.sum(delta_b * g[1]),
                                 rtol=1e-03,
                                 atol=1e-02)
