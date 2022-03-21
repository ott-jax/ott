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
"""Tests for the cost/norm functions."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs


class CostFnTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def test_cosine(self):
    """Test the cosine cost function."""

    x = jnp.array([0, 0], dtype=jnp.float32)
    y = jnp.array([0, 0], dtype=jnp.float32)
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 0.0, rtol=1E-5, atol=1E-5)

    x = jnp.array([1.0, 0], dtype=jnp.float32)
    y = jnp.array([1.0, 0], dtype=jnp.float32)
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 1.0, rtol=1E-5, atol=1E-5)

    x = jnp.array([1.0, 0], dtype=jnp.float32)
    y = jnp.array([-1.0, 0], dtype=jnp.float32)
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - -1.0, rtol=1E-5, atol=1E-5)

    n, m, d = 10, 12, 7
    keys = jax.random.split(self.rng, 2)
    x = jax.random.normal(keys[0], (n, d))
    y = jax.random.normal(keys[1], (m, d))

    cosine_fn = costs.Cosine()
    normalize = lambda v: v / jnp.sqrt(jnp.sum(v**2))
    for i in range(n):
      for j in range(m):
        exp_sim_xi_yj = jnp.sum(normalize(x[i]) * normalize(y[j]))
        exp_dist_xi_yj = 1.0 - exp_sim_xi_yj
        np.testing.assert_allclose(
            cosine_fn.pairwise(x[i], y[j]),
            exp_dist_xi_yj,
            rtol=1E-5,
            atol=1E-5)

    all_pairs = cosine_fn.all_pairs(x, y)
    for i in range(n):
      for j in range(m):
        self.assertEqual(cosine_fn.pairwise(x[i], y[j]), all_pairs[i, j])


if __name__ == '__main__':
  absltest.main()
