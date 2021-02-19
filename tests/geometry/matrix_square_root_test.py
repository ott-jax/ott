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

# Lint as: python3
"""Tests for matrix square roots."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.test_util
from ott.geometry import matrix_square_root


class MatrixSquareRootTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 23
    self.batch = 3

  def test_matrix_sqrtm(self):
    """Sample a random p.s.d. (Wishart) matrix, check its sqrt matches."""

    matrices = jax.random.normal(self.rng, (self.batch, self.dim, 2 * self.dim))

    for x in (matrices, matrices[0, :, :]):  # try with many and only one.
      x = jnp.matmul(x, jnp.swapaxes(x, -1, -2))
      threshold = 1e-4

      sqrt_x, inv_sqrt_x, errors = matrix_square_root.sqrtm(x, self.dim,
                                                            threshold)
      err = errors[errors > -1][-1]
      self.assertGreater(threshold, err)
      self.assertAllClose(x, jnp.matmul(sqrt_x, sqrt_x), rtol=1e-3, atol=1e-3)
      ids = jnp.eye(self.dim)
      if jnp.ndim(x) == 3:
        ids = ids[jnp.newaxis, :, :]
      self.assertAllClose(
          jnp.zeros_like(x),
          jnp.matmul(x, jnp.matmul(inv_sqrt_x, inv_sqrt_x)) - ids,
          atol=1e-2)


if __name__ == '__main__':
  absltest.main()
