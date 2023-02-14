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
"""Tests for matrix square roots."""
from typing import Any, Callable

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.math import matrix_square_root


def _get_random_spd_matrix(dim: int, key: jnp.ndarray):
  # Get a random symmetric, positive definite matrix of a specified size.

  key, subkey0, subkey1 = jax.random.split(key, num=3)
  # Step 1: generate a random orthogonal matrix
  m = jax.random.normal(key=subkey0, shape=[dim, dim])
  q, _ = jnp.linalg.qr(m)

  # Step 2: generate random eigenvalues in [1/2. , 2.] to ensure the condition
  # number is reasonable.
  eigs = 2. ** (2. * jax.random.uniform(key=subkey1, shape=(dim,)) - 1.)

  return jnp.matmul(eigs[None, :] * q, jnp.transpose(q))


def _get_test_fn(
    fn: Callable[[jnp.ndarray], jnp.ndarray], dim: int, key: jnp.ndarray,
    **kwargs: Any
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  # We want to test gradients of a function fn that maps positive definite
  # matrices to positive definite matrices by comparing them to finite
  # difference approximations. We'll do so via a test function that
  # (1) takes an arbitrary real as an input,
  # (2) maps the real to a positive definite matrix,
  # (3) applies fn, then
  # (4) maps the matrix-valued output of fn to a scalar.
  key, subkey0, subkey1, subkey2, subkey3 = jax.random.split(key, num=5)
  m0 = _get_random_spd_matrix(dim=dim, key=subkey0)
  m1 = _get_random_spd_matrix(dim=dim, key=subkey1)
  dx = _get_random_spd_matrix(dim=dim, key=subkey2)
  unit = jax.random.normal(key=subkey3, shape=(dim, dim))
  unit /= jnp.sqrt(jnp.sum(unit ** 2.))

  def _test_fn(x: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
    # m is the product of 2 symmetric, positive definite matrices
    # so it will be positive definite but not necessarily symmetric
    m = jnp.matmul(m0, m1 + x * dx)
    return jnp.sum(fn(m, **kwargs) * unit)

  return _test_fn


def _sqrt_plus_inv_sqrt(x: jnp.ndarray) -> jnp.ndarray:
  sqrtm = matrix_square_root.sqrtm(x)
  return sqrtm[0] + sqrtm[1]


class TestMatrixSquareRoot:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    self.dim = 13
    self.batch = 3
    # Values for testing the Sylvester solver
    # Sylvester equations have the form AX - XB = C
    # Shapes: A = (m, m), B = (n, n), C = (m, n), X = (m, n)
    m = 3
    n = 2
    key, subkey0, subkey1, subkey2 = jax.random.split(rng, 4)
    self.a = jax.random.normal(key=subkey0, shape=(2, m, m))
    self.b = jax.random.normal(key=subkey1, shape=(2, n, n))
    self.x = jax.random.normal(key=subkey2, shape=(2, m, n))
    # make sure the system has a solution
    self.c = jnp.matmul(self.a, self.x) - jnp.matmul(self.x, self.b)

    self.rng = key

  def test_sqrtm(self):
    """Sample a random p.s.d. (Wishart) matrix, check its sqrt matches."""

    matrices = jax.random.normal(self.rng, (self.batch, self.dim, 2 * self.dim))

    for x in (matrices, matrices[0, :, :]):  # try with many and only one.
      x = jnp.matmul(x, jnp.swapaxes(x, -1, -2))
      threshold = 1e-4

      sqrt_x, inv_sqrt_x, errors = matrix_square_root.sqrtm(
          x, min_iterations=self.dim, threshold=threshold
      )
      err = errors[errors > -1][-1]
      assert threshold > err
      np.testing.assert_allclose(
          x, jnp.matmul(sqrt_x, sqrt_x), rtol=1e-3, atol=1e-3
      )
      ids = jnp.eye(self.dim)
      if jnp.ndim(x) == 3:
        ids = ids[jnp.newaxis, :, :]
      np.testing.assert_allclose(
          jnp.zeros_like(x),
          jnp.matmul(x, jnp.matmul(inv_sqrt_x, inv_sqrt_x)) - ids,
          atol=1e-2
      )

  @pytest.mark.fast
  def test_sqrtm_batch(self):
    """Check sqrtm on larger of matrices."""
    batch_dim0 = 2
    batch_dim1 = 2
    threshold = 1e-4

    m = jax.random.normal(
        self.rng, (batch_dim0, batch_dim1, self.dim, 2 * self.dim)
    )
    x = jnp.matmul(m, jnp.swapaxes(m, axis1=-2, axis2=-1))
    sqrt_x, inv_sqrt_x, errors = matrix_square_root.sqrtm(
        x,
        threshold=threshold,
        min_iterations=self.dim,
    )

    err = errors[errors > -1][-1]
    assert threshold > err

    eye = jnp.eye(self.dim)
    for i in range(batch_dim0):
      for j in range(batch_dim1):
        np.testing.assert_allclose(
            x[i, j],
            jnp.matmul(sqrt_x[i, j], sqrt_x[i, j]),
            rtol=1e-3,
            atol=1e-3
        )
        np.testing.assert_allclose(
            eye,
            jnp.matmul(x[i, j], jnp.matmul(inv_sqrt_x[i, j], inv_sqrt_x[i, j])),
            atol=1e-2
        )

  # requires Schur decomposition, which jax does not implement on GPU
  @pytest.mark.cpu
  def test_solve_bartels_stewart(self):
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=self.a[0], b=self.b[0], c=self.c[0]
    )
    np.testing.assert_allclose(self.x[0], x, atol=1.e-5)

  # requires Schur decomposition, which jax does not implement on GPU
  @pytest.mark.cpu
  def test_solve_bartels_stewart_batch(self):
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=self.a, b=self.b, c=self.c
    )
    np.testing.assert_allclose(self.x, x, atol=1.e-5)
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=self.a[None], b=self.b[None], c=self.c[None]
    )
    np.testing.assert_allclose(self.x, x[0], atol=1.e-5)
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=self.a[None, None], b=self.b[None, None], c=self.c[None, None]
    )
    np.testing.assert_allclose(self.x, x[0, 0], atol=1.e-5)

  # requires Schur decomposition, which jax does not implement on GPU
  @pytest.mark.cpu
  @pytest.mark.fast.with_args(
      "fn,n_tests,dim,epsilon,atol,rtol",
      [(lambda x: matrix_square_root.sqrtm(x)[0], 3, 3, 1e-6, 1e-6, 1e-6),
       (lambda x: matrix_square_root.sqrtm(x)[1], 3, 3, 1e-6, 1e-8, 1e-8),
       (_sqrt_plus_inv_sqrt, 3, 3, 1e-6, 1e-8, 1e-8),
       (matrix_square_root.sqrtm_only, 3, 3, 1e-6, 1e-8, 1e-8),
       (matrix_square_root.inv_sqrtm_only, 3, 2, 1e-6, 1e-8, 1e-8)],
      ids=[
          "sqrtm_sqrtm", "sqrtm_inv_sqrtm", "sqrtm_sqrtm_plus_inv_sqrtm",
          "sqrtm_only", "inv_sqrtm_only"
      ],
      only_fast=-1,
  )
  def test_grad(
      self, enable_x64, fn: Callable, n_tests: int, dim: int, epsilon: float,
      atol: float, rtol: float
  ):
    key = self.rng
    for _ in range(n_tests):
      key, subkey = jax.random.split(key)
      test_fn = _get_test_fn(fn, dim=dim, key=subkey, threshold=1e-5)
      expected = (test_fn(epsilon) - test_fn(-epsilon)) / (2. * epsilon)
      actual = jax.grad(test_fn)(0.)
      np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
