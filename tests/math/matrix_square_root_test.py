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
import dataclasses
from typing import Any, Callable

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.math import matrix_square_root


def _get_random_spd_matrix(dim: int, rng: jax.Array):
  # Get a random symmetric, positive definite matrix of a specified size.

  rng, rng0, rng1 = jr.split(rng, 3)
  # Step 1: generate a random orthogonal matrix
  m = jr.normal(rng0, shape=[dim, dim])
  q, _ = jnp.linalg.qr(m)

  # Step 2: generate random eigenvalues in [1/2. , 2.] to ensure the condition
  # number is reasonable.
  eigs = 2.0 ** (2.0 * jr.uniform(rng1, shape=(dim,)) - 1.0)

  return jnp.matmul(eigs[None, :] * q, jnp.transpose(q))


def _get_test_fn(
    fn: Callable[[jnp.ndarray], jnp.ndarray], dim: int, rng: jax.Array,
    **kwargs: Any
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  # We want to test gradients of a function fn that maps positive definite
  # matrices to positive definite matrices by comparing them to finite
  # difference approximations. We'll do so via a test function that
  # (1) takes an arbitrary real as an input,
  # (2) maps the real to a positive definite matrix,
  # (3) applies fn, then
  # (4) maps the matrix-valued output of fn to a scalar.
  rng, rng0, rng1, rng2, rng3 = jr.split(rng, num=5)
  m0 = _get_random_spd_matrix(dim=dim, rng=rng0)
  m1 = _get_random_spd_matrix(dim=dim, rng=rng1)
  dx = _get_random_spd_matrix(dim=dim, rng=rng2)
  unit = jr.normal(rng3, shape=(dim, dim))
  unit /= jnp.sqrt(jnp.sum(unit ** 2))

  def _test_fn(x: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
    # m is the product of 2 symmetric, positive definite matrices
    # so it will be positive definite but not necessarily symmetric
    m = jnp.matmul(m0, m1 + x * dx)
    return jnp.sum(fn(m, **kwargs) * unit)

  return _test_fn


def _sqrt_plus_inv_sqrt(x: jnp.ndarray) -> jnp.ndarray:
  sqrtm = matrix_square_root.sqrtm(x)
  return sqrtm[0] + sqrtm[1]


DIM = 13


@dataclasses.dataclass(frozen=True)
class Sylvester:
  """A Sylvester system ``a @ x - x @ b == c``, with a known solution."""
  a: jnp.ndarray  # (2, m, m)
  b: jnp.ndarray  # (2, n, n)
  x: jnp.ndarray  # (2, m, n), the solution
  c: jnp.ndarray  # (2, m, n)


@pytest.fixture(scope="module")
def sylvester() -> Sylvester:
  """A solvable Sylvester system."""
  m, n = 3, 2
  rng0, rng1, rng2 = jr.split(jr.key(0), 3)
  a = jr.normal(rng0, shape=(2, m, m))
  b = jr.normal(rng1, shape=(2, n, n))
  x = jr.normal(rng2, shape=(2, m, n))
  return Sylvester(a=a, b=b, x=x, c=jnp.matmul(a, x) - jnp.matmul(x, b))


class TestMatrixSquareRoot:

  def test_sqrtm(self, rng: jax.Array):
    """Sample a random p.s.d. (Wishart) matrix, check its sqrt matches."""

    matrices = jr.normal(rng, (3, DIM, 2 * DIM))

    for x in (matrices, matrices[0, :, :]):  # try with many and only one.
      x = jnp.matmul(x, jnp.swapaxes(x, -1, -2))
      threshold = 1e-4

      sqrt_x, inv_sqrt_x, errors = matrix_square_root.sqrtm(
          x, min_iterations=DIM, threshold=threshold
      )
      err = errors[errors > -1][-1]
      assert threshold > err
      np.testing.assert_allclose(
          x, jnp.matmul(sqrt_x, sqrt_x), rtol=1e-3, atol=1e-3
      )
      ids = jnp.eye(DIM)
      if jnp.ndim(x) == 3:
        ids = ids[jnp.newaxis, :, :]
      np.testing.assert_allclose(
          jnp.zeros_like(x),
          jnp.matmul(x, jnp.matmul(inv_sqrt_x, inv_sqrt_x)) - ids,
          atol=1e-2
      )

  @pytest.mark.fast()
  def test_sqrtm_batch(self, rng: jax.Array):
    """Check sqrtm on larger of matrices."""
    batch_dim0 = 2
    batch_dim1 = 2
    threshold = 1e-4

    m = jr.normal(rng, (batch_dim0, batch_dim1, DIM, 2 * DIM))
    x = jnp.matmul(m, jnp.swapaxes(m, axis1=-2, axis2=-1))
    sqrt_x, inv_sqrt_x, errors = matrix_square_root.sqrtm(
        x,
        threshold=threshold,
        min_iterations=DIM,
    )

    err = errors[errors > -1][-1]
    assert threshold > err

    eye = jnp.eye(DIM)
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
  @pytest.mark.cpu()
  def test_solve_bartels_stewart(self, sylvester: Sylvester):
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=sylvester.a[0], b=sylvester.b[0], c=sylvester.c[0]
    )
    np.testing.assert_allclose(sylvester.x[0], x, atol=1e-5)

  # requires Schur decomposition, which jax does not implement on GPU
  @pytest.mark.cpu()
  def test_solve_bartels_stewart_batch(self, sylvester: Sylvester):
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=sylvester.a, b=sylvester.b, c=sylvester.c
    )
    np.testing.assert_allclose(sylvester.x, x, atol=1e-4)
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=sylvester.a[None], b=sylvester.b[None], c=sylvester.c[None]
    )
    np.testing.assert_allclose(sylvester.x, x[0], atol=1e-4)
    x = matrix_square_root.solve_sylvester_bartels_stewart(
        a=sylvester.a[None, None],
        b=sylvester.b[None, None],
        c=sylvester.c[None, None]
    )
    np.testing.assert_allclose(sylvester.x, x[0, 0], atol=1e-4)

  # requires Schur decomposition, which jax does not implement on GPU
  @pytest.mark.cpu()
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
  @pytest.mark.usefixtures("enable_x64")
  def test_grad(
      self, rng: jax.Array, fn: Callable, n_tests: int, dim: int,
      epsilon: float, atol: float, rtol: float
  ):
    for _ in range(n_tests):
      rng, rng0 = jr.split(rng)
      test_fn = _get_test_fn(fn, dim=dim, rng=rng0, threshold=1e-5)
      expected = (test_fn(epsilon) - test_fn(-epsilon)) / (2.0 * epsilon)
      actual = jax.grad(test_fn)(0.0)
      np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
