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
"""Tests for linalg."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.tools.gaussian_mixture import linalg


@pytest.mark.fast
class TestLinalg:

  def test_get_mean_and_var(self, rng: jnp.ndarray):
    points = jax.random.normal(key=rng, shape=(10, 2))
    weights = jnp.ones(10)
    expected_mean = jnp.mean(points, axis=0)
    expected_var = jnp.var(points, axis=0)
    actual_mean, actual_var = linalg.get_mean_and_var(
        points=points, weights=weights
    )
    np.testing.assert_allclose(expected_mean, actual_mean, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(expected_var, actual_var, atol=1E-5, rtol=1E-5)

  def test_get_mean_and_var_nonuniform_weights(self, rng: jnp.ndarray):
    points = jax.random.normal(key=rng, shape=(10, 2))
    weights = jnp.concatenate([jnp.ones(5), jnp.zeros(5)], axis=-1)
    expected_mean = jnp.mean(points[:5], axis=0)
    expected_var = jnp.var(points[:5], axis=0)
    actual_mean, actual_var = linalg.get_mean_and_var(
        points=points, weights=weights
    )
    np.testing.assert_allclose(expected_mean, actual_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(expected_var, actual_var, rtol=1e-6, atol=1e-6)

  def test_get_mean_and_cov(self, rng: jnp.ndarray):
    points = jax.random.normal(key=rng, shape=(10, 2))
    weights = jnp.ones(10)
    expected_mean = jnp.mean(points, axis=0)
    expected_cov = jnp.cov(points, rowvar=False, bias=True)
    actual_mean, actual_cov = linalg.get_mean_and_cov(
        points=points, weights=weights
    )
    np.testing.assert_allclose(expected_mean, actual_mean, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(expected_cov, actual_cov, atol=1e-5, rtol=1e-5)

  def test_get_mean_and_cov_nonuniform_weights(self, rng: jnp.ndarray):
    points = jax.random.normal(key=rng, shape=(10, 2))
    weights = jnp.concatenate([jnp.ones(5), jnp.zeros(5)], axis=-1)
    expected_mean = jnp.mean(points[:5], axis=0)
    expected_cov = jnp.cov(points[:5], rowvar=False, bias=True)
    actual_mean, actual_cov = linalg.get_mean_and_cov(
        points=points, weights=weights
    )
    np.testing.assert_allclose(expected_mean, actual_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(expected_cov, actual_cov, rtol=1e-6, atol=1e-6)

  def test_flat_to_tril(self, rng: jnp.ndarray):
    size = 3
    x = jax.random.normal(key=rng, shape=(5, 4, size * (size + 1) // 2))
    m = linalg.flat_to_tril(x, size)
    # check size of m
    np.testing.assert_array_equal(m.shape, (5, 4, size, size))

    # make sure m is lower triangular
    for i in range(size):
      for j in range(size):
        if j <= i:
          continue
        np.testing.assert_array_equal(
            m[..., i, j], jnp.zeros_like(m[..., i, j])
        )

    # make sure we can invert
    actual = linalg.tril_to_flat(m)
    np.testing.assert_allclose(x, actual)

  def test_tril_to_flat(self, rng: jnp.ndarray):
    size = 3
    m = jax.random.normal(key=rng, shape=(5, 4, size, size))
    for i in range(size):
      for j in range(size):
        if j > i:
          m = m.at[..., i, j].set(0.)
    m = jnp.array(m)
    flat = linalg.tril_to_flat(m)

    # check size of flat
    np.testing.assert_array_equal(flat.shape, (5, 4, size * (size + 1) // 2))

    # make sure flattening is invertible
    inverted = linalg.flat_to_tril(flat, size)
    np.testing.assert_allclose(m, inverted)

  def test_apply_to_diag(self, rng: jnp.ndarray):
    size = 3
    m = jax.random.normal(key=rng, shape=(5, 4, size, size))
    mnew = linalg.apply_to_diag(m, jnp.exp)
    for i in range(size):
      for j in range(size):
        if i != j:
          np.testing.assert_allclose(m[..., i, j], mnew[..., i, j])
        else:
          np.testing.assert_allclose(jnp.exp(m[..., i, j]), mnew[..., i, j])

  def test_matrix_powers(self, rng: jnp.ndarray):
    key, subkey = jax.random.split(rng)
    m = jax.random.normal(key=subkey, shape=(4, 4))
    m += jnp.swapaxes(m, axis1=-2, axis2=-1)  # symmetric
    m = jnp.matmul(m, m)  # symmetric, pos def
    inv_m = jnp.linalg.inv(m)
    msq = jnp.matmul(m, m)
    actual = linalg.matrix_powers(msq, powers=(0.5, -0.5))
    np.testing.assert_allclose(m, actual[0], rtol=1.e-5)
    np.testing.assert_allclose(inv_m, actual[1], rtol=1.e-4)

  def test_invmatvectril(self, rng: jnp.ndarray):
    key, subkey = jax.random.split(rng)
    m = jax.random.normal(key=subkey, shape=(2, 2))
    m += jnp.swapaxes(m, axis1=-2, axis2=-1)  # symmetric
    m = jnp.matmul(m, m)  # symmetric, pos def
    cholesky = jnp.linalg.cholesky(m)  # lower triangular
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(10, 2))
    inv_cholesky = jnp.linalg.inv(cholesky)
    expected = jnp.transpose(jnp.matmul(inv_cholesky, jnp.transpose(x)))
    actual = linalg.invmatvectril(m=cholesky, x=x, lower=True)
    np.testing.assert_allclose(expected, actual, atol=1e-4, rtol=1.e-4)

  def test_get_random_orthogonal(self, rng: jnp.ndarray):
    key, subkey = jax.random.split(rng)
    q = linalg.get_random_orthogonal(key=subkey, dim=3)
    qt = jnp.transpose(q)
    expected = jnp.eye(3)
    actual = jnp.matmul(q, qt)

    assert jnp.linalg.norm(expected - actual) < 1e-4
