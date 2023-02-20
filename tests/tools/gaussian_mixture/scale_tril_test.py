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
"""Tests for ScaleTriL."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.math import matrix_square_root
from ott.tools.gaussian_mixture import scale_tril


@pytest.fixture()
def chol() -> scale_tril.ScaleTriL:
  params = jnp.array([0., 2., jnp.log(3.)])
  return scale_tril.ScaleTriL(params=params, size=2)


@pytest.mark.fast
class TestScaleTriL:

  def test_cholesky(self, chol: scale_tril.ScaleTriL):
    expected = jnp.array([[1., 0.], [2., 3.]])
    np.testing.assert_allclose(chol.cholesky(), expected, atol=1e-4, rtol=1e-4)

  def test_covariance(self, chol: scale_tril.ScaleTriL):
    expected = jnp.array([[1., 0.], [2., 3.]])
    np.testing.assert_allclose(chol.covariance(), expected @ expected.T)

  def test_covariance_sqrt(self, chol: scale_tril.ScaleTriL):
    actual = chol.covariance_sqrt()
    expected = matrix_square_root.sqrtm_only(chol.covariance())
    np.testing.assert_allclose(expected, actual, atol=1e-4, rtol=1e-4)

  def test_log_det_covariance(self, chol: scale_tril.ScaleTriL):
    expected = jnp.log(jnp.linalg.det(chol.covariance()))
    actual = chol.log_det_covariance()
    np.testing.assert_almost_equal(actual, expected)

  def test_from_random(self, rng: jnp.ndarray):
    n_dimensions = 4
    cov = scale_tril.ScaleTriL.from_random(
        key=rng, n_dimensions=n_dimensions, stdev=0.1
    )
    np.testing.assert_array_equal(
        cov.cholesky().shape, (n_dimensions, n_dimensions)
    )

  def test_from_cholesky(self, rng: jnp.ndarray):
    n_dimensions = 4
    cholesky = scale_tril.ScaleTriL.from_random(
        key=rng, n_dimensions=n_dimensions, stdev=1.
    ).cholesky()
    scale = scale_tril.ScaleTriL.from_cholesky(cholesky)
    np.testing.assert_allclose(cholesky, scale.cholesky(), atol=1e-4, rtol=1e-4)

  def test_w2_dist(self, rng: jnp.ndarray):
    # make sure distance between a random normal and itself is 0
    key, subkey = jax.random.split(rng)
    s = scale_tril.ScaleTriL.from_random(key=subkey, n_dimensions=3)
    w2 = s.w2_dist(s)
    expected = 0.
    np.testing.assert_allclose(expected, w2, atol=1e-4, rtol=1e-4)

    # When covariances commute (e.g. if covariance is diagonal), have
    # distance between covariances = Frobenius norm^2 of (delta sqrt(cov))
    # see https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/  # pylint: disable=line-too-long
    size = 4
    key, subkey0, subkey1 = jax.random.split(key, num=3)
    diag0 = jnp.exp(jax.random.normal(key=subkey0, shape=(size,)))
    diag1 = jnp.exp(jax.random.normal(key=subkey1, shape=(size,)))
    s0 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag0))
    s1 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag1))
    w2 = s0.w2_dist(s1)
    delta_sigma = jnp.sum((jnp.sqrt(diag0) - jnp.sqrt(diag1)) ** 2.)
    np.testing.assert_allclose(delta_sigma, w2, atol=1e-4, rtol=1e-4)

  def test_transport(self, rng: jnp.ndarray):
    size = 4
    key, subkey0, subkey1 = jax.random.split(rng, num=3)
    diag0 = jnp.exp(jax.random.normal(key=subkey0, shape=(size,)))
    s0 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag0))
    diag1 = jnp.exp(jax.random.normal(key=subkey1, shape=(size,)))
    s1 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag1))

    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(100, size))
    transported = s0.transport(s1, points=x)
    expected = x * jnp.sqrt(diag1)[None] / jnp.sqrt(diag0)[None]
    np.testing.assert_allclose(expected, transported, atol=1e-4, rtol=1e-4)

  def test_flatten_unflatten(self, rng: jnp.ndarray):
    scale = scale_tril.ScaleTriL.from_random(key=rng, n_dimensions=3)
    children, aux_data = jax.tree_util.tree_flatten(scale)
    scale_new = jax.tree_util.tree_unflatten(aux_data, children)
    np.testing.assert_array_equal(scale.params, scale_new.params)
    assert scale == scale_new

  def test_pytree_mapping(self, rng: jnp.ndarray):
    scale = scale_tril.ScaleTriL.from_random(key=rng, n_dimensions=3)
    scale_x_2 = jax.tree_map(lambda x: 2 * x, scale)
    np.testing.assert_allclose(2. * scale.params, scale_x_2.params)
