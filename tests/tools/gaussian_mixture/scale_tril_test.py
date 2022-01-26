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

"""Tests for google3.experimental.users.geoffd.contour.clustering.ot.parameters.scale_tril_params."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax.test_util

from ott.geometry import matrix_square_root
from ott.tools.gaussian_mixture import scale_tril


def get_w2_dist(scale0: scale_tril.ScaleTriL,
                scale1: scale_tril.ScaleTriL) -> jnp.ndarray:
  """Get Wasserstein distance W_2^2 to another Gaussian with same mean."""
  sigma0 = scale0.covariance()
  sigma1 = scale1.covariance()
  sqrt0 = scale0.covariance_sqrt()
  m = matrix_square_root.sqrtm_only(
      jnp.matmul(sqrt0, jnp.matmul(sigma1, sqrt0)))
  return jnp.trace(sigma0 + sigma1 - 2. * m, axis1=-2, axis2=-1)


class ScaleTriLTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    # note there is a little slop because of the diag_shift parameter in
    # the tfp.bijectors.FillScaleTriL bijector
    params = jnp.array([0., 2., jnp.log(3.)])
    self.chol = scale_tril.ScaleTriL(params=params, size=2)
    self.m_chol = jnp.array([[1., 0.], [2., 3.]])
    self.m_cov = jnp.matmul(self.m_chol, self.m_chol.T)
    self.key = jax.random.PRNGKey(seed=0)

  def test_cholesky(self):
    self.assertArraysAllClose(
        self.m_chol, self.chol.cholesky(), atol=1e-4, rtol=1e-4)

  def test_covariance(self):
    self.assertArraysAllClose(
        self.m_cov, self.chol.covariance())

  def test_covariance_sqrt(self):
    actual = self.chol.covariance_sqrt()
    expected = matrix_square_root.sqrtm_only(self.chol.covariance())
    self.assertArraysAllClose(expected, actual, atol=1e-4, rtol=1e-4)

  def test_log_det_covariance(self):
    expected = jnp.log(jnp.linalg.det(self.chol.covariance()))
    actual = self.chol.log_det_covariance()
    self.assertAlmostEqual(actual, expected)

  def test_from_random(self):
    n_dimensions = 4
    cov = scale_tril.ScaleTriL.from_random(
        key=self.key, n_dimensions=n_dimensions, stdev=0.1)
    self.assertEqual(cov.cholesky().shape, (n_dimensions, n_dimensions))

  def test_from_cholesky(self):
    n_dimensions = 4
    cholesky = scale_tril.ScaleTriL.from_random(
        key=self.key, n_dimensions=n_dimensions, stdev=1.).cholesky()
    scale = scale_tril.ScaleTriL.from_cholesky(cholesky)
    self.assertArraysAllClose(cholesky, scale.cholesky(), atol=1e-4, rtol=1e-4)

  def test_w2_dist(self):
    # make sure distance between a random normal and itself is 0
    key, subkey = jax.random.split(self.key)
    s = scale_tril.ScaleTriL.from_random(key=subkey, n_dimensions=3)
    w2 = s.w2_dist(s)
    expected = 0.
    self.assertAllClose(expected, w2, atol=1e-4, rtol=1e-4)

    # When covariances commute (e.g. if covariance is diagonal), have
    # distance between covariances = frobenius norm^2 of (delta sqrt(cov))
    # see https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/  # pylint: disable=line-too-long
    size = 4
    key, subkey0, subkey1 = jax.random.split(key, num=3)
    diag0 = jnp.exp(jax.random.normal(key=subkey0, shape=(size,)))
    diag1 = jnp.exp(jax.random.normal(key=subkey1, shape=(size,)))
    s0 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag0))
    s1 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag1))
    w2 = s0.w2_dist(s1)
    delta_sigma = jnp.sum((jnp.sqrt(diag0) - jnp.sqrt(diag1))**2.)
    self.assertArraysAllClose(delta_sigma, w2, atol=1e-4, rtol=1e-4)

  def test_transport(self):
    size = 4
    key, subkey0, subkey1 = jax.random.split(self.key, num=3)
    diag0 = jnp.exp(jax.random.normal(key=subkey0, shape=(size,)))
    s0 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag0))
    diag1 = jnp.exp(jax.random.normal(key=subkey1, shape=(size,)))
    s1 = scale_tril.ScaleTriL.from_covariance(jnp.diag(diag1))

    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(100, size))
    transported = s0.transport(s1, points=x)
    expected = x * jnp.sqrt(diag1)[None] / jnp.sqrt(diag0)[None]
    self.assertAllClose(expected, transported, atol=1e-4, rtol=1e-4)

  def test_flatten_unflatten(self):
    scale = scale_tril.ScaleTriL.from_random(key=self.key, n_dimensions=3)
    children, aux_data = jax.tree_util.tree_flatten(scale)
    scale_new = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertArraysEqual(scale.params, scale_new.params)
    self.assertEqual(scale, scale_new)

  def test_pytree_mapping(self):
    scale = scale_tril.ScaleTriL.from_random(key=self.key, n_dimensions=3)
    scale_x_2 = jax.tree_map(lambda x: 2 * x, scale)
    self.assertArraysAllClose(2. * scale.params, scale_x_2.params)

if __name__ == '__main__':
  absltest.main()
