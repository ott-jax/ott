# coding=utf-8
"""Tests for fit_gmm_pair."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax.test_util

from ott.tools.gaussian_mixture import fit_gmm
from ott.tools.gaussian_mixture import gaussian_mixture


class FitGmmTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mean_generator = jnp.array([[2., -1.],
                                [-2., 0.],
                                [4., 3.]])
    cov_generator = jnp.array([[[0.2, 0.], [0., 0.1]],
                               [[0.6, 0.], [0., 0.3]],
                               [[0.5, 0.4], [0.4, 0.5]]])
    weights_generator = jnp.array([0.3, 0.3, 0.4])

    gmm_generator = (
        gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
            mean=mean_generator,
            cov=cov_generator,
            component_weights=weights_generator))

    key = jax.random.PRNGKey(0)
    self.key, subkey = jax.random.split(key)
    self.samples = gmm_generator.sample(key=subkey, size=2000)

  def test_integration(self):
    # dumb integration test that makes sure nothing crashes

    # Fit a GMM to the samples
    gmm_init = fit_gmm.initialize(
        key=self.key,
        points=self.samples,
        point_weights=None,
        n_components=3,
        verbose=False)
    _ = fit_gmm.fit_model_em(
        gmm=gmm_init,
        points=self.samples,
        point_weights=None,
        steps=20)

if __name__ == '__main__':
  absltest.main()
