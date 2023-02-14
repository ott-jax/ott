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
"""Tests for fit_gmm_pair."""

import pytest

import jax
import jax.numpy as jnp
import jax.test_util

from ott.tools.gaussian_mixture import fit_gmm, gaussian_mixture


@pytest.mark.fast
class TestFitGmm:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    mean_generator = jnp.array([[2., -1.], [-2., 0.], [4., 3.]])
    cov_generator = jnp.array([[[0.2, 0.], [0., 0.1]], [[0.6, 0.], [0., 0.3]],
                               [[0.5, 0.4], [0.4, 0.5]]])
    weights_generator = jnp.array([0.3, 0.3, 0.4])

    gmm_generator = (
        gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
            mean=mean_generator,
            cov=cov_generator,
            component_weights=weights_generator
        )
    )

    self.key, subkey = jax.random.split(rng)
    self.samples = gmm_generator.sample(key=subkey, size=2000)

  def test_integration(self):
    # dumb integration test that makes sure nothing crashes

    # Fit a GMM to the samples
    gmm_init = fit_gmm.initialize(
        key=self.key,
        points=self.samples,
        point_weights=None,
        n_components=3,
        verbose=False
    )
    _ = fit_gmm.fit_model_em(
        gmm=gmm_init, points=self.samples, point_weights=None, steps=20
    )
