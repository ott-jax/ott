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
import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.test_util

from ott.tools.gaussian_mixture import fit_gmm, gaussian_mixture


@pytest.fixture(scope="module")
def samples(gmm_reference: gaussian_mixture.GaussianMixture) -> jnp.ndarray:
  """Points drawn from the reference mixture."""
  return gmm_reference.sample(rng=jr.key(0), size=2000)


@pytest.mark.fast()
class TestFitGmm:

  def test_integration(self, rng: jax.Array, samples: jnp.ndarray):
    # dumb integration test that makes sure nothing crashes

    # Fit a GMM to the samples
    gmm_init = fit_gmm.initialize(
        rng=rng,
        points=samples,
        point_weights=None,
        n_components=3,
        verbose=False
    )
    _ = fit_gmm.fit_model_em(
        gmm=gmm_init, points=samples, point_weights=None, steps=20
    )
