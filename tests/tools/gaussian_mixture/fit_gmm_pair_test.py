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
import jax
import jax.numpy as jnp
import pytest
from ott.tools.gaussian_mixture import (
    fit_gmm,
    fit_gmm_pair,
    gaussian_mixture,
    gaussian_mixture_pair,
)


class TestFitGmmPair:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    mean_generator0 = jnp.array([[2., -1.], [-2., 0.], [4., 3.]])
    cov_generator0 = jnp.array([[[0.2, 0.], [0., 0.1]], [[0.6, 0.], [0., 0.3]],
                                [[0.5, 0.4], [0.4, 0.5]]])
    weights_generator0 = jnp.array([0.3, 0.3, 0.4])

    gmm_generator0 = (
        gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
            mean=mean_generator0,
            cov=cov_generator0,
            component_weights=weights_generator0
        )
    )

    # shift the means to the right by varying amounts
    mean_generator1 = mean_generator0 + jnp.array([[1., -0.5], [-1., -1.],
                                                   [-1., 0.]])
    cov_generator1 = cov_generator0
    weights_generator1 = weights_generator0 + jnp.array([0., 0.1, -0.1])

    gmm_generator1 = (
        gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
            mean=mean_generator1,
            cov=cov_generator1,
            component_weights=weights_generator1
        )
    )

    self.epsilon = 1.e-2
    self.rho = 0.1
    self.tau = self.rho / (self.rho + self.epsilon)

    self.rng, subrng0, subrng1 = jax.random.split(rng, num=3)
    self.samples_gmm0 = gmm_generator0.sample(rng=subrng0, size=2000)
    self.samples_gmm1 = gmm_generator1.sample(rng=subrng1, size=2000)

  # requires Schur decomposition, which jax does not implement on GPU
  @pytest.mark.cpu()
  @pytest.mark.fast.with_args(
      balanced=[False, True], weighted=[False, True], only_fast=0
  )
  def test_fit_gmm(self, balanced, weighted):
    # dumb integration test that makes sure nothing crashes
    tau = 1.0 if balanced else self.tau

    if weighted:
      weights0 = jnp.ones(self.samples_gmm0.shape[0])
      weights1 = jnp.ones(self.samples_gmm0.shape[0])
      weights_pooled = jnp.concatenate([weights0, weights1], axis=0)
    else:
      weights0 = None
      weights1 = None
      weights_pooled = None

      # Fit a GMM to the pooled samples
    samples = jnp.concatenate([self.samples_gmm0, self.samples_gmm1])
    gmm_init = fit_gmm.initialize(
        rng=self.rng,
        points=samples,
        point_weights=weights_pooled,
        n_components=3,
        verbose=False
    )
    gmm = fit_gmm.fit_model_em(
        gmm=gmm_init, points=samples, point_weights=None, steps=20
    )
    # use the same mixture model for gmm0 and gmm1 initially
    pair_init = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=gmm, gmm1=gmm, epsilon=self.epsilon, tau=tau
    )
    fit_model_em_fn = fit_gmm_pair.get_fit_model_em_fn(
        weight_transport=0.1, jit=True
    )
    fit_model_em_fn(
        pair=pair_init,
        points0=self.samples_gmm0,
        points1=self.samples_gmm1,
        point_weights0=weights0,
        point_weights1=weights1,
        em_steps=1,
        m_steps=10,
        verbose=False
    )
