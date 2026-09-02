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
from typing import Tuple

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr

from ott.tools.gaussian_mixture import (
    fit_gmm,
    fit_gmm_pair,
    gaussian_mixture,
    gaussian_mixture_pair,
)

EPSILON = 1e-2


@pytest.fixture(scope="module")
def samples(
    gmm_reference: gaussian_mixture.GaussianMixture,
    gmm_shifted: gaussian_mixture.GaussianMixture
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Points drawn from the reference mixture and from its shifted variant."""
  rng0, rng1 = jr.split(jr.key(0), 2)
  return (
      gmm_reference.sample(rng=rng0, size=2000),
      gmm_shifted.sample(rng=rng1, size=2000),
  )


class TestFitGmmPair:

  # requires Schur decomposition, which jax does not implement on GPU
  @pytest.mark.cpu()
  @pytest.mark.fast.with_args(
      balanced=[False, True], weighted=[False, True], only_fast=0
  )
  def test_fit_gmm(
      self, rng: jax.Array, samples: Tuple[jnp.ndarray, jnp.ndarray],
      balanced: bool, weighted: bool
  ):
    # dumb integration test that makes sure nothing crashes
    samples0, samples1 = samples
    rho = 0.1
    tau = 1.0 if balanced else rho / (rho + EPSILON)

    if weighted:
      weights0 = jnp.ones(samples0.shape[0])
      weights1 = jnp.ones(samples0.shape[0])
      weights_pooled = jnp.concatenate([weights0, weights1], axis=0)
    else:
      weights0 = None
      weights1 = None
      weights_pooled = None

    # Fit a GMM to the pooled samples
    pooled = jnp.concatenate([samples0, samples1])
    gmm_init = fit_gmm.initialize(
        rng=rng,
        points=pooled,
        point_weights=weights_pooled,
        n_components=3,
        verbose=False
    )
    gmm = fit_gmm.fit_model_em(
        gmm=gmm_init, points=pooled, point_weights=None, steps=20
    )
    # use the same mixture model for gmm0 and gmm1 initially
    pair_init = gaussian_mixture_pair.GaussianMixturePair(
        gmm0=gmm, gmm1=gmm, epsilon=EPSILON, tau=tau
    )
    fit_model_em_fn = fit_gmm_pair.get_fit_model_em_fn(
        weight_transport=0.1, jit=True
    )
    fit_model_em_fn(
        pair=pair_init,
        points0=samples0,
        points1=samples1,
        point_weights0=weights0,
        point_weights1=weights1,
        em_steps=1,
        m_steps=10,
        verbose=False
    )
