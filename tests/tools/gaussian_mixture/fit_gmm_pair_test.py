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
    gaussian_mixture_pair,
)
from tests import _utils
from tests.tools.gaussian_mixture import _gmm

#: Regularization strength and unbalancedness of the pair.
EPSILON = 1e-2
RHO = 0.1
TAU = RHO / (RHO + EPSILON)


@pytest.fixture(scope="module")
def samples() -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Points drawn from the reference mixture and from its shifted variant."""
  _, subrng0, subrng1 = jr.split(_utils.root_key(), num=3)
  return (
      _gmm.reference().sample(rng=subrng0, size=_gmm.NUM_SAMPLES),
      _gmm.shifted().sample(rng=subrng1, size=_gmm.NUM_SAMPLES),
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
    tau = 1.0 if balanced else TAU

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
