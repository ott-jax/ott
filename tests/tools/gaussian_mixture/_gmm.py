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
"""Reference mixtures shared by the Gaussian mixture tests."""
import jax.numpy as jnp

from ott.tools.gaussian_mixture import gaussian_mixture

#: Parameters of the reference 3-component mixture in 2D.
MEAN = jnp.array([[2.0, -1.0], [-2.0, 0.0], [4.0, 3.0]])
COV = jnp.array([[[0.2, 0.0], [0.0, 0.1]], [[0.6, 0.0], [0.0, 0.3]],
                 [[0.5, 0.4], [0.4, 0.5]]])
WEIGHTS = jnp.array([0.3, 0.3, 0.4])

#: How the second mixture of a pair differs from the reference one.
MEAN_SHIFT = jnp.array([[1.0, -0.5], [-1.0, -1.0], [-1.0, 0.0]])
WEIGHTS_SHIFT = jnp.array([0.0, 0.1, -0.1])

#: Number of points sampled from a mixture.
NUM_SAMPLES = 2000


def reference() -> gaussian_mixture.GaussianMixture:
  """Reference mixture that the fitting tests sample from."""
  return gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
      mean=MEAN, cov=COV, component_weights=WEIGHTS
  )


def shifted() -> gaussian_mixture.GaussianMixture:
  """:func:`reference` with its means and component weights shifted."""
  return gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
      mean=MEAN + MEAN_SHIFT,
      cov=COV,
      component_weights=WEIGHTS + WEIGHTS_SHIFT
  )
