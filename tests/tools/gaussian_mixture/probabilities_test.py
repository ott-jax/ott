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
"""Tests for probabilities."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.tools.gaussian_mixture import probabilities


@pytest.mark.fast
class TestProbabilities:

  def test_probs(self):
    pp = probabilities.Probabilities(jnp.array([1., 2.]))
    probs = pp.probs()
    np.testing.assert_array_equal(probs.shape, (3,))
    np.testing.assert_allclose(jnp.sum(probs), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(probs > 0., True)

  def test_log_probs(self):
    pp = probabilities.Probabilities(jnp.array([1., 2.]))
    log_probs = pp.log_probs()
    probs = jnp.exp(log_probs)

    np.testing.assert_array_equal(log_probs.shape, (3,))
    np.testing.assert_array_equal(probs.shape, (3,))
    np.testing.assert_allclose(jnp.sum(probs), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(probs > 0., True)

  def test_from_random(self):
    n_dimensions = 4
    key = jax.random.PRNGKey(0)
    pp = probabilities.Probabilities.from_random(
        key=key, n_dimensions=n_dimensions, stdev=0.1
    )
    np.testing.assert_array_equal(pp.probs().shape, (4,))

  def test_from_probs(self):
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    pp = probabilities.Probabilities.from_probs(probs)
    np.testing.assert_allclose(probs, pp.probs(), rtol=1e-6, atol=1e-6)

  def test_sample(self):
    p = 0.4
    probs = jnp.array([p, 1. - p])
    pp = probabilities.Probabilities.from_probs(probs)
    samples = pp.sample(key=jax.random.PRNGKey(0), size=10000)
    sd = jnp.sqrt(p * (1. - p))
    np.testing.assert_allclose(jnp.mean(samples == 0), p, atol=3. * sd)

  def test_flatten_unflatten(self):
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    pp = probabilities.Probabilities.from_probs(probs)
    children, aux_data = jax.tree_util.tree_flatten(pp)
    pp_new = jax.tree_util.tree_unflatten(aux_data, children)
    np.testing.assert_array_equal(pp.params, pp_new.params)
    assert pp == pp_new

  def test_pytree_mapping(self):
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    pp = probabilities.Probabilities.from_probs(probs)
    pp_x_2 = jax.tree_map(lambda x: 2 * x, pp)
    np.testing.assert_allclose(
        2. * pp.params, pp_x_2.params, rtol=1e-6, atol=1e-6
    )
