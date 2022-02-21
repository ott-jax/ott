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

"""Tests for probabilities."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax.test_util

from ott.tools.gaussian_mixture import probabilities


@jax.test_util.with_config(jax_numpy_rank_promotion='allow')
class ProbabilitiesTest(jax.test_util.JaxTestCase):

  def test_probs(self):
    pp = probabilities.Probabilities(jnp.array([1., 2.]))
    probs = pp.probs()
    self.assertEqual(probs.shape, (3,))
    self.assertAlmostEqual(jnp.sum(probs), 1., places=6)
    self.assertTrue(jnp.all(probs > 0.))

  def test_log_probs(self):
    pp = probabilities.Probabilities(jnp.array([1., 2.]))
    log_probs = pp.log_probs()
    self.assertEqual(log_probs.shape, (3,))
    probs = jnp.exp(log_probs)
    self.assertAlmostEqual(jnp.sum(probs), 1., places=6)
    self.assertTrue(jnp.all(probs > 0.))

  def test_from_random(self):
    n_dimensions = 4
    key = jax.random.PRNGKey(0)
    pp = probabilities.Probabilities.from_random(
        key=key, n_dimensions=n_dimensions, stdev=0.1)
    self.assertEqual(pp.probs().shape, (4,))

  def test_from_probs(self):
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    pp = probabilities.Probabilities.from_probs(probs)
    self.assertArraysAllClose(probs, pp.probs())

  def test_sample(self):
    p = 0.4
    probs = jnp.array([p, 1. - p])
    pp = probabilities.Probabilities.from_probs(probs)
    samples = pp.sample(key=jax.random.PRNGKey(0), size=10000)
    sd = jnp.sqrt(p * (1. - p))
    self.assertAllClose(jnp.mean(samples == 0), p, atol=3. * sd)

  def test_flatten_unflatten(self):
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    pp = probabilities.Probabilities.from_probs(probs)
    children, aux_data = jax.tree_util.tree_flatten(pp)
    pp_new = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertArraysEqual(pp.params, pp_new.params)
    self.assertEqual(pp, pp_new)

  def test_pytree_mapping(self):
    probs = jnp.array([0.1, 0.2, 0.3, 0.4])
    pp = probabilities.Probabilities.from_probs(probs)
    pp_x_2 = jax.tree_map(lambda x: 2 * x, pp)
    self.assertArraysAllClose(2. * pp.params, pp_x_2.params)

if __name__ == '__main__':
  absltest.main()
