# coding=utf-8
# Copyright 2022 Apple
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

# Lint as: python3
"""Tests for Continuous barycenters."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
from ott.geometry import pointcloud
from ott.core import bar_problems
from ott.core import continuous_barycenter


@jax.test_util.with_config(jax_numpy_rank_promotion='allow')
class Barycenter(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self._dim = 4
    self._num_points = 113
    self.rng, *rngs = jax.random.split(self.rng, 3)
    b = jax.random.uniform(rngs[1], (self._num_points,))
    self._b = b / jnp.sum(b)

  def test_euclidean_barycenter(self):
    rngs = jax.random.split(self.rng, 2)    
    y = jax.random.uniform(rngs[0], (self._num_points, self._dim))
    bar_prob = bar_problems.BarycenterProblem(y, segment_ids=[45, 29, 15,24])
    solver = continuous_barycenter.WassersteinBarycenter(epsilon=.01)
    out = solver(bar_prob)
    self.assertLen(out, 3)
