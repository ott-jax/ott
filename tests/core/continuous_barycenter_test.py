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
from ott.geometry import pointcloud
from ott.core import bar_problems
from ott.core import continuous_barycenter

# from jax.config import config
# config.update('jax_disable_jit', True)
class Barycenter(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self._dim = 4
    self._num_points = 113
    self.rng, *rngs = jax.random.split(self.rng, 3)
    b = jax.random.uniform(rngs[1], (self._num_points,))
    self._b = b / jnp.sum(b)
  
  @parameterized.product(
      rank=[-1, 6],
      epsilon=[1e-1, 1e-2],
      debiased=[True, False],
      jit=[True, False])
  def test_euclidean_barycenter(self, rank, epsilon, debiased, jit):
    print('Rank: ', rank, 'Epsilon: ', epsilon, 'Debiased', debiased)
    rngs = jax.random.split(self.rng, 2)    
    y1 = jax.random.uniform(rngs[0], (self._num_points, self._dim))
    y2 = jax.random.uniform(rngs[1], (self._num_points, self._dim)) + 2
    y = jnp.concatenate((y1, y2))
    bar_prob = bar_problems.BarycenterProblem(
      y, num_per_segment=jnp.array([32, 29, 23, 29, 25, 29, 27, 22]),
      num_segments=8,
      max_measure_size=40,
      debiased=debiased)
    threshold = 1e-3
    kwargs = {'gamma' : 1} if rank > 0 else {}
    solver = continuous_barycenter.WassersteinBarycenter(
      epsilon=epsilon,
      rank=rank,
      threshold = threshold, jit=jit,
      **kwargs)
    out = solver(bar_prob, bar_size=31)
    costs = out.costs
    costs = costs[costs > -1]
    # Check convergence
    self.assertTrue(jnp.isclose(costs[-2], costs[-1], rtol=threshold))
    # Check barycenter has points roughly in [1,2]^4.
    # (Note sampled points where either in [0,1]^4 or [2,3]^4)
    self.assertTrue(jnp.all(out.x.ravel()<2.1))
    self.assertTrue(jnp.all(out.x.ravel()>.9))

if __name__ == '__main__':
  absltest.main()