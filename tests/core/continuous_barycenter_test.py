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

class Barycenter(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self._dim = 4
    self._num_points = 113
    
  @parameterized.product(
      rank=[-1, 6],
      epsilon=[1e-1, 1e-2],
      debiased=[True, False],
      jit=[True, False],
      init_random=[True, False])
  def test_euclidean_barycenter(self, rank, epsilon, debiased, jit, init_random):
    print('Rank: ', rank, 'Epsilon: ', epsilon, 'Debiased', debiased)
    rngs = jax.random.split(self.rng, 20)
    # Sample 2 point clouds, each of size 113, the first around [0,1]^4,
    # Second around [2,3]^4.
    y1 = jax.random.uniform(rngs[0], (self._num_points, self._dim))
    y2 = jax.random.uniform(rngs[1], (self._num_points, self._dim)) + 2
    # Merge them
    y = jnp.concatenate((y1, y2))
    
    # Define segments
    num_per_segment = jnp.array([33, 29, 24, 27, 27, 31, 30, 25])
    # Set weights for each segment that sum to 1.
    b = []
    for i in range(num_per_segment.shape[0]):
      c = jax.random.uniform(rngs[i], (num_per_segment[i],))
      b.append(c / jnp.sum(c))
    b = jnp.concatenate(b, axis=0)
    print(b.shape)
    # Set a barycenter problem with 8 measures, of irregular sizes.

    bar_prob = bar_problems.BarycenterProblem(
      y, b, 
      num_per_segment=num_per_segment,
      num_segments=num_per_segment.shape[0],
      max_measure_size=jnp.max(num_per_segment)+3, # +3 set with no purpose.
      debiased=debiased)
    
    # Define solver
    threshold = 1e-3
    solver = continuous_barycenter.WassersteinBarycenter(
      epsilon=epsilon,
      rank=rank,
      threshold = threshold, jit=jit)
    
    # Set barycenter size to 31. 
    bar_size = 31

    # We consider either a random initialization, with points chosen
    # in [0,1]^4, or the default (init_random is False) where the 
    # initialization consists in selecting randomly points in the y's.    
    if init_random:
      # choose points randomly in area relevant to the problem.
      x_init= 3 * jax.random.uniform(rngs[-1], (bar_size, self._dim))
      out = solver(
        bar_prob, bar_size=bar_size, x_init=x_init)
    else:      
      out = solver(bar_prob, bar_size=bar_size)

    # Check shape is as expected
    self.assertTrue(out.x.shape==(bar_size,self._dim))

    # Check convergence by looking at cost evolution.
    costs = out.costs
    costs = costs[costs > -1]
    self.assertTrue(jnp.isclose(costs[-2], costs[-1], rtol=threshold))
    
    # Check barycenter has all points roughly in [1,2]^4.
    # (this is because sampled points where equally set in either [0,1]^4
    # or [2,3]^4)
    self.assertTrue(jnp.all(out.x.ravel()<2.3))
    self.assertTrue(jnp.all(out.x.ravel()>.7))

if __name__ == '__main__':
  absltest.main()
