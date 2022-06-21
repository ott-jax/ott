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
"""Tests for Continuous barycenter of GMMs with Bures cost."""

import jax.numpy as jnp
from absl.testing import absltest, parameterized

from ott.core import bar_problems, continuous_barycenter
from ott.geometry import costs


class Barycenter(parameterized.TestCase):

  def setUp(self):
    super().setUp()

  @parameterized.product(
      lse_mode=[True, False], epsilon=[1e-1, 5e-1], jit=[True, False]
  )
  def test_bures_barycenter(self, lse_mode, epsilon, jit):
    num_of_measures = 2
    num_of_components = 2
    dimension = 2
    bar_size = 2

    means1 = jnp.array([[-0.8, 0.8], [-0.8, -0.8]])
    means2 = jnp.array([[0.8, 0.8], [0.8, -0.8]])
    sigma = 0.01
    covs1 = sigma * jnp.asarray([
        jnp.eye(dimension) for i in range(num_of_components)
    ])
    covs2 = sigma * jnp.asarray([
        jnp.eye(dimension) for i in range(num_of_components)
    ])

    b1 = jnp.ones(num_of_components) / num_of_components
    b2 = jnp.ones(num_of_components) / num_of_components

    y1 = jnp.asarray([
        jnp.concatenate(
            (means1[i], jnp.reshape(covs1[i], (dimension * dimension,)))
        ) for i in range(num_of_components)
    ])

    y2 = jnp.asarray([
        jnp.concatenate(
            (means2[i], jnp.reshape(covs2[i], (dimension * dimension,)))
        ) for i in range(num_of_components)
    ])

    y = jnp.concatenate((y1, y2))
    b = jnp.concatenate((b1, b2))

    barycentric_weights = jnp.asarray([0.5, 0.5])
    bures_cost = costs.Bures(dimension=dimension)

    bar_p = bar_problems.BarycenterProblem(
        y,
        b,
        weights=barycentric_weights,
        num_per_segment=jnp.asarray([num_of_components, num_of_components]),
        num_segments=num_of_measures,
        max_measure_size=num_of_components,
        cost_fn=bures_cost,
        epsilon=epsilon
    )

    solver = continuous_barycenter.WassersteinBarycenter(
        lse_mode=lse_mode, jit=jit
    )

    out = solver(bar_p, bar_size=bar_size)
    barycenter = out.x

    means_bary, covs_bary = bures_cost.x_to_mean_and_cov(barycenter)

    self.assertTrue(jnp.allclose(means_bary, jnp.zeros((bar_size, dimension))))
    self.assertTrue(
        jnp.allclose(
            covs_bary,
            jnp.array([sigma * jnp.eye(dimension) for i in range(bar_size)])
        )
    )


if __name__ == '__main__':
  absltest.main()
