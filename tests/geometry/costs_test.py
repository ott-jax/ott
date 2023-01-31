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
"""Tests for the cost/norm functions."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import costs


@pytest.mark.fast
class TestCostFn:

  def test_cosine(self, rng: jnp.ndarray):
    """Test the cosine cost function."""
    x = jnp.array([0, 0])
    y = jnp.array([0, 0])
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 0.0, rtol=1e-5, atol=1e-5)

    x = jnp.array([1.0, 0])
    y = jnp.array([1.0, 0])
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 1.0, rtol=1e-5, atol=1e-5)

    x = jnp.array([1.0, 0])
    y = jnp.array([-1.0, 0])
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - -1.0, rtol=1e-5, atol=1e-5)

    n, m, d = 10, 12, 7
    keys = jax.random.split(rng, 2)
    x = jax.random.normal(keys[0], (n, d))
    y = jax.random.normal(keys[1], (m, d))

    cosine_fn = costs.Cosine()
    normalize = lambda v: v / jnp.sqrt(jnp.sum(v ** 2))
    for i in range(n):
      for j in range(m):
        exp_sim_xi_yj = jnp.sum(normalize(x[i]) * normalize(y[j]))
        exp_dist_xi_yj = 1.0 - exp_sim_xi_yj
        np.testing.assert_allclose(
            cosine_fn.pairwise(x[i], y[j]),
            exp_dist_xi_yj,
            rtol=1e-5,
            atol=1e-5
        )

    all_pairs = cosine_fn.all_pairs(x, y)
    for i in range(n):
      for j in range(m):
        np.testing.assert_allclose(
            cosine_fn.pairwise(x[i], y[j]),
            all_pairs[i, j],
            rtol=1e-5,
            atol=1e-5,
        )

  @pytest.mark.fast
  class TestBuresBarycenter:

    def test_buresb(self, rng: jnp.ndarray):
      d = 5
      r = jnp.array([0.3206, 0.8825, 0.1113, 0.00052, 0.9454])
      Sigma1 = r * jnp.eye(d)
      s = jnp.array([0.3075, 0.8545, 0.1110, 0.0054, 0.9206])
      Sigma2 = s * jnp.eye(d)
      # initializing Bures cost function
      weights = jnp.array([.3, .7])
      bures = costs.Bures(d)
      # stacking parameter values
      xs = jnp.vstack((
          costs.mean_and_cov_to_x(jnp.zeros((d,)), Sigma1, d),
          costs.mean_and_cov_to_x(jnp.zeros((d,)), Sigma2, d)
      ))

      output = bures.barycenter(weights, xs, tolerance=1e-4, threshold=1e-6)
      _, sigma = costs.x_to_means_and_covs(output, 5)
      ground_truth = (weights[0] * jnp.sqrt(r) + weights[1] * jnp.sqrt(s)) ** 2
      np.testing.assert_allclose(
          ground_truth, jnp.diag(sigma), rtol=1e-5, atol=1e-5
      )
