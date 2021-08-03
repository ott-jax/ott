# coding=utf-8
# Copyright 2021 Google LLC.
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

# Lint as: python3
"""Tests Anderson acceleration for sinkhorn."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
from ott.core import sinkhorn
from ott.geometry import pointcloud


class SinkhornJacobianTest(jax.test_util.JaxTestCase):
  """Tests for Anderson acceleration."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.product(
      lse_mode=[True, False],
      tau_a=[1.0, .98],
      tau_b=[1.0, .985],
      shape=[(237, 153)],
      refresh_anderson_frequency=[1, 5]
      )
  def test_anderson(self, lse_mode, tau_a,
                    tau_b, shape, refresh_anderson_frequency):
    """Test efficiency of Anderson acceleration.

    Args:
      lse_mode: whether to run in lse (True) or kernel (false) mode.
      tau_a: unbalanced parameter w.r.t. 1st marginal
      tau_b: unbalanced parameter w.r.t. 1st marginal
      shape: shape of test problem
      refresh_anderson_frequency: how often to Anderson interpolation should be
        recomputed.
    """
    n, m = shape
    dim = 4
    rngs = jax.random.split(self.rng, 9)
    x = jax.random.uniform(rngs[0], (n, dim)) / dim
    y = jax.random.uniform(rngs[1], (m, dim)) / dim + .2
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    a = jax.ops.index_update(a, 0, 0)
    b = jax.ops.index_update(b, 3, 0)

    # Make weights roughly sum to 1 if unbalanced, normalize else.
    a = a / (0.5 * n) if tau_a < 1.0 else a / jnp.sum(a)
    b = b / (0.5 * m) if tau_b < 1.0 else b / jnp.sum(b)

    # Here epsilon must be small enough to valide gain in performance using
    # Anderson, but large enough when lse_mode=False to avoid underflow.
    epsilon = 1e-4 if lse_mode else 5e-3
    threshold = 1e-3
    iterations_anderson = []

    anderson_memory = [0, 5, 8]
    for anderson_acceleration in anderson_memory:
      out = sinkhorn.sinkhorn(
          pointcloud.PointCloud(x, y, epsilon=epsilon),
          a=a,
          b=b,
          tau_a=tau_a,
          tau_b=tau_b,
          lse_mode=lse_mode,
          threshold=threshold,
          anderson_acceleration=anderson_acceleration,
          refresh_anderson_frequency=refresh_anderson_frequency)
      errors = out.errors
      clean_errors = errors[errors > -1]
      # Check convergence
      self.assertGreater(threshold, clean_errors[-1])
      # Record number of inner_iterations needed to converge.
      iterations_anderson.append(jnp.size(clean_errors))

    # Check Anderson acceleration speeds up execution when compared to none.
    for i in range(1, len(anderson_memory)):
      self.assertGreater(iterations_anderson[0], iterations_anderson[i])

if __name__ == '__main__':
  absltest.main()
