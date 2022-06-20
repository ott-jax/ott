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

# Lint as: python3
"""Tests Online option for PointCloud geometry."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from ott.core import sinkhorn
from ott.core.sinkhorn import SinkhornOutput
from ott.geometry import pointcloud


class SinkhornOnlineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.dim = 3
    self.n = 4000
    self.m = 402
    self.rng, *rngs = jax.random.split(self.rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))
    #  adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @parameterized.parameters([True], [False])
  def test_euclidean_point_cloud(self, lse_mode):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-1
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=1, online=True)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode,
        implicit_differentiation=True
    ).errors
    err = errors[errors > -1][-1]
    self.assertGreater(threshold, err)

  @parameterized.parameters([1], [13], [402], [4000])
  def test_online_matches_offline_size(self, online: int):
    threshold, rtol, atol = 1e-1, 1e-6, 1e-6
    geom_offline = pointcloud.PointCloud(
        self.x, self.y, epsilon=1, online=False
    )
    geom_online = pointcloud.PointCloud(
        self.x, self.y, epsilon=1, online=online
    )

    sol_online = sinkhorn.sinkhorn(
        geom_online,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=True,
        implicit_differentiation=True
    )
    errors_online = sol_online.errors
    err_online = errors_online[errors_online > -1][-1]

    sol_offline = sinkhorn.sinkhorn(
        geom_offline,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=True,
        implicit_differentiation=True
    )

    self.assertGreater(threshold, err_online)
    np.testing.assert_allclose(
        sol_online.matrix, sol_offline.matrix, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        sol_online.a, sol_offline.a, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        sol_online.b, sol_offline.b, rtol=rtol, atol=atol
    )

  def test_online_sinkhorn_jit(self):
    threshold = 1e-1
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=1, online=512)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        jit=True,
        lse_mode=True,
        implicit_differentiation=True
    ).errors
    err = errors[errors > -1][-1]

    self.assertGreater(threshold, err)

  def test_online_external_jit(self):

    @partial(jax.jit, static_argnums=1)
    def callback(epsilon: float, online: int) -> SinkhornOutput:
      geom = pointcloud.PointCloud(
          self.x, self.y, epsilon=epsilon, online=online
      )
      return sinkhorn.sinkhorn(
          geom,
          a=self.a,
          b=self.b,
          threshold=threshold,
          jit=True,
          lse_mode=True,
          implicit_differentiation=True
      )

    threshold = 1e-1
    sol = callback(epsilon=1, online=42)
    errors = sol.errors
    err = errors[errors > -1][-1]

    self.assertGreater(threshold, err)


if __name__ == '__main__':
  absltest.main()
