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

import pytest

import jax
import jax.numpy as jnp

from ott.geometry import costs, distrib_costs, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import lower_bound


class TestLowerBound:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.Array):
    d_x = 2
    d_y = 3
    self.n, self.m = 13, 15
    rngs = jax.random.split(rng, 4)
    self.x = jax.random.uniform(rngs[0], (self.n, d_x))
    self.y = jax.random.uniform(rngs[1], (self.m, d_y))
    a = jnp.ones(self.n)
    b = jnp.ones(self.m)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(rngs[2], (self.n, self.n))
    self.cy = jax.random.uniform(rngs[3], (self.m, self.m))

  @pytest.mark.fast.with_args(
      "ground_cost",
      [costs.SqEuclidean(), costs.PNormP(1.5)],
      only_fast=0,
  )
  def test_lb_pointcloud(self, ground_cost: costs.TICost):
    x, y = self.x, self.y

    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b
    )
    distrib_cost = distrib_costs.UnivariateWasserstein(ground_cost=ground_cost)

    out = jax.jit(lower_bound.third_lower_bound
                 )(prob, distrib_cost, epsilon=1e-1)

    assert jnp.isfinite(out.reg_ot_cost)
