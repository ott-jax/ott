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
import numpy as np

from ott.geometry import costs, distrib_costs, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import univariate
from ott.solvers.quadratic import lower_bound


@pytest.mark.fast()
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

  @pytest.mark.parametrize(
      "ground_cost",
      [costs.SqEuclidean(), costs.PNormP(1.3)]
  )
  def test_lb_pointcloud(self, ground_cost: costs.TICost):
    x, y = self.x, self.y

    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b
    )
    solve_fn = univariate.quantile_distance
    distrib_cost = distrib_costs.UnivariateWasserstein(
        solve_fn, ground_cost=ground_cost
    )

    solver = jax.jit(lower_bound.third_lower_bound)
    out = solver(prob, distrib_cost, epsilon=1e-3)

    assert jnp.isfinite(out.reg_ot_cost)

  @pytest.mark.parametrize(("ground_cost", "uniform", "eps"),
                           [(costs.SqEuclidean(), True, 1e-2),
                            (costs.PNormP(1.3), False, 1e-1)])
  def test_lb_different_solvers(
      self, ground_cost: costs.TICost, uniform: bool, eps: float
  ):
    x, y, a, b = self.x, self.y, self.a, self.b
    if uniform:
      k = min(self.n, self.m)
      x, y, a, b = x[:k], y[:k], a[:k], b[:k]

    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a, b=b)

    distrib_cost_unif = distrib_costs.UnivariateWasserstein(
        solve_fn=univariate.uniform_distance, ground_cost=ground_cost
    )
    distrib_cost_quant = distrib_costs.UnivariateWasserstein(
        solve_fn=univariate.quantile_distance, ground_cost=ground_cost
    )
    distrib_cost_nw = distrib_costs.UnivariateWasserstein(
        solve_fn=univariate.north_west_distance, ground_cost=ground_cost
    )

    solver = jax.jit(lower_bound.third_lower_bound)

    out_unif = solver(prob, distrib_cost_unif, epsilon=eps) if uniform else None
    out_quant = solver(prob, distrib_cost_quant, epsilon=eps)
    out_nw = solver(prob, distrib_cost_nw, epsilon=eps)

    np.testing.assert_allclose(
        out_quant.reg_ot_cost, out_nw.reg_ot_cost, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        out_quant.matrix, out_nw.matrix, rtol=1e-6, atol=1e-6
    )
    if out_unif is not None:
      np.testing.assert_allclose(
          out_quant.reg_ot_cost, out_unif.reg_ot_cost, rtol=1e-6, atol=1e-6
      )
      np.testing.assert_allclose(
          out_quant.matrix, out_unif.matrix, rtol=1e-6, atol=1e-6
      )
