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
import jax.random as jr
import numpy as np

from ott.geometry import costs, distrib_costs, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import univariate
from ott.solvers.quadratic import lower_bound
from tests import _utils


@pytest.fixture(scope="module")
def clouds(rng: jax.Array) -> _utils.QuadClouds:
  """Clouds with uniform marginals, drawn as this module always has."""
  n, m, d_x, d_y = 13, 15, 2, 3
  rngs = jr.split(rng, 4)
  return _utils.QuadClouds(
      x=jr.uniform(rngs[0], (n, d_x)),
      y=jr.uniform(rngs[1], (m, d_y)),
      a=jnp.ones(n) / n,
      b=jnp.ones(m) / m,
      cx=jr.uniform(rngs[2], (n, n)),
      cy=jr.uniform(rngs[3], (m, m)),
  )


@pytest.mark.fast()
class TestLowerBound:

  @pytest.mark.parametrize(
      "ground_cost",
      [costs.SqEuclidean(), costs.PNormP(1.3)]
  )
  def test_lb_pointcloud(
      self, clouds: _utils.QuadClouds, ground_cost: costs.TICost
  ):
    x, y = clouds.x, clouds.y

    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=clouds.a, b=clouds.b
    )
    solve_fn = univariate.quantile_solver
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
      self, clouds: _utils.QuadClouds, ground_cost: costs.TICost, uniform: bool,
      eps: float
  ):
    x, y, a, b = clouds.x, clouds.y, clouds.a, clouds.b
    if uniform:
      k = min(clouds.n, clouds.m)
      x, y, a, b = x[:k], y[:k], a[:k], b[:k]

    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a, b=b)

    distrib_cost_unif = distrib_costs.UnivariateWasserstein(
        solve_fn=univariate.uniform_solver, ground_cost=ground_cost
    )
    distrib_cost_quant = distrib_costs.UnivariateWasserstein(
        solve_fn=univariate.quantile_solver, ground_cost=ground_cost
    )
    distrib_cost_nw = distrib_costs.UnivariateWasserstein(
        solve_fn=univariate.north_west_solver, ground_cost=ground_cost
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
