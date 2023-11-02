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
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, univariate


class TestUnivariate:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    self.rng = rng
    self.n = 17
    self.m = 29
    self.rng, *rngs = jax.random.split(self.rng, 4)
    self.x = jax.random.uniform(rngs[0], [self.n])
    self.y = jax.random.uniform(rngs[1], [self.m])
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))

    #  adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @pytest.mark.fast.with_args(
      "cost_fn", [
          costs.SqEuclidean(),
          costs.PNormP(1.0),
          costs.PNormP(2.0),
          costs.PNormP(1.7)
      ]
  )
  def test_cdf_distance_and_sinkhorn(self, cost_fn: costs.CostFn):
    """The Univariate distance coincides with the sinkhorn solver"""
    univariate_solver = univariate.UnivariateSolver(
        method="wasserstein", cost_fn=cost_fn
    )
    distance = univariate_solver(self.x, self.y, self.a, self.b)

    geom = pointcloud.PointCloud(
        x=self.x[:, None],
        y=self.y[:, None],
        cost_fn=costs.PNormP(cost_fn),
        epsilon=5e-5
    )
    prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)
    sinkhorn_solver = sinkhorn.Sinkhorn(max_iterations=int(1e6))
    sinkhorn_soln = sinkhorn_solver(prob)

    np.testing.assert_allclose(
        sinkhorn_soln.primal_cost, distance, atol=0, rtol=1e-2
    )

  @pytest.mark.fast()
  def test_cdf_distance_and_scipy(self):
    """The OTT solver coincides with scipy solver"""

    # The `scipy` solver only has the solution for p=1.0 visible
    univariate_solver = univariate.UnivariateSolver(
        method="wasserstein", cost_fn=costs.PNormP(1.0)
    )
    ott_distance = univariate_solver(self.x, self.y, self.a, self.b)

    scipy_distance = sp.stats.wasserstein_distance(
        self.x, self.y, self.a, self.b
    )

    np.testing.assert_allclose(scipy_distance, ott_distance, atol=0, rtol=1e-2)
