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

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import histogram_transport


class TestHistogramTransport:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    d_x = 2
    d_y = 3
    self.n, self.m = 6, 7
    rngs = jax.random.split(rng, 6)
    self.x = jax.random.uniform(rngs[0], (self.n, d_x))
    self.y = jax.random.uniform(rngs[1], (self.m, d_y))
    # Currently Histogram Transport only supports uniform distributions:
    a = jnp.ones(self.n)
    b = jnp.ones(self.m)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(rngs[4], (self.n, self.n))
    self.cy = jax.random.uniform(rngs[5], (self.m, self.m))

  @pytest.mark.fast.with_args(
      epsilon_sort=[0.0, 1e-2, 1.0],
      method=["subsample", "quantile"],
      cost_fn=[costs.SqEuclidean(), costs.PNormP(1.0)]
  )
  def test_ht_pointcloud(
      self, epsilon_sort: float, method: Literal["subsample", "quantile",
                                                 "equal"], cost_fn: costs.CostFn
  ):
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    tau_a, tau_b = 1.0, 1.0
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b, tau_a=tau_a, tau_b=tau_b
    )
    solver = histogram_transport.HistogramTransport(
        epsilon=1e-1,
        min_iterations=100,
        max_iterations=100,
        epsilon_sort=epsilon_sort,
        cost_fn=cost_fn,
        method=method,
        n_subsamples=min([self.x.shape[0], self.y.shape[0]]),
    )

    out = solver(prob)

    np.testing.assert_allclose(
        out.primal_cost, jnp.sum(out.geom.cost_matrix * out.matrix), rtol=1e-3
    )

    assert not jnp.isnan(out.reg_ot_cost)
