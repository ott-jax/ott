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
from ott.geometry import geometry, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn
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
    a = jax.random.uniform(rngs[2], (self.n,)) + 1e-1
    b = jax.random.uniform(rngs[3], (self.m,)) + 1e-1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(rngs[4], (self.n, self.n))
    self.cy = jax.random.uniform(rngs[5], (self.m, self.m))
    self.tau_a = 0.8
    self.tau_b = 0.9

  @pytest.mark.fast()
  def test_ht_pointcloud(self):
    """Test basic computations point clouds."""
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    tau_a, tau_b = (1.0, 1.0)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b, tau_a=tau_a, tau_b=tau_b
    )
    solver = histogram_transport.HistogramTransport(
        softness=-1,
        p=2.0,
        epsilon=1e-1,
        min_iterations=100,
        max_iterations=100
    )

    out = solver(prob)

    np.testing.assert_allclose(
        out.primal_cost, jnp.sum(out.geom.cost_matrix * out.matrix), rtol=1e-3
    )

    assert not jnp.isnan(out.reg_ht_cost)

  @pytest.mark.parametrize(
      "is_cost",
      [True, False],
  )
  def test_gradient_ht_geometry(
      self,
      is_cost: bool,
  ):
    """Test gradient w.r.t. the geometries."""

    def reg_ht(
        x: jnp.ndarray,
        y: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        implicit: bool,
    ) -> float:
      if is_cost:
        geom_x = geometry.Geometry(cost_matrix=x)
        geom_y = geometry.Geometry(cost_matrix=y)
      else:
        geom_x = pointcloud.PointCloud(x)
        geom_y = pointcloud.PointCloud(y)
      prob = quadratic_problem.QuadraticProblem(
          geom_x,
          geom_y,
      )

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None

      solver = histogram_transport.HistogramTransport(
          epsilon=1.0, max_iterations=100
      )

      lin_solver = sinkhorn.Sinkhorn(
          lse_mode=True, max_iterations=1000, implicit_diff=implicit_diff
      )

      solver.linear_ot_solver = lin_solver

      return solver(prob).reg_ht_cost

    grad_matrices = [None, None]
    x, y = (self.cx, self.cy) if is_cost else (self.x, self.y)
    reg_ht_grad = jax.grad(reg_ht, argnums=(0, 1))

    for i, implicit in enumerate([True, False]):
      grad_matrices[i] = reg_ht_grad(x, y, self.a, self.b, implicit)
      assert not jnp.any(jnp.isnan(grad_matrices[i][0]))
      assert not jnp.any(jnp.isnan(grad_matrices[i][1]))

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-02, atol=1e-02
    )
