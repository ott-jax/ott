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
import functools
from typing import Callable

import pytest

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as st

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers import linear
from ott.solvers.linear import univariate


class TestUnivariate:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.Array):
    rngs = jax.random.split(rng, 6)
    self.n = 7
    self.m = 5
    self.d = 3
    self.x = jax.random.uniform(rngs[0], (self.n, self.d))
    self.y = jax.random.uniform(rngs[1], (self.m, self.d))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))
    # introduce family of points of the same size as x.
    self.z = jax.random.uniform(rngs[4], (self.n, self.d))
    c = jax.random.uniform(rngs[5], (self.n,))

    #  adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    c = c.at[1].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.c = c / jnp.sum(c)

  @pytest.mark.parametrize(
      "cost_fn", [
          costs.Euclidean(),
          costs.SqEuclidean(),
          costs.SqPNorm(1.5),
          costs.PNormP(2.2)
      ]
  )
  def test_solvers_match(self, rng: jax.Array, cost_fn: costs.CostFn):
    rng1, rng2 = jax.random.split(rng, 2)
    n, d = 12, 5

    x = jax.random.normal(rng1, (n, d))
    y = jax.random.normal(rng2, (n, d)) + 1.0
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom)

    unif_costs = univariate.uniform_solver(prob).ot_costs
    quant_costs = univariate.quantile_solver(prob).ot_costs
    nw_costs = univariate.north_west_solver(prob).ot_costs

    np.testing.assert_allclose(unif_costs, quant_costs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(unif_costs, nw_costs, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("cost_fn", [costs.SqEuclidean(), costs.PNormP(1.8)])
  def test_cdf_distance_and_sinkhorn(self, cost_fn: costs.TICost):

    @jax.jit
    @functools.partial(jax.vmap, in_axes=[1, 1, None, None])
    def sliced_sinkhorn(
        x: jnp.ndarray, y: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray
    ):
      geom = pointcloud.PointCloud(
          x[:, None], y[:, None], cost_fn=cost_fn, epsilon=1e-4
      )
      out = linear.solve(geom, a=a, b=b, max_iterations=50_000)
      return out.primal_cost, out.matrix, out.converged

    geom = pointcloud.PointCloud(self.x, self.y, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)
    out = univariate.quantile_solver(prob, return_transport=True)
    costs_1d, matrices_1d = out.ot_costs, out.transport_matrices.todense()
    mean_matrices_1d = out.mean_transport_matrix.todense()

    costs_sink, matrices_sink, converged = sliced_sinkhorn(
        self.x, self.y, self.a, self.b
    )
    np.testing.assert_array_equal(converged, True)
    scale = 1.0 / (self.n * self.m)

    np.testing.assert_allclose(costs_1d, costs_sink, atol=scale, rtol=1e-1)

    np.testing.assert_allclose(
        jnp.mean(matrices_1d, axis=0).sum(1), self.a, atol=1e-3
    )
    np.testing.assert_allclose(
        jnp.mean(matrices_1d, axis=0).sum(0), self.b, atol=1e-3
    )

    np.testing.assert_allclose(
        matrices_sink, matrices_1d, atol=0.5 * scale, rtol=1e-1
    )
    np.testing.assert_allclose(
        jnp.mean(matrices_sink, axis=0),
        mean_matrices_1d,
        atol=0.5 * scale,
        rtol=1e-1
    )

  @pytest.mark.fast()
  def test_cdf_distance_and_scipy(self, rng: jax.Array):
    x, y, a, b = self.x, self.y, self.a, self.b
    xx = jax.random.normal(rng, x.shape)
    # The `scipy` solver only computes the solution for p=1.0 visible
    geom = pointcloud.PointCloud(x, y, cost_fn=costs.PNormP(1.0))

    # non-uniform variant
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    ott_d = univariate.quantile_solver(prob).ot_costs[0]
    scipy_d = st.wasserstein_distance(x[:, 0], y[:, 0], a, b)

    np.testing.assert_allclose(scipy_d, ott_d, atol=1e-2, rtol=1e-2)

    # uniform variants
    prob = linear_problem.LinearProblem(geom)
    ott_d = univariate.quantile_solver(prob).ot_costs[0]
    scipy_d2 = st.wasserstein_distance(x[:, 0], y[:, 0])

    np.testing.assert_allclose(scipy_d2, ott_d, atol=1e-2, rtol=1e-2)

    geom = pointcloud.PointCloud(x, xx, cost_fn=costs.Euclidean())
    prob = linear_problem.LinearProblem(geom)
    ott_d = univariate.uniform_solver(prob).ot_costs[0]
    scipy_d2 = st.wasserstein_distance(x[:, 0], xx[:, 0])

    np.testing.assert_allclose(scipy_d2, ott_d, atol=1e-2, rtol=1e-2)

  @pytest.mark.fast()
  @pytest.mark.parametrize(
      "univariate_fn", [
          univariate.uniform_solver, univariate.quantile_solver,
          univariate.north_west_solver
      ],
      ids=["uniform", "quant", "north-west"]
  )
  def test_univariate_grad(
      self, rng: jax.Array,
      univariate_fn: Callable[[linear_problem.LinearProblem],
                              univariate.UnivariateOutput]
  ):

    def univ_dist(
        x: jnp.ndarray, y: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray
    ) -> float:
      geom = pointcloud.PointCloud(x[:, None], y[:, None])
      prob = linear_problem.LinearProblem(geom, a=a, b=b)
      return univariate_fn(prob).ot_costs.squeeze()

    rngs = jax.random.split(rng, 4)
    eps, tol = 1e-4, 1e-3
    x, y = self.x[:, 1], self.y[:, 1]
    a, b = self.a, self.b

    grad_univ_dist = jax.jit(jax.grad(univ_dist, argnums=(0, 1, 2, 3)))
    if univariate_fn is univariate.uniform_solver:
      a, b, y = None, None, x
    grad_x, grad_y, grad_a, grad_b = grad_univ_dist(x, y, a, b)

    # Checking geometric grads:
    v_x = jax.random.normal(rngs[0], shape=x.shape)
    v_x = (v_x / jnp.linalg.norm(v_x, axis=-1, keepdims=True)) * eps
    expected = univ_dist(x + v_x, y, a, b) - univ_dist(x - v_x, y, a, b)
    actual = 2.0 * jnp.vdot(v_x, grad_x)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    v_y = jax.random.normal(rngs[1], shape=y.shape)
    v_y = (v_y / jnp.linalg.norm(v_y, axis=-1, keepdims=True)) * eps
    expected = univ_dist(x, y + v_y, a, b) - univ_dist(x, y - v_y, a, b)
    actual = 2.0 * jnp.vdot(v_y, grad_y)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    # Checking probability grads:
    if univariate_fn is not univariate.uniform_solver:
      v_a = jax.random.normal(rngs[2], shape=a.shape)
      v_a -= jnp.mean(v_a, axis=-1, keepdims=True)
      v_a = (v_a / jnp.linalg.norm(v_a, axis=-1, keepdims=True)) * eps
      expected = univ_dist(x, y, a + v_a, b) - univ_dist(x, y, a - v_a, b)
      actual = 2.0 * jnp.vdot(v_a, grad_a)
      np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

      v_b = jax.random.normal(rngs[3], shape=b.shape)
      v_b -= jnp.mean(v_b, axis=-1, keepdims=True)
      v_b = (v_b / jnp.linalg.norm(v_b, axis=-1, keepdims=True)) * eps
      expected = univ_dist(x, y, a, b + v_b) - univ_dist(x, y, a, b - v_b)
      actual = 2.0 * jnp.vdot(v_b, grad_b)
      np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

  @pytest.mark.fast()
  @pytest.mark.parametrize("cost_fn", [costs.SqEuclidean(), costs.SqPNorm(1.1)])
  @pytest.mark.parametrize("weight_source", ["uniform", "a"])
  @pytest.mark.parametrize(("target", "weight_target"), [("y", "b"),
                                                         ("z", "c")])
  def test_dual_vectors(
      self, cost_fn: costs.TICost, weight_source: str, target: str,
      weight_target: str
  ):
    x = self.x
    a = (jnp.ones(self.n) / self.n
        ) if weight_source == "uniform" else getattr(self, weight_source)
    y = getattr(self, target)
    b = getattr(self, weight_target)

    solve_fn = jax.jit(univariate.north_west_solver)

    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    out = solve_fn(prob)
    f, g = out.dual_a, out.dual_b

    np.testing.assert_allclose(
        out.ot_costs, out.dual_costs, atol=1e-2, rtol=1e-2
    )

    # check dual variables are feasible on locations that matter
    # (with positive weights).
    mask = (a > 0)[:, None] * (b > 0)[None, :]
    min_val = jnp.min(
        mask[None] * (geom.cost_matrix - f[:, :, None] - g[:, None, :])
    )
    np.testing.assert_allclose(min_val, 0, atol=1e-5)
