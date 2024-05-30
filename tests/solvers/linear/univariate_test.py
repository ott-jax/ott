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

import pytest

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers import linear
from ott.solvers.linear import univariate


class TestUnivariate:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.Array):
    self.rng = jax.random.PRNGKey(4)
    self.n = 7
    self.m = 5
    self.d = 2
    self.rng, *rngs = jax.random.split(self.rng, 7)
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

  @pytest.mark.parametrize("cost_fn", [costs.SqEuclidean(), costs.PNormP(1.8)])
  def test_cdf_distance_and_sinkhorn(self, cost_fn: costs.CostFn):
    """The Univariate distance coincides with the sinkhorn solver"""
    # FIXME(michalk8)
    geom = pointcloud.PointCloud(self.x, self.y, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)
    out = univariate.quantile_distance(prob, return_transport=True)
    costs_1d, matrices_1d = out.ot_costs, out.transport_matrices
    mean_matrices_1d = out.mean_transport_matrix

    @jax.jit
    @functools.partial(jax.vmap, in_axes=[1, 1, None, None])
    def sliced_sinkhorn(x, y, a, b):
      geom = pointcloud.PointCloud(
          x[:, None], y[:, None], cost_fn=cost_fn, epsilon=0.0015
      )
      out = linear.solve(geom, a=self.a, b=self.b)
      return out.primal_cost, out.matrix, out.converged

    costs_sink, matrices_sink, converged = sliced_sinkhorn(
        self.x, self.y, self.a, self.b
    )
    assert jnp.all(converged)
    scale = 1 / (self.n * self.m)

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
  def test_cdf_distance_and_scipy(self):
    """The OTT solver coincides with scipy solver"""
    # FIXME
    x, y, a, b = self.x, self.y, self.a, self.b
    # The `scipy` solver only computes the solution for p=1.0 visible

    # non-uniform: vanilla or subsampling
    geom = pointcloud.PointCloud(x, y, cost_fn=costs.PNormP(1.0))
    prob = linear_problem.LinearProblem(geom=geom, a=a, b=b)
    ott_d = univariate.UnivariateSolver()(prob).ot_costs[0]
    scipy_d = sp.stats.wasserstein_distance(x[:, 0], y[:, 0], a, b)
    np.testing.assert_allclose(scipy_d, ott_d, atol=1e-2, rtol=1e-2)

    num_subsamples = 100
    ott_dss = univariate.UnivariateSolver(num_subsamples=num_subsamples
                                         )(prob).ot_costs[0]
    np.testing.assert_allclose(scipy_d, ott_dss, atol=1e2, rtol=1e-2)

    # uniform variants
    prob = linear_problem.LinearProblem(geom=geom)
    scipy_d2 = sp.stats.wasserstein_distance(x[:, 0], y[:, 0])

    ott_d = univariate.UnivariateSolver()(prob).ot_costs[0]
    ott_dq = univariate.UnivariateSolver(quantiles=8)(prob).ot_costs[0]
    np.testing.assert_allclose(scipy_d2, ott_d, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(scipy_d2, ott_dq, atol=1e-1, rtol=1e-1)

  @pytest.mark.fast()
  def test_univariate_grad(
      self,
      rng: jax.Array,
  ):
    # TODO: Once a `check_grad` function is implemented, replace the code
    # blocks before with `check_grad`'s.
    rngs = jax.random.split(rng, 4)
    eps, tol = 1e-4, 1e-3
    x, y = self.x[:, 1][:, None], self.y[:, 1][:, None]
    a, b = self.a, self.b

    def univ_dist(x, y, a, b):
      geom = pointcloud.PointCloud(x, y)
      prob = linear_problem.LinearProblem(geom=geom, a=a, b=b)
      return univariate.quantile_distance(prob).ot_costs.squeeze()

    grad_x, grad_y, grad_a, grad_b = jax.jit(jax.grad(univ_dist, (0, 1, 2, 3))
                                            )(x, y, a, b)

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
  def test_dual_vectors(self):
    # FIXME(michalk8)
    n = self.n
    x, a, y, b, z, c = self.x, self.a, self.y, self.b, self.z, self.c
    unif_n = jnp.ones((n,)) / n

    solve_fn = jax.jit(univariate.north_west_distance)

    for (target, weights_target) in ((y, b), (z, c)):
      for weights_source in (a, unif_n):
        geom = pointcloud.PointCloud(x, target, cost_fn=costs.SqEuclidean())
        prob = linear_problem.LinearProblem(
            geom=geom, a=weights_source, b=weights_target
        )
        out = solve_fn(prob)

        f, g = out.dual_a, out.dual_b
        dual_obj = jnp.sum(f * weights_source[None, :], axis=1)
        dual_obj += jnp.sum(g * weights_target[None, :], axis=1)
        # check objective returned with primal computation matches dual
        np.testing.assert_allclose(out.ot_costs, dual_obj, atol=1e-2, rtol=1e-2)
        # check dual variables are feasible on locations that matter (with
        # positive weights).
        mask = (weights_source > 0)[:, None] * (weights_target > 0)[None, :]
        min_val = jnp.min(
            mask[None] * (geom.cost_matrix - f[:, :, None] - g[:, None, :])
        )
        np.testing.assert_allclose(min_val, 0, atol=1e-5)
