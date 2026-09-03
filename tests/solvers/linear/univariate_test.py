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
from collections.abc import Callable

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.stats as st

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers import linear
from ott.solvers.linear import univariate
from tests import _utils


@pytest.fixture(scope="module")
def clouds() -> _utils.PointClouds:
  """Override with marginals containing zeros, to test their handling."""
  return _utils.random_clouds(
      jr.key(0), n=7, m=5, dim=3, offset=0.0, zero_a=(0,), zero_b=(3,)
  )


@pytest.fixture(scope="module")
def clouds_xz(clouds: _utils.PointClouds) -> _utils.PointClouds:
  """:func:`clouds`' source paired with a second cloud of the same size."""
  rng_y, rng_b = jr.split(jr.fold_in(jr.key(0), 1), 2)
  return _utils.PointClouds(
      x=clouds.x,
      y=jr.uniform(rng_y, (clouds.n, clouds.dim)),
      a=clouds.a,
      b=_utils.random_weights(rng_b, clouds.n, offset=0.0, zero_at=(1,)),
  )


class TestUnivariate:

  @pytest.mark.parametrize(
      "cost_fn", [
          costs.Euclidean(),
          costs.SqEuclidean(),
          costs.SqPNorm(1.5),
          costs.PNormP(2.2)
      ]
  )
  def test_solvers_match(self, rng: jax.Array, cost_fn: costs.CostFn):
    rng1, rng2 = jr.split(rng, 2)
    n, d = 12, 5

    x = jr.normal(rng1, (n, d))
    y = jr.normal(rng2, (n, d)) + 1.0
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom)

    unif_costs = univariate.uniform_solver(prob).ot_costs
    quant_costs = univariate.quantile_solver(prob).ot_costs
    nw_costs = univariate.north_west_solver(prob).ot_costs

    np.testing.assert_allclose(unif_costs, quant_costs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(unif_costs, nw_costs, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("cost_fn", [costs.SqEuclidean(), costs.PNormP(1.8)])
  def test_cdf_distance_and_sinkhorn(
      self, clouds: _utils.PointClouds, cost_fn: costs.TICost
  ):

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

    prob = clouds.problem(cost_fn=cost_fn)
    out = univariate.quantile_solver(prob, return_transport=True)
    costs_1d, matrices_1d = out.ot_costs, out.transport_matrices.todense()
    mean_matrices_1d = out.mean_transport_matrix.todense()

    costs_sink, matrices_sink, converged = sliced_sinkhorn(
        clouds.x, clouds.y, clouds.a, clouds.b
    )
    np.testing.assert_array_equal(converged, True)
    scale = 1.0 / (clouds.n * clouds.m)

    np.testing.assert_allclose(costs_1d, costs_sink, atol=scale, rtol=1e-1)

    np.testing.assert_allclose(
        jnp.mean(matrices_1d, axis=0).sum(1), clouds.a, atol=1e-3
    )
    np.testing.assert_allclose(
        jnp.mean(matrices_1d, axis=0).sum(0), clouds.b, atol=1e-3
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
  def test_cdf_distance_and_scipy(
      self, clouds: _utils.PointClouds, rng: jax.Array
  ):
    x, y, a, b = clouds.x, clouds.y, clouds.a, clouds.b
    xx = jr.normal(rng, x.shape)
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
      self, clouds: _utils.PointClouds, rng: jax.Array,
      univariate_fn: Callable[[linear_problem.LinearProblem],
                              univariate.UnivariateOutput]
  ):

    def univ_dist(
        x: jnp.ndarray, y: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray
    ) -> float:
      geom = pointcloud.PointCloud(x[:, None], y[:, None])
      prob = linear_problem.LinearProblem(geom, a=a, b=b)
      return univariate_fn(prob).ot_costs.squeeze()

    rngs = jr.split(rng, 4)
    eps, tol = 1e-4, 1e-3
    x, y = clouds.x[:, 1], clouds.y[:, 1]
    a, b = clouds.a, clouds.b

    grad_univ_dist = jax.jit(jax.grad(univ_dist, argnums=(0, 1, 2, 3)))
    if univariate_fn is univariate.uniform_solver:
      a, b, y = None, None, x
    grad_x, grad_y, grad_a, grad_b = grad_univ_dist(x, y, a, b)

    # Checking geometric grads:
    v_x = jr.normal(rngs[0], shape=x.shape)
    v_x = (v_x / jnp.linalg.norm(v_x, axis=-1, keepdims=True)) * eps
    expected = univ_dist(x + v_x, y, a, b) - univ_dist(x - v_x, y, a, b)
    actual = 2.0 * jnp.vdot(v_x, grad_x)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    v_y = jr.normal(rngs[1], shape=y.shape)
    v_y = (v_y / jnp.linalg.norm(v_y, axis=-1, keepdims=True)) * eps
    expected = univ_dist(x, y + v_y, a, b) - univ_dist(x, y - v_y, a, b)
    actual = 2.0 * jnp.vdot(v_y, grad_y)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    # Checking probability grads:
    if univariate_fn is not univariate.uniform_solver:
      v_a = jr.normal(rngs[2], shape=a.shape)
      v_a -= jnp.mean(v_a, axis=-1, keepdims=True)
      v_a = (v_a / jnp.linalg.norm(v_a, axis=-1, keepdims=True)) * eps
      expected = univ_dist(x, y, a + v_a, b) - univ_dist(x, y, a - v_a, b)
      actual = 2.0 * jnp.vdot(v_a, grad_a)
      np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

      v_b = jr.normal(rngs[3], shape=b.shape)
      v_b -= jnp.mean(v_b, axis=-1, keepdims=True)
      v_b = (v_b / jnp.linalg.norm(v_b, axis=-1, keepdims=True)) * eps
      expected = univ_dist(x, y, a, b + v_b) - univ_dist(x, y, a, b - v_b)
      actual = 2.0 * jnp.vdot(v_b, grad_b)
      np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

  @pytest.mark.fast()
  @pytest.mark.parametrize("cost_fn", [costs.SqEuclidean(), costs.SqPNorm(1.1)])
  @pytest.mark.parametrize("uniform_source", [True, False])
  @pytest.mark.parametrize("pairing", ["clouds", "clouds_xz"])
  def test_dual_vectors(
      self, request: pytest.FixtureRequest, cost_fn: costs.TICost,
      uniform_source: bool, pairing: str
  ):
    data = request.getfixturevalue(pairing)
    x, y, b = data.x, data.y, data.b
    a = jnp.ones(data.n) / data.n if uniform_source else data.a

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
