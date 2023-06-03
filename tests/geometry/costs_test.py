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
from typing import Type

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, pointcloud
from ott.solvers.linear import sinkhorn

try:
  from tslearn import metrics as ts_metrics
except ImportError:
  ts_metrics = None


@pytest.mark.fast()
class TestCostFn:

  def test_cosine(self, rng: jax.random.PRNGKeyArray):
    """Test the cosine cost function."""
    x = jnp.array([0, 0])
    y = jnp.array([0, 0])
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 0.0, rtol=1e-5, atol=1e-5)

    x = jnp.array([1.0, 0])
    y = jnp.array([1.0, 0])
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 1.0, rtol=1e-5, atol=1e-5)

    x = jnp.array([1.0, 0])
    y = jnp.array([-1.0, 0])
    dist_x_y = costs.Cosine().pairwise(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - -1.0, rtol=1e-5, atol=1e-5)

    n, m, d = 10, 12, 7
    rngs = jax.random.split(rng, 2)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d))

    cosine_fn = costs.Cosine()
    normalize = lambda v: v / jnp.sqrt(jnp.sum(v ** 2))
    for i in range(n):
      for j in range(m):
        exp_sim_xi_yj = jnp.sum(normalize(x[i]) * normalize(y[j]))
        exp_dist_xi_yj = 1.0 - exp_sim_xi_yj
        np.testing.assert_allclose(
            cosine_fn.pairwise(x[i], y[j]),
            exp_dist_xi_yj,
            rtol=1e-5,
            atol=1e-5
        )

    all_pairs = cosine_fn.all_pairs_pairwise(x, y)
    for i in range(n):
      for j in range(m):
        np.testing.assert_allclose(
            cosine_fn.pairwise(x[i], y[j]),
            all_pairs[i, j],
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.fast()
class TestBuresBarycenter:

  def test_bures(self, rng: jax.random.PRNGKeyArray):
    d = 5
    r = jnp.array([1.2036, 0.2825, 0.013, 0.00052, 0.1454])
    Sigma1 = r * jnp.eye(d)
    s = jnp.array([3.3075, 0.8545, 0.1110, 0.54, 0.9206])
    Sigma2 = s * jnp.eye(d)
    # initializing Bures cost function
    weights = jnp.array([.3, .7])
    tolerance = 1e-6
    min_iterations = 13
    inner_iterations = 1
    max_iterations = 123
    bures = costs.Bures(d, sqrtm_kw={"max_iterations": 134, "threshold": 1e-8})
    # stacking parameter values
    xs = jnp.vstack((
        costs.mean_and_cov_to_x(jnp.zeros((d,)), Sigma1, d),
        costs.mean_and_cov_to_x(jnp.zeros((d,)), Sigma2, d)
    ))

    cov, diffs = bures.barycenter(
        weights,
        xs,
        tolerance=tolerance,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        inner_iterations=inner_iterations
    )

    _, sigma = costs.x_to_means_and_covs(cov, 5)
    ground_truth = (weights[0] * jnp.sqrt(r) + weights[1] * jnp.sqrt(s)) ** 2
    np.testing.assert_allclose(
        ground_truth, jnp.diag(sigma), rtol=1e-4, atol=1e-4
    )
    # Check that outer loop ran for at leat min_iterations
    np.testing.assert_array_less(
        0, diffs[min_iterations // inner_iterations - 1]
    )
    # Check converged
    np.testing.assert_array_less((diffs[diffs > -1])[-1], tolerance)
    # Check right output size of difference vectors
    np.testing.assert_equal(diffs.shape[0], max_iterations // inner_iterations)


@pytest.mark.fast()
class TestRegTICost:

  @pytest.mark.parametrize(
      "cost_fn",
      [
          costs.ElasticL1(gamma=5),
          costs.ElasticL1(gamma=0.0),
          costs.ElasticSTVS(gamma=2.2),
          costs.ElasticSTVS(gamma=10),
      ],
      ids=[
          "elasticnet",
          "elasticnet-gam0",
          "stvs-gam2.2",
          "stvs-gam10",
      ],
  )
  def test_reg_cost_legendre(
      self, rng: jax.random.PRNGKeyArray, cost_fn: costs.RegTICost
  ):
    for d in [5, 10, 50, 100, 1000]:
      rng, rng1 = jax.random.split(rng)
      expected = jax.random.normal(rng1, (d,))
      actual = jax.grad(cost_fn.h_legendre)(jax.grad(cost_fn.h)(expected))
      np.testing.assert_allclose(
          actual, expected, rtol=1e-5, atol=1e-5, err_msg=f"d={d}"
      )

  @pytest.mark.parametrize("k", [1, 2, 7, 10])
  @pytest.mark.parametrize("d", [10, 50, 100])
  def test_elastic_sq_k_overlap(
      self, rng: jax.random.PRNGKeyArray, k: int, d: int
  ):
    expected = jax.random.normal(rng, (d,))

    cost_fn = costs.ElasticSqKOverlap(k=k, gamma=1e-2)
    actual = jax.grad(cost_fn.h_legendre)(jax.grad(cost_fn.h)(expected))
    # should hold for small gamma
    assert np.corrcoef(expected, actual)[0][1] > 0.97

  @pytest.mark.parametrize(
      "cost_fn", [
          costs.ElasticL1(gamma=100),
          costs.ElasticSTVS(gamma=10),
          costs.ElasticSqKOverlap(k=3, gamma=20)
      ]
  )
  def test_sparse_displacement(
      self, rng: jax.random.PRNGKeyArray, cost_fn: costs.RegTICost
  ):
    frac_sparse = 0.8
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (50, 30))
    y = jax.random.normal(rng2, (71, 30))
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)

    dp = sinkhorn.solve(geom).to_dual_potentials()

    for arr, fwd in zip([x, y], [True, False]):
      arr_t = dp.transport(arr, forward=fwd)
      assert np.sum(np.isclose(arr, arr_t)) / arr.size > frac_sparse

  @pytest.mark.parametrize("cost_clazz", [costs.ElasticL1, costs.ElasticSTVS])
  def test_stronger_regularization_increases_sparsity(
      self, rng: jax.random.PRNGKeyArray, cost_clazz: Type[costs.RegTICost]
  ):
    d, rngs = 30, jax.random.split(rng, 4)
    x = jax.random.normal(rngs[0], (50, d))
    y = jax.random.normal(rngs[1], (71, d))
    xx = jax.random.normal(rngs[2], (25, d))
    yy = jax.random.normal(rngs[3], (35, d))

    sparsity = {False: [], True: []}
    for gamma in [9, 10, 100]:
      cost_fn = cost_clazz(gamma=gamma)
      geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)

      dp = sinkhorn.solve(geom).to_dual_potentials()
      for arr, fwd in zip([xx, yy], [True, False]):
        arr_t = dp.transport(arr, forward=True)
        sparsity[fwd].append(np.sum(np.isclose(arr, arr_t)))

    for fwd in [False, True]:
      np.testing.assert_array_equal(np.diff(sparsity[fwd]) > 0.0, True)


@pytest.mark.skipif(ts_metrics is None, reason="Not supported for Python 3.11")
@pytest.mark.fast()
class TestSoftDTW:

  @pytest.mark.parametrize("n", [11, 16])
  @pytest.mark.parametrize("m", [13, 16])
  @pytest.mark.parametrize("gamma", [1e-3, 1.0, 5])
  def test_soft_dtw(
      self, rng: jax.random.PRNGKeyArray, n: int, m: int, gamma: float
  ):
    rng1, rng2 = jax.random.split(rng, 2)
    t1 = jax.random.normal(rng1, (n,))
    t2 = jax.random.normal(rng2, (m,))

    expected = ts_metrics.soft_dtw(t1, t2, gamma=gamma)
    actual = costs.SoftDTW(gamma=gamma)(t1, t2)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize(("debiased", "jit"), [(False, True), (True, False)])
  def test_soft_dtw_debiased(
      self,
      rng: jax.random.PRNGKeyArray,
      debiased: bool,
      jit: bool,
  ):
    gamma = 1e-1
    rng1, rng2 = jax.random.split(rng, 2)
    t1 = jax.random.normal(rng1, (16,))
    t2 = jax.random.normal(rng2, (32,))

    expected = ts_metrics.soft_dtw(t1, t2, gamma=gamma)
    if debiased:
      expected -= 0.5 * (
          ts_metrics.soft_dtw(t1, t1, gamma=gamma) +
          ts_metrics.soft_dtw(t2, t2, gamma=gamma)
      )
    cost_fn = costs.SoftDTW(gamma=gamma, debiased=debiased)
    actual = jax.jit(cost_fn)(t1, t2) if jit else cost_fn(t1, t2)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    if debiased:
      assert expected >= 0
      np.testing.assert_allclose(cost_fn(t1, t1), 0.0, rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(cost_fn(t2, t2), 0.0, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize(("debiased", "jit"), [(False, False), (True, True)])
  @pytest.mark.parametrize("gamma", [1e-2, 1])
  def test_soft_dtw_grad(
      self, rng: jax.random.PRNGKeyArray, debiased: bool, jit: bool,
      gamma: float
  ):
    rngs = jax.random.split(rng, 4)
    eps, tol = 1e-3, 1e-5
    t1 = jax.random.normal(rngs[0], (9,))
    t2 = jax.random.normal(rngs[1], (16,))

    v_t1 = jax.random.normal(rngs[2], shape=t1.shape)
    v_t1 = (v_t1 / jnp.linalg.norm(v_t1, axis=-1, keepdims=True)) * eps
    v_t2 = jax.random.normal(rngs[3], shape=t2.shape) * eps
    v_t2 = (v_t2 / jnp.linalg.norm(v_t2, axis=-1, keepdims=True)) * eps

    cost_fn = costs.SoftDTW(gamma=gamma, debiased=debiased)
    grad_cost = jax.grad(cost_fn, argnums=[0, 1])
    grad_t1, grad_t2 = jax.jit(grad_cost)(t1, t2) if jit else grad_cost(t1, t2)

    expected = cost_fn(t1 + v_t1, t2) - cost_fn(t1 - v_t1, t2)
    actual = 2 * jnp.vdot(v_t1, grad_t1)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    expected = cost_fn(t1, t2 + v_t2) - cost_fn(t1, t2 - v_t2)
    actual = 2 * jnp.vdot(v_t2, grad_t2)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
