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
from typing import Any

import pytest

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import scipy as sp

from ott.geometry import costs, pointcloud, regularizers
from ott.math import utils as mu
from ott.solvers import linear


def _proj(matrix: jnp.ndarray) -> jnp.ndarray:
  u, _, v_h = jnp.linalg.svd(matrix, full_matrices=False)
  return u.dot(v_h)


@pytest.mark.fast()
class TestCostFn:

  def test_cosine(self, rng: jax.Array):
    """Test the cosine cost function."""
    x = jnp.array([0, 0])
    y = jnp.array([0, 0])
    cost_fn = costs.Cosine()
    dist_x_y = cost_fn(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 0.0, rtol=1e-5, atol=1e-5)

    x = jnp.array([1.0, 0])
    y = jnp.array([1.0, 0])
    dist_x_y = cost_fn(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - 1.0, rtol=1e-5, atol=1e-5)

    x = jnp.array([1.0, 0])
    y = jnp.array([-1.0, 0])
    dist_x_y = cost_fn(x, y)
    np.testing.assert_allclose(dist_x_y, 1.0 - -1.0, rtol=1e-5, atol=1e-5)

    n, m, d = 10, 12, 7
    rngs = jax.random.split(rng, 2)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d))

    normalize = lambda v: v / jnp.sqrt(jnp.sum(v ** 2))
    for i in range(n):
      for j in range(m):
        exp_sim_xi_yj = jnp.sum(normalize(x[i]) * normalize(y[j]))
        exp_dist_xi_yj = 1.0 - exp_sim_xi_yj
        np.testing.assert_allclose(
            cost_fn(x[i], y[j]), exp_dist_xi_yj, rtol=1e-5, atol=1e-5
        )

    all_pairs = cost_fn.all_pairs(x, y)
    for i in range(n):
      for j in range(m):
        np.testing.assert_allclose(
            cost_fn(x[i], y[j]),
            all_pairs[i, j],
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.fast()
class TestBuresBarycenter:

  def test_bures(self, rng: jax.Array):
    d = 3
    r = jnp.array([1.2036, 0.2825, 0.013])
    Sigma1 = r * jnp.eye(d)
    s = jnp.array([3.3075, 0.8545, 0.1110])
    Sigma2 = s * jnp.eye(d)
    # initializing Bures cost function
    weights = jnp.array([0.3, 0.7])
    tolerance = 1e-6
    min_iterations = 2
    inner_iterations = 1
    max_iterations = 37
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

    _, sigma = costs.x_to_means_and_covs(cov, d)
    ground_truth = (weights[0] * jnp.sqrt(r) + weights[1] * jnp.sqrt(s)) ** 2
    np.testing.assert_allclose(
        ground_truth, jnp.diag(sigma), rtol=1e-4, atol=1e-4
    )
    # Check that outer loop ran for at least min_iterations
    np.testing.assert_array_less(
        0.0, diffs[min_iterations // inner_iterations - 1]
    )
    # Check converged
    np.testing.assert_array_less((diffs[diffs > -1])[-1], tolerance)
    # Check right output size of difference vectors
    np.testing.assert_equal(diffs.shape[0], max_iterations // inner_iterations)


class TestTICost:

  @pytest.mark.parametrize(
      "cost_fn", [
          costs.SqPNorm(1.05),
          costs.SqPNorm(2.4),
          costs.PNormP(1.1),
          costs.PNormP(1.3),
          costs.SqEuclidean()
      ]
  )
  def test_transport_map(self, rng: jax.Array, cost_fn: costs.TICost):
    n, d = 15, 5
    rng_x, rng_A = jax.random.split(rng)
    x = jax.random.normal(rng_x, (n, d))
    A = jax.random.normal(rng_A, (d, d * 2))
    A = A @ A.T

    transport_fn = cost_fn.transport_map(lambda z: -jnp.sum(z * (A.dot(z))))
    transport_fn = jax.jit(transport_fn)

    y = transport_fn(x)
    cost_matrix = cost_fn.all_pairs(x, y)

    row_ixs, col_ixs = sp.optimize.linear_sum_assignment(cost_matrix)
    np.testing.assert_array_equal(row_ixs, jnp.arange(n))
    np.testing.assert_array_equal(col_ixs, jnp.arange(n))

  @pytest.mark.parametrize(
      "cost_fn", [
          costs.SqEuclidean(),
          costs.PNormP(2),
          costs.RegTICost(regularizers.SqL2(), lam=0.0, rho=1.0)
      ]
  )
  def test_sqeucl_transport(
      self,
      rng: jax.Array,
      cost_fn: costs.TICost,
  ):
    sqeucl = costs.SqEuclidean()
    x = jax.random.normal(rng, (12, 7))
    f = mu.logsumexp

    h_f = cost_fn.h_transform(f)
    expected_fn = jax.jit(cost_fn.transport_map(f))
    if isinstance(cost_fn, costs.SqEuclidean):
      # multiply by `0.5`, because `SqEuclidean := |x|_2^2`
      actual_fn = jax.jit(jax.vmap(lambda x: x - 0.5 * jax.grad(h_f)(x)))
    else:
      actual_fn = jax.jit(jax.vmap(lambda x: x - jax.grad(h_f)(x)))

    np.testing.assert_allclose(
        expected_fn(x), actual_fn(x), rtol=1e-6, atol=1e-6
    )
    if not isinstance(cost_fn, costs.SqEuclidean):
      np.testing.assert_allclose(
          0.5 * sqeucl.all_pairs(x, x),
          cost_fn.all_pairs(x, x),
          rtol=1e-6,
          atol=1e-6
      )

  @pytest.mark.parametrize("cost_fn", [costs.SqEuclidean(), costs.PNormP(2)])
  @pytest.mark.parametrize("d", [5, 10])
  def test_h_transform_matches_unreg(
      self, rng: jax.Array, cost_fn: costs.TICost, d: int
  ):
    n = 13
    rngs = jax.random.split(rng, 2)
    u = jnp.abs(jax.random.uniform(rngs[0], (d,)))
    x = jax.random.normal(rngs[1], (n, d))

    gt_cost = costs.RegTICost(regularizers.SqL2(), lam=0.0)
    concave_gt = lambda z: -cost_fn.h(z) + jnp.dot(z, u)

    if isinstance(cost_fn, costs.PNormP):
      concave = concave_gt
    else:
      concave = lambda z: 0.5 * (-cost_fn.h(z) + jnp.dot(z, u))

    pred = jax.jit(jax.vmap(jax.grad(cost_fn.h_transform(concave, ridge=1e-6))))
    gt = jax.jit(jax.vmap(jax.grad(gt_cost.h_transform(concave_gt))))

    np.testing.assert_allclose(pred(x), gt(x), rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("cost_fn", [costs.SqEuclidean(), costs.PNormP(1.5)])
  def test_h_transform_solver(self, rng: jax.Array, cost_fn: costs.TICost):

    def gd_solver(
        fun, x: jnp.ndarray, x_init: jnp.ndarray, **kwargs: Any
    ) -> jnp.ndarray:
      solver = jaxopt.GradientDescent(fun=fun, **kwargs)
      return solver.run(x, x_init).params

    n, d = 21, 6
    rngs = jax.random.split(rng, 2)
    u = jnp.abs(jax.random.uniform(rngs[0], (d,)))
    x = jax.random.normal(rngs[1], (n, d))

    concave_fn = lambda z: -cost_fn.h(z) + jnp.dot(z, u)

    expected = jax.vmap(cost_fn.h_transform(concave_fn, solver=None))
    actual = jax.vmap(cost_fn.h_transform(concave_fn, solver=gd_solver))

    np.testing.assert_allclose(expected(x), actual(x), rtol=1e-4, atol=1e-4)


@pytest.mark.fast()
class TestRegTICost:

  @pytest.mark.parametrize("d", [5, 31, 77])
  @pytest.mark.parametrize(
      "cost_fn",
      [
          costs.RegTICost(regularizers.L1(), lam=8.0),
          costs.RegTICost(regularizers.SqL2(), lam=13.0),
          # lam must be 1.0
          costs.RegTICost(regularizers.STVS(8.0), lam=1.0),
          costs.RegTICost(regularizers.STVS(13.0), lam=1.0),
      ]
  )
  def test_reg_legendre(
      self,
      rng: jax.Array,
      cost_fn: costs.RegTICost,
      d: int,
  ):
    expected = jax.random.normal(rng, (d,))
    actual = jax.grad(cost_fn.h_legendre)(jax.grad(cost_fn.h)(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("lam", [1e-1, 0.5, 1.25])
  def test_h_transform_x_init(self, rng: jax.Array, lam: float):
    n, d = 11, 6
    rng_x, rng_y, rng_u = jax.random.split(rng, 3)
    y = jax.random.normal(rng_x, (d,)) + 1.0
    u = jnp.abs(jax.random.uniform(rng_u, (d,)))
    x_inits = jax.random.normal(rng_x, (n, d)) * jnp.linspace(
        -5.0, 5.0, num=n
    )[:, None]

    cost_fn = costs.RegTICost(regularizers.L1(), lam=lam)
    f = lambda z: -cost_fn.h(z) + jnp.dot(z, u)

    h_f = jax.vmap(cost_fn.h_transform(f), in_axes=[None, 0])
    res = h_f(y, x_inits)

    assert res.shape == (n,)
    np.testing.assert_allclose(
        jnp.abs(jnp.diff(res)), 0.0, rtol=1e-3, atol=1e-3
    )

  @pytest.mark.parametrize(
      "cost_fn",
      [
          costs.RegTICost(regularizers.L1(), lam=113.0),
          costs.RegTICost(regularizers.STVS(12.0), lam=1.0),
          costs.RegTICost(regularizers.SqKOverlap(3), lam=17.0)
      ],
  )
  def test_sparse_displacement(
      self,
      rng: jax.Array,
      cost_fn: costs.RegTICost,
  ):
    rng1, rng2 = jax.random.split(rng, 2)
    d = 17

    x = jax.random.normal(rng1, (25, d))
    y = jax.random.normal(rng2, (37, d))
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, relative_epsilon="mean")

    dp = linear.solve(geom).to_dual_potentials()

    for arr, fwd in zip([x, y], [True, False]):
      arr_t = dp.transport(arr, forward=fwd)
      assert np.mean(np.isclose(arr, arr_t)) > 0.6

  @pytest.mark.parametrize(
      "reg", [
          regularizers.L1(),
          regularizers.STVS(),
          regularizers.SqKOverlap(10),
      ]
  )
  def test_stronger_regularization_increases_sparsity(
      self,
      rng: jax.Array,
      reg: regularizers.ProximalOperator,
  ):
    d, rngs = 17, jax.random.split(rng, 4)
    x = jax.random.normal(rngs[0], (50, d))
    y = jax.random.normal(rngs[1], (71, d))
    xx = jax.random.normal(rngs[2], (25, d))
    yy = jax.random.normal(rngs[3], (35, d))

    sparsity = {False: [], True: []}
    for lam in [9, 89]:
      if isinstance(reg, regularizers.STVS):
        reg, lam = regularizers.STVS(lam), 1.0
      cost_fn = costs.RegTICost(reg, lam=lam)
      geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)

      dp = linear.solve(geom).to_dual_potentials()
      for arr, fwd in zip([xx, yy], [True, False]):
        arr_t = dp.transport(arr, forward=True)
        sparsity[fwd].append(np.sum(np.isclose(arr, arr_t)))

    for fwd in [False, True]:
      np.testing.assert_array_equal(np.diff(sparsity[fwd]) > 0.0, True)

  @pytest.mark.parametrize(
      "cost_fn", [
          costs.RegTICost(regularizers.L1(), lam=0.1),
          costs.RegTICost(regularizers.SqL2(), lam=3.3),
          costs.RegTICost(regularizers.STVS(1.0), lam=1.0),
          costs.RegTICost(regularizers.SqKOverlap(3), lam=1.05),
      ]
  )
  def test_reg_transport_fn(
      self,
      rng: jax.Array,
      cost_fn: costs.RegTICost,
  ):

    @jax.jit
    @functools.partial(jax.vmap, in_axes=0)
    def expected_fn(x: jnp.ndarray) -> jnp.ndarray:
      f_h = cost_fn.h_transform(f)
      return x - cost_fn.regularizer.prox(jax.grad(f_h)(x))

    x = jax.random.normal(rng, (11, 9))
    f = mu.logsumexp

    actual_fn = cost_fn.transport_map(f)
    actual_fn = jax.jit(actual_fn)

    np.testing.assert_array_equal(expected_fn(x), actual_fn(x))


@pytest.mark.fast()
class TestSoftDTW:

  @pytest.mark.parametrize("n", [7, 10])
  @pytest.mark.parametrize("m", [9, 10])
  @pytest.mark.parametrize("gamma", [1e-3, 5])
  def test_soft_dtw(self, rng: jax.Array, n: int, m: int, gamma: float):
    ts_metrics = pytest.importorskip("tslearn.metrics")
    rng1, rng2 = jax.random.split(rng, 2)
    t1 = jax.random.normal(rng1, (n,))
    t2 = jax.random.normal(rng2, (m,))

    expected = ts_metrics.soft_dtw(t1, t2, gamma=gamma)
    actual = costs.SoftDTW(gamma=gamma)(t1, t2)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize(("debiased", "jit"), [(False, True), (True, False)])
  def test_soft_dtw_debiased(
      self,
      rng: jax.Array,
      debiased: bool,
      jit: bool,
  ):
    ts_metrics = pytest.importorskip("tslearn.metrics")
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
      self, rng: jax.Array, debiased: bool, jit: bool, gamma: float
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
