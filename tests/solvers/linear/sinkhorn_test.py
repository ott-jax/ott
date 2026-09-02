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
import io
import sys
from typing import Optional, Tuple

import pytest

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import scipy as sp

from ott import utils
from ott.geometry import costs, epsilon_scheduler, geometry, grid, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers import linear
from ott.solvers.linear import acceleration, sinkhorn
from tests import _utils


@pytest.fixture(scope="module")
def clouds() -> _utils.PointClouds:
  """Override with marginals containing zeros, to test their handling."""
  return _utils.random_clouds(jr.key(0), zero_a=(0,), zero_b=(3,))


class TestSinkhorn:

  def test_diag_sum(self, clouds: _utils.PointClouds):
    """Test Diag sum high when comparing PC (to LRC) with itself for low eps."""
    x = clouds.x
    epsilon = .01
    geom = pointcloud.PointCloud(x, epsilon=epsilon)
    geom_lr = geom.to_LRCGeometry()

    for g in (geom, geom_lr):
      out = linear.solve(g)
      np.testing.assert_array_less(.99, jnp.sum(out.diag))
      np.testing.assert_array_less(jnp.sum(out.diag), 1.0)

  def test_SqEucl_matches_hungarian(
      self, clouds: _utils.PointClouds, rng: jax.Array
  ):
    """Test that Sinkhorn matches Hungarian for low regularization."""
    x = clouds.x
    y = jr.uniform(rng, x.shape)
    epsilon = .01
    geom = pointcloud.PointCloud(
        clouds.x, y, cost_fn=costs.SqEuclidean(), epsilon=epsilon
    )
    geom2 = pointcloud.PointCloud(
        clouds.x, y, cost_fn=costs.NegDotProduct(), epsilon=epsilon
    )
    cost_matrix = geom.cost_matrix
    row_ixs, col_ixs = sp.optimize.linear_sum_assignment(cost_matrix)
    paired_ids = jnp.stack((row_ixs, col_ixs)).swapaxes(0, 1)
    unit_mass = jnp.ones((clouds.n,)) / clouds.n
    out_h = jesp.BCOO((unit_mass, paired_ids),
                      shape=(clouds.n, clouds.n)).todense().ravel()

    out1_m = linear.solve(geom).matrix.ravel()
    out2_m = linear.solve(geom2).matrix.ravel()
    cos_cost = costs.Cosine()
    np.testing.assert_array_less(cos_cost(out1_m, out2_m), 0.02)
    np.testing.assert_array_less(cos_cost(out1_m, out_h), 0.2)

  @pytest.mark.fast.with_args(tau_a=[1.0, 0.93], tau_b=[1.0, 0.91], only_fast=0)
  def test_lse_matches(
      self, clouds: _utils.PointClouds, tau_a: float, tau_b: float
  ):
    """Test that regardless of lse_mode, Sinkhorn returns same value."""
    geom = pointcloud.PointCloud(clouds.x, clouds.y)

    solve_fn = functools.partial(
        linear.solve,
        geom=geom,
        a=clouds.a,
        b=clouds.b,
        tau_a=tau_a,
        tau_b=tau_b
    )
    solve_fn = jax.jit(solve_fn, static_argnames=["lse_mode"])
    lse_out = solve_fn(lse_mode=True)
    ker_out = solve_fn(lse_mode=False)

    assert lse_out.converged
    assert ker_out.converged
    np.testing.assert_allclose(
        lse_out.ent_reg_cost, ker_out.ent_reg_cost, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        lse_out.reg_ot_cost, ker_out.reg_ot_cost, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        lse_out.dual_cost, ker_out.dual_cost, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        lse_out.primal_cost, ker_out.primal_cost, rtol=1e-5, atol=1e-5
    )

  @pytest.mark.fast.with_args(
      "lse_mode,mom_value,mom_start,inner_iterations,norm_error,cost_fn",
      [(True, 1.0, 29, 10, 1, costs.SqEuclidean()),
       (False, 1.0, 30, 10, 1, costs.SqPNorm(p=2.2)),
       (True, 1.0, 60, 1, 2, costs.Euclidean()),
       (True, 1.0, 12, 24, 4, costs.SqPNorm(p=1.0))],
      ids=["lse-Leh-mom", "scal-Leh-mom", "lse-Leh-1", "lse-Leh-24"],
      only_fast=[0, -1],
  )
  def test_euclidean_point_cloud(
      self,
      clouds: _utils.PointClouds,
      lse_mode: bool,
      mom_value: float,
      mom_start: int,
      inner_iterations: int,
      norm_error: int,
      cost_fn: costs.CostFn,
  ):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    momentum = acceleration.Momentum(start=mom_start, value=mom_value)

    geom = pointcloud.PointCloud(
        clouds.x, clouds.y, cost_fn=cost_fn, epsilon=0.1
    )
    out = linear.solve(
        geom,
        a=clouds.a,
        b=clouds.b,
        lse_mode=lse_mode,
        norm_error=norm_error,
        inner_iterations=inner_iterations,
        momentum=momentum
    )
    errors = out.errors
    err = errors[errors > -1][-1]
    np.testing.assert_array_less(err, threshold)
    np.testing.assert_allclose(out.transport_mass, 1.0, rtol=1e-4, atol=1e-4)

    other_geom = pointcloud.PointCloud(clouds.x, clouds.y + 0.3, epsilon=0.1)
    cost_other = out.transport_cost_at_geom(other_geom)
    np.testing.assert_array_equal(jnp.isnan(cost_other), False)

  def test_autoepsilon(self, clouds: _utils.PointClouds):
    """Check that with mean rescaling, dual potentials scale."""
    scale = 2.77
    tau_a = 0.99
    tau_b = 0.97
    geom_1 = pointcloud.PointCloud(clouds.x, clouds.y, relative_epsilon="mean")
    # not jitting
    f_1 = linear.solve(
        geom_1,
        a=clouds.a,
        b=clouds.b,
        tau_a=tau_a,
        tau_b=tau_b,
    ).f

    geom_2 = pointcloud.PointCloud(
        scale * clouds.x, scale * clouds.y, relative_epsilon="mean"
    )
    # jitting
    compute_f = jax.jit(linear.solve, static_argnames=["tau_a", "tau_b"])
    f_2 = compute_f(geom_2, clouds.a, clouds.b, tau_a=tau_a, tau_b=tau_b).f

    # Ensure epsilon and optimal f's are a scale^2 apart (^2 comes from ^2 cost)
    np.testing.assert_allclose(
        geom_1.epsilon * scale ** 2, geom_2.epsilon, rtol=1e-3, atol=1e-3
    )

    np.testing.assert_allclose(
        geom_1.epsilon_scheduler(2) * scale ** 2,
        geom_2.epsilon_scheduler(2),
        rtol=1e-3,
        atol=1e-3
    )

    np.testing.assert_allclose(f_1 * scale ** 2, f_2, rtol=1e-3, atol=1e-3)

  @pytest.mark.fast.with_args(
      lse_mode=[False, True],
      init=[5],
      decay=[0.9],
      tau_a=[1.0, 0.93],
      tau_b=[1.0, 0.91],
      only_fast=0
  )
  def test_autoepsilon_with_decay(
      self, clouds: _utils.PointClouds, lse_mode: bool, init: float,
      decay: float, tau_a: float, tau_b: float
  ):
    """Check that variations in init/decay work, and result in same solution."""
    geom = pointcloud.PointCloud(clouds.x, clouds.y)
    target = epsilon_scheduler.DEFAULT_EPSILON_SCALE * geom.std_cost_matrix
    epsilon = epsilon_scheduler.Epsilon(target, init=init, decay=decay)
    geom_eps = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=epsilon)
    run_fn = jax.jit(
        linear.solve,
        static_argnames=[
            "tau_a", "tau_b", "lse_mode", "threshold", "recenter_potentials"
        ]
    )

    out_1 = run_fn(
        geom_eps,
        clouds.a,
        clouds.b,
        tau_a=tau_a,
        tau_b=tau_b,
        lse_mode=lse_mode,
        threshold=1e-5,
        recenter_potentials=True
    )
    out_2 = run_fn(
        geom,
        clouds.a,
        clouds.b,
        tau_a=tau_a,
        tau_b=tau_b,
        lse_mode=lse_mode,
        threshold=1e-5,
        recenter_potentials=True
    )
    # recenter the problem, since in that case solution is only
    # valid up to additive constant in the balanced case

    assert out_1.converged
    assert out_2.converged
    f_1, f_2 = out_1.f, out_2.f
    np.testing.assert_allclose(f_1, f_2, rtol=1e-4, atol=1e-4)

  @pytest.mark.fast()
  def test_euclidean_point_cloud_min_iter(self, clouds: _utils.PointClouds):
    """Testing the min_iterations parameter."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=0.1)
    errors = linear.solve(
        geom,
        a=clouds.a,
        b=clouds.b,
        threshold=threshold,
        min_iterations=34,
    ).errors
    err = errors[jnp.logical_and(errors > -1, jnp.isfinite(errors))][-1]
    assert threshold > err
    assert errors[0] == jnp.inf
    assert errors[1] == jnp.inf
    assert errors[2] == jnp.inf
    assert errors[3] > 0

  @pytest.mark.fast()
  def test_euclidean_point_cloud_scan_loop(self, clouds: _utils.PointClouds):
    """Testing the scan loop behavior."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=0.1)
    out = linear.solve(
        geom,
        a=clouds.a,
        b=clouds.b,
        threshold=threshold,
        min_iterations=50,
        max_iterations=50
    )
    # Test converged flag is True despite running in scan mode
    assert out.converged
    # Test last error recomputed at the final iteration, and below threshold.
    assert out.errors[-1] > 0
    assert out.errors[-1] < threshold

  def test_geom_vs_point_cloud(self, clouds: _utils.PointClouds):
    """Two point clouds vs. simple cost_matrix execution of Sinkhorn."""
    geom_1 = pointcloud.PointCloud(clouds.x, clouds.y)
    geom_2 = geometry.Geometry(geom_1.cost_matrix)

    f_1 = linear.solve(geom_1, a=clouds.a, b=clouds.b).f
    f_2 = linear.solve(geom_2, a=clouds.a, b=clouds.b).f
    # re-centering to remove ambiguity on equality up to additive constant.
    f_1 -= jnp.mean(f_1[jnp.isfinite(f_1)])
    f_2 -= jnp.mean(f_2[jnp.isfinite(f_2)])

    np.testing.assert_allclose(f_1, f_2, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_online_euclidean_point_cloud(
      self, clouds: _utils.PointClouds, lse_mode: bool
  ):
    """Testing the online way to handle geometry."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=0.1, batch_size=5)
    errors = linear.solve(
        geom, a=clouds.a, b=clouds.b, threshold=threshold, lse_mode=lse_mode
    ).errors
    err = errors[errors > -1][-1]
    assert threshold > err

  @pytest.mark.fast.with_args("lse_mode", [False, True], only_fast=0)
  def test_online_vs_batch_euclidean_point_cloud(
      self, clouds: _utils.PointClouds, lse_mode: bool
  ):
    """Comparing online vs batch geometry."""
    eps = 0.1
    online_geom = pointcloud.PointCloud(
        clouds.x, clouds.y, epsilon=eps, batch_size=7
    )
    online_geom_euc = pointcloud.PointCloud(
        clouds.x,
        clouds.y,
        cost_fn=costs.SqEuclidean(),
        epsilon=eps,
        batch_size=10
    )

    batch_geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=eps)
    batch_geom_euc = pointcloud.PointCloud(
        clouds.x, clouds.y, cost_fn=costs.SqEuclidean(), epsilon=eps
    )

    out_online = linear.solve(
        online_geom, a=clouds.a, b=clouds.b, lse_mode=lse_mode
    )
    out_batch = linear.solve(
        batch_geom, a=clouds.a, b=clouds.b, lse_mode=lse_mode
    )
    out_online_euc = linear.solve(
        online_geom_euc, a=clouds.a, b=clouds.b, lse_mode=lse_mode
    )
    out_batch_euc = linear.solve(
        batch_geom_euc, a=clouds.a, b=clouds.b, lse_mode=lse_mode
    )

    # Checks regularized transport costs match.
    np.testing.assert_allclose(
        out_online.reg_ot_cost, out_batch.reg_ot_cost, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_online.kl_reg_cost, out_batch.kl_reg_cost, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_online.ent_reg_cost, out_batch.ent_reg_cost, rtol=1e-5
    )
    # Check values are bigger than 0 for KL regularized OT.
    np.testing.assert_array_less(0.0, out_online.kl_reg_cost)
    # check regularized transport matrices match
    np.testing.assert_allclose(
        online_geom.transport_from_potentials(out_online.f, out_online.g),
        batch_geom.transport_from_potentials(out_batch.f, out_batch.g),
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        online_geom_euc.transport_from_potentials(
            out_online_euc.f, out_online_euc.g
        ),
        batch_geom_euc.transport_from_potentials(
            out_batch_euc.f, out_batch_euc.g
        ),
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        batch_geom.transport_from_potentials(out_batch.f, out_batch.g),
        batch_geom_euc.transport_from_potentials(
            out_batch_euc.f, out_batch_euc.g
        ),
        rtol=1e-5,
        atol=1e-5
    )

  @pytest.mark.parametrize(
      "from_scalings", [False, True], ids=["potentials", "scalings"]
  )
  def test_apply_transport_geometry(self, rng: jax.Array, from_scalings: bool):
    """Applying transport matrix P on vector without instantiating P."""
    rng_data, rng_a, rng_b = jr.split(rng, 3)
    data = _utils.random_clouds(rng_data, n=20, m=23, dim=3)
    vec_a = jr.normal(rng_a, (data.n,))
    vec_b = jr.normal(rng_b, (7, data.m))

    applied_a, applied_b = [], []
    # test with lse_mode and online = True / False
    for lse_mode in [True, False]:
      for batch_size in [2, None]:
        geom = data.geom(batch_size=batch_size, epsilon=0.2)
        out = linear.solve(geom, data.a, data.b, lse_mode=lse_mode)

        if from_scalings:
          u = geom.scaling_from_potential(out.f)
          v = geom.scaling_from_potential(out.g)
          apply_fn = functools.partial(geom.apply_transport_from_scalings, u, v)
          transport = geom.transport_from_scalings(u, v)
        else:
          apply_fn = functools.partial(
              geom.apply_transport_from_potentials, out.f, out.g
          )
          transport = geom.transport_from_potentials(out.f, out.g)

        applied_a.append(apply_fn(vec_a, axis=0))
        applied_b.append(apply_fn(vec_b, axis=1))

        np.testing.assert_allclose(
            applied_a[-1], jnp.dot(transport.T, vec_a).T, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            applied_b[-1], jnp.dot(transport, vec_b.T).T, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_array_equal(jnp.isnan(applied_a[-1]), False)

    for i in range(1, len(applied_a)):
      np.testing.assert_allclose(
          applied_b[i], applied_b[0], rtol=1e-3, atol=1e-3
      )
      np.testing.assert_allclose(
          applied_a[i], applied_a[0], rtol=1e-3, atol=1e-3
      )

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_restart(self, clouds: _utils.PointClouds, lse_mode: bool):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-2
    geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=0.05)
    out = linear.solve(
        geom,
        a=clouds.a,
        b=clouds.b,
        threshold=threshold,
        lse_mode=lse_mode,
        inner_iterations=1
    )
    errors = out.errors
    err = errors[errors > -1][-1]
    assert threshold > err

    # recover solution from previous and ensure faster convergence.
    if lse_mode:
      init_dual_a, init_dual_b = out.f, out.g
    else:
      init_dual_a, init_dual_b = (
          geom.scaling_from_potential(out.f),
          geom.scaling_from_potential(out.g)
      )

    if lse_mode:
      default_a = jnp.zeros_like(init_dual_a)
      default_b = jnp.zeros_like(init_dual_b)
    else:
      default_a = jnp.ones_like(init_dual_a)
      default_b = jnp.ones_like(init_dual_b)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(default_a, init_dual_a)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(default_b, init_dual_b)

    prob = linear_problem.LinearProblem(geom, a=clouds.a, b=clouds.b)
    solver = sinkhorn.Sinkhorn(
        threshold=threshold, lse_mode=lse_mode, inner_iterations=1
    )
    out_restarted = solver(prob, (init_dual_a, init_dual_b))

    errors_restarted = out_restarted.errors
    err_restarted = errors_restarted[errors_restarted > -1][-1]
    assert threshold > err_restarted

    num_iter_restarted = jnp.sum(errors_restarted > -1)
    # check we can only improve on error
    assert err > err_restarted
    # check first error in restart does at least as well as previous best
    assert err > errors_restarted[0]
    # check only one iteration suffices when restarting with same data.
    assert num_iter_restarted == 1

  @pytest.mark.cpu()
  @pytest.mark.limit_memory("80 MB")
  @pytest.mark.fast()
  def test_sinkhorn_online_memory_jit(self, rng: jax.Array):
    # test that full matrix is not materialized.
    # Only storing it would result in 250 * 8000 = 2e6 entries = 16Mb,
    # which would overflow due to other overheads
    batch_size = 10
    rngs = jr.split(rng, 4)
    n, m = 250, 8000
    x = jr.uniform(rngs[0], (n, 2))
    y = jr.uniform(rngs[1], (m, 2))
    geom = pointcloud.PointCloud(x, y, batch_size=batch_size, epsilon=1)
    problem = linear_problem.LinearProblem(geom)
    solver = jax.jit(sinkhorn.Sinkhorn())

    out = solver(problem)
    assert out.converged
    assert out.primal_cost > 0.0

  @pytest.mark.fast.with_args(cost_fn=[None, costs.SqPNorm(1.6)])
  def test_primal_cost_grid(
      self, rng: jax.Array, cost_fn: Optional[costs.CostFn]
  ):
    """Test computation of primal / costs for Grids."""
    rng_a, rng_b = jr.split(rng)
    ns = [6, 7, 11]
    xs = [jr.normal(jr.key(i), (n,)) for i, n in enumerate(ns)]
    geom = grid.Grid(xs, cost_fns=[cost_fn], epsilon=0.1)
    a = jr.uniform(rng_a, (geom.shape[0],))
    b = jr.uniform(rng_b, (geom.shape[0],))
    a, b = a / jnp.sum(a), b / jnp.sum(b)
    lin_prob = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn()
    out = solver(lin_prob)

    # Recover full cost matrix by applying it to columns of identity matrix.
    cost_matrix = geom.apply_cost(jnp.eye(geom.shape[0]))
    # Recover full transport by applying it to columns of identity matrix.
    transport_matrix = out.apply(jnp.eye(geom.shape[0]))
    cost = jnp.sum(transport_matrix * cost_matrix)
    assert cost > 0.0
    assert out.primal_cost > 0.0
    np.testing.assert_allclose(cost, out.primal_cost, rtol=1e-5, atol=1e-5)
    assert jnp.isfinite(out.dual_cost)
    assert out.primal_cost - out.dual_cost > 0.0

  @pytest.mark.fast.with_args(
      cost_fn=[costs.SqEuclidean(), costs.SqPNorm(1.6)],
  )
  def test_primal_cost_pointcloud(
      self, clouds: _utils.PointClouds, cost_fn: costs.CostFn
  ):
    """Test computation of primal and dual costs for PointCouds."""
    geom = pointcloud.PointCloud(
        clouds.x, clouds.y, cost_fn=cost_fn, epsilon=1e-3
    )

    lin_prob = linear_problem.LinearProblem(geom, a=clouds.a, b=clouds.b)
    solver = sinkhorn.Sinkhorn()
    out = solver(lin_prob)
    assert out.primal_cost > 0.0
    assert jnp.isfinite(out.dual_cost)
    # Check duality gap
    assert out.primal_cost - out.dual_cost > 0.0
    # Check that it is small
    np.testing.assert_allclose((out.primal_cost - out.dual_cost) /
                               out.primal_cost,
                               0,
                               atol=1e-1)
    cost = jnp.sum(out.matrix * out.geom.cost_matrix)
    np.testing.assert_allclose(cost, out.primal_cost, rtol=1e-5, atol=1e-5)

  @pytest.mark.fast.with_args(
      cost_fn=[costs.SqEuclidean(), costs.NegDotProduct()]
  )
  def test_entropy(self, clouds: _utils.PointClouds, cost_fn: costs.CostFn):
    """Test computation of entropy of solution."""
    geom = pointcloud.PointCloud(
        clouds.x, clouds.y, cost_fn=cost_fn, epsilon=1e-3
    )

    lin_prob = linear_problem.LinearProblem(geom, a=clouds.a, b=clouds.b)
    solver = sinkhorn.Sinkhorn(threshold=1e-4)
    out = solver(lin_prob)
    ent_transport = jnp.sum(jsp.special.entr(out.matrix))
    np.testing.assert_allclose(ent_transport, out.entropy, atol=1e-2, rtol=1e-2)

  @pytest.mark.fast.with_args(
      cost_fn=[costs.SqEuclidean(), costs.NegDotProduct()]
  )
  def test_normalized_entropy(
      self, clouds: _utils.PointClouds, cost_fn: costs.CostFn
  ):
    """Test computation of normalized entropy of solution."""
    geom = pointcloud.PointCloud(clouds.x, clouds.x, cost_fn=cost_fn)
    out = linear.solve(geom)
    ent_transport = jnp.sum(jsp.special.entr(out.matrix))
    norm_ent = ent_transport / jnp.log(geom.shape[0]) - 1.0
    np.testing.assert_allclose(
        norm_ent, out.normalized_entropy, atol=1e-3, rtol=1e-3
    )

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_f_potential_is_zero_centered(
      self, clouds: _utils.PointClouds, lse_mode: bool
  ):
    prob = clouds.problem()
    assert prob.is_balanced
    solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, recenter_potentials=True)

    f = solver(prob).f
    f_mean = jnp.mean(jnp.where(jnp.isfinite(f), f, 0.0))

    np.testing.assert_allclose(f_mean, 0.0, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize(("use_tqdm", "custom_buffer"), [(False, False),
                                                           (False, True),
                                                           (True, False)])
  def test_progress_fn(
      self, clouds: _utils.PointClouds, capsys: pytest.CaptureFixture[str],
      use_tqdm: bool, custom_buffer: bool
  ):
    geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=1e-1)

    if use_tqdm:
      tqdm = pytest.importorskip("tqdm")
      pbar = tqdm.tqdm()
      progress_fn = utils.tqdm_progress_fn(pbar, fmt="err: {error}")
    else:
      stream = io.StringIO() if custom_buffer else sys.stdout
      progress_fn = utils.default_progress_fn(
          fmt="foo {iter}/{max_iter}", stream=stream
      )

    _ = linear.solve(
        geom,
        progress_fn=progress_fn,
        min_iterations=0,
        inner_iterations=7,
        max_iterations=14,
    )

    if use_tqdm:
      pbar.close()
      assert pbar.postfix.startswith("err: ")
    elif custom_buffer:
      val = stream.getvalue()
      assert val.startswith("foo 7/14"), val
      captured = capsys.readouterr()
      assert captured.out == "", captured.out
    else:
      captured = capsys.readouterr()
      assert captured.out.startswith("foo 7/14"), captured

  @pytest.mark.fast()
  def test_custom_progress_fn(self, clouds: _utils.PointClouds):
    """Check that the callback function is actually called."""
    num_iterations = 30

    def progress_fn(
        status: Tuple[np.ndarray, np.ndarray, np.ndarray,
                      sinkhorn.SinkhornState],
    ) -> None:
      # Convert arguments.
      iteration, inner_iterations, total_iter, state = status
      iteration = int(iteration)
      inner_iterations = int(inner_iterations)
      total_iter = int(total_iter)
      errors = np.array(state.errors).ravel()

      # Avoid reporting error on each iteration,
      # because errors are only computed every `inner_iterations`.
      if (iteration + 1) % inner_iterations == 0:
        error_idx = max((iteration + 1) // inner_iterations - 1, 0)
        error = errors[error_idx]

        traced_values["iters"].append(iteration)
        traced_values["error"].append(error)
        traced_values["total"].append(total_iter)

    traced_values = {"iters": [], "error": [], "total": []}

    lin_prob = clouds.problem(epsilon=1e-3)

    inner_iterations = 10

    _ = sinkhorn.Sinkhorn(
        progress_fn=progress_fn,
        max_iterations=num_iterations,
        inner_iterations=inner_iterations
    )(
        lin_prob
    )

    # check that the function is called on the 10th iteration (iter #9), the
    # 20th iteration (iter #19) etc.
    assert traced_values["iters"] == [
        10 * v - 1 for v in range(1, num_iterations // inner_iterations + 1)
    ]

    # check that error decreases
    np.testing.assert_array_equal(np.diff(traced_values["error"]) < 0, True)

    # check that max iterations is provided each time: [30, 30]
    assert traced_values["total"] == [
        num_iterations
        for _ in range(1, num_iterations // inner_iterations + 1)
    ]

  @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
  def test_sinkhorn_dtype(self, clouds: _utils.PointClouds, dtype: jnp.ndarray):
    x = clouds.x.astype(dtype)
    y = clouds.y.astype(dtype)
    geom = pointcloud.PointCloud(x, y, epsilon=jnp.array(1e-1, dtype=dtype))

    out = linear.solve(geom, threshold=1e-2)

    assert out.f.dtype == dtype
    assert out.g.dtype == dtype
    assert out.errors.dtype == dtype
    assert out.matrix.dtype == dtype
    assert out.reg_ot_cost.dtype == dtype
    assert out.ent_reg_cost.dtype == dtype
    assert out.kl_reg_cost.dtype == dtype
    assert out.primal_cost.dtype == dtype
    assert out.dual_cost.dtype == dtype
    assert out.transport_mass.dtype == dtype
