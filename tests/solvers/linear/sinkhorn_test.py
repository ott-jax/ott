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
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, epsilon_scheduler, geometry, grid, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import acceleration, sinkhorn


class TestSinkhorn:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    self.rng = rng
    self.dim = 4
    self.n = 17
    self.m = 29
    self.rng, *rngs = jax.random.split(self.rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))

    #  adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

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

    geom = pointcloud.PointCloud(self.x, self.y, cost_fn=cost_fn, epsilon=0.1)
    out = sinkhorn.solve(
        geom,
        a=self.a,
        b=self.b,
        lse_mode=lse_mode,
        norm_error=norm_error,
        inner_iterations=inner_iterations,
        momentum=momentum
    )
    errors = out.errors
    err = errors[errors > -1][-1]
    assert threshold > err

    other_geom = pointcloud.PointCloud(self.x, self.y + 0.3, epsilon=0.1)
    cost_other = out.transport_cost_at_geom(other_geom)
    assert not jnp.isnan(cost_other)

  def test_autoepsilon(self):
    """Check that with auto-epsilon, dual potentials scale."""
    scale = 2.77
    # First geom specifies explicitly relative_epsilon to be True. This is not
    # needed in principle, but introduced here to test logic.
    geom_1 = pointcloud.PointCloud(self.x, self.y, relative_epsilon=True)
    # not jitting
    f_1 = sinkhorn.solve(
        geom_1,
        a=self.a,
        b=self.b,
        tau_a=.99,
        tau_b=.97,
    ).f

    # Second geom does not provide whether epsilon is relative.
    geom_2 = pointcloud.PointCloud(scale * self.x, scale * self.y)
    # jitting
    compute_f = jax.jit(sinkhorn.solve, static_argnames=["tau_a", "tau_b"])
    f_2 = compute_f(geom_2, self.a, self.b, tau_a=0.99, tau_b=0.97).f

    # Ensure epsilon and optimal f's are a scale^2 apart (^2 comes from ^2 cost)
    np.testing.assert_allclose(
        geom_1.epsilon * scale ** 2, geom_2.epsilon, rtol=1e-3, atol=1e-3
    )

    np.testing.assert_allclose(
        geom_1._epsilon.at(2) * scale ** 2,
        geom_2._epsilon.at(2),
        rtol=1e-3,
        atol=1e-3
    )

    np.testing.assert_allclose(f_1 * scale ** 2, f_2, rtol=1e-3, atol=1e-3)

  @pytest.mark.fast.with_args(
      lse_mode=[False, True],
      init=[5],
      decay=[.9],
      tau_a=[1.0, .93],
      tau_b=[1.0, .91],
      only_fast=0
  )
  def test_autoepsilon_with_decay(
      self, lse_mode: bool, init: float, decay: float, tau_a: float,
      tau_b: float
  ):
    """Check that variations in init/decay work, and result in same solution."""
    epsilon = epsilon_scheduler.Epsilon(init=init, decay=decay)
    geom1 = pointcloud.PointCloud(self.x, self.y, epsilon=epsilon)
    geom2 = pointcloud.PointCloud(self.x, self.y)
    run_fn = jax.jit(
        sinkhorn.solve,
        static_argnames=[
            "tau_a", "tau_b", "lse_mode", "threshold", "recenter_potentials"
        ]
    )

    out_1 = run_fn(
        geom1,
        self.a,
        self.b,
        tau_a=tau_a,
        tau_b=tau_b,
        lse_mode=lse_mode,
        threshold=1e-5,
        recenter_potentials=True
    )
    out_2 = run_fn(
        geom2,
        self.a,
        self.b,
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
  def test_euclidean_point_cloud_min_iter(self):
    """Testing the min_iterations parameter."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1)
    errors = sinkhorn.solve(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        min_iterations=34,
    ).errors
    err = errors[jnp.logical_and(errors > -1, jnp.isfinite(errors))][-1]
    assert threshold > err
    assert errors[0] == jnp.inf
    assert errors[1] == jnp.inf
    assert errors[2] == jnp.inf
    assert errors[3] > 0

  def test_geom_vs_point_cloud(self):
    """Two point clouds vs. simple cost_matrix execution of Sinkhorn."""
    geom_1 = pointcloud.PointCloud(self.x, self.y)
    geom_2 = geometry.Geometry(geom_1.cost_matrix)

    f_1 = sinkhorn.solve(geom_1, a=self.a, b=self.b).f
    f_2 = sinkhorn.solve(geom_2, a=self.a, b=self.b).f
    # re-centering to remove ambiguity on equality up to additive constant.
    f_1 -= jnp.mean(f_1[jnp.isfinite(f_1)])
    f_2 -= jnp.mean(f_2[jnp.isfinite(f_2)])

    np.testing.assert_allclose(f_1, f_2, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_online_euclidean_point_cloud(self, lse_mode: bool):
    """Testing the online way to handle geometry."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1, batch_size=5)
    errors = sinkhorn.solve(
        geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode
    ).errors
    err = errors[errors > -1][-1]
    assert threshold > err

  @pytest.mark.fast.with_args("lse_mode", [False, True], only_fast=0)
  def test_online_vs_batch_euclidean_point_cloud(self, lse_mode: bool):
    """Comparing online vs batch geometry."""
    threshold = 1e-3
    eps = 0.1
    online_geom = pointcloud.PointCloud(
        self.x, self.y, epsilon=eps, batch_size=7
    )
    online_geom_euc = pointcloud.PointCloud(
        self.x, self.y, cost_fn=costs.SqEuclidean(), epsilon=eps, batch_size=10
    )

    batch_geom = pointcloud.PointCloud(self.x, self.y, epsilon=eps)
    batch_geom_euc = pointcloud.PointCloud(
        self.x, self.y, cost_fn=costs.SqEuclidean(), epsilon=eps
    )

    out_online = sinkhorn.solve(
        online_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode
    )
    out_batch = sinkhorn.solve(
        batch_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode
    )
    out_online_euc = sinkhorn.solve(
        online_geom_euc,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode
    )
    out_batch_euc = sinkhorn.solve(
        batch_geom_euc,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode
    )

    # Checks regularized transport costs match.
    np.testing.assert_allclose(
        out_online.reg_ot_cost, out_batch.reg_ot_cost, rtol=1e-6
    )
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

  def test_apply_transport_geometry_from_potentials(self):
    """Applying transport matrix P on vector without instantiating P."""
    n, m, d = 160, 230, 6
    rngs = jax.random.split(self.rng, 6)
    x = jax.random.uniform(rngs[0], (n, d))
    y = jax.random.uniform(rngs[1], (m, d))
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    transport_t_vec_a = [None, None, None, None]
    transport_vec_b = [None, None, None, None]

    batch_b = 8

    vec_a = jax.random.normal(rngs[4], (n,))
    vec_b = jax.random.normal(rngs[5], (batch_b, m))

    # test with lse_mode and online = True / False
    for j, lse_mode in enumerate([True, False]):
      for i, batch_size in enumerate([16, None]):
        geom = pointcloud.PointCloud(x, y, batch_size=batch_size, epsilon=0.2)
        out = sinkhorn.solve(geom, a, b, lse_mode=lse_mode)

        transport_t_vec_a[i + 2 * j] = geom.apply_transport_from_potentials(
            out.f, out.g, vec_a, axis=0
        )
        transport_vec_b[i + 2 * j] = geom.apply_transport_from_potentials(
            out.f, out.g, vec_b, axis=1
        )

        transport = geom.transport_from_potentials(out.f, out.g)

        np.testing.assert_allclose(
            transport_t_vec_a[i + 2 * j],
            jnp.dot(transport.T, vec_a).T,
            rtol=1e-3,
            atol=1e-3
        )
        np.testing.assert_allclose(
            transport_vec_b[i + 2 * j],
            jnp.dot(transport, vec_b.T).T,
            rtol=1e-3,
            atol=1e-3
        )

    for i in range(4):
      np.testing.assert_allclose(
          transport_vec_b[i], transport_vec_b[0], rtol=1e-3, atol=1e-3
      )
      np.testing.assert_allclose(
          transport_t_vec_a[i], transport_t_vec_a[0], rtol=1e-3, atol=1e-3
      )

  def test_apply_transport_geometry_from_scalings(self):
    """Applying transport matrix P on vector without instantiating P."""
    n, m, d = 160, 230, 6
    rngs = jax.random.split(self.rng, 6)
    x = jax.random.uniform(rngs[0], (n, d))
    y = jax.random.uniform(rngs[1], (m, d))
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    transport_t_vec_a = [None, None, None, None]
    transport_vec_b = [None, None, None, None]

    batch_b = 8

    vec_a = jax.random.normal(rngs[4], (n,))
    vec_b = jax.random.normal(rngs[5], (batch_b, m))

    # test with lse_mode and online = True / False
    for j, lse_mode in enumerate([True, False]):
      for i, batch_size in enumerate([64, None]):
        geom = pointcloud.PointCloud(x, y, batch_size=batch_size, epsilon=0.2)
        out = sinkhorn.solve(geom, a, b, lse_mode=lse_mode)

        u = geom.scaling_from_potential(out.f)
        v = geom.scaling_from_potential(out.g)

        transport_t_vec_a[i + 2 * j] = geom.apply_transport_from_scalings(
            u, v, vec_a, axis=0
        )
        transport_vec_b[i + 2 * j] = geom.apply_transport_from_scalings(
            u, v, vec_b, axis=1
        )

        transport = geom.transport_from_scalings(u, v)

        np.testing.assert_allclose(
            transport_t_vec_a[i + 2 * j],
            jnp.dot(transport.T, vec_a).T,
            rtol=1e-3,
            atol=1e-3
        )
        np.testing.assert_allclose(
            transport_vec_b[i + 2 * j],
            jnp.dot(transport, vec_b.T).T,
            rtol=1e-3,
            atol=1e-3
        )
        np.testing.assert_array_equal(
            jnp.isnan(transport_t_vec_a[i + 2 * j]), False
        )

    for i in range(4):
      np.testing.assert_allclose(
          transport_vec_b[i], transport_vec_b[0], rtol=1e-3, atol=1e-3
      )
      np.testing.assert_allclose(
          transport_t_vec_a[i], transport_t_vec_a[0], rtol=1e-3, atol=1e-3
      )

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_restart(self, lse_mode: bool):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-4
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.01)
    out = sinkhorn.solve(
        geom,
        a=self.a,
        b=self.b,
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

    prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)
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
  @pytest.mark.limit_memory("110 MB")
  @pytest.mark.fast.with_args("batch_size", [500, 1000], only_fast=0)
  def test_sinkhorn_online_memory_jit(self, batch_size: int):
    # offline: Total memory allocated: 240.1MiB
    # online (500): Total memory allocated: 33.4MiB; GPU: 203.4MiB
    # online (1000): Total memory allocated: 45.6MiB
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    n, m = 5000, 4000
    x = jax.random.uniform(rngs[0], (n, 2))
    y = jax.random.uniform(rngs[1], (m, 2))
    geom = pointcloud.PointCloud(x, y, batch_size=batch_size, epsilon=1)
    problem = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(jit=False)
    solver = jax.jit(solver)

    out = solver(problem)
    assert out.converged
    assert out.primal_cost > 0.0

  @pytest.mark.fast.with_args(
      cost_fn=[None, costs.SqPNorm(1.6)],
  )
  def test_primal_cost_grid(self, cost_fn: Optional[costs.CostFn]):
    """Test computation of primal / costs for Grids."""
    ns = [6, 7, 11]
    xs = [
        jax.random.normal(jax.random.PRNGKey(i), (n,))
        for i, n in enumerate(ns)
    ]
    geom = grid.Grid(xs, cost_fns=[cost_fn], epsilon=0.1)
    a = jax.random.uniform(jax.random.PRNGKey(0), (geom.shape[0],))
    b = jax.random.uniform(jax.random.PRNGKey(1), (geom.shape[0],))
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
  def test_primal_cost_pointcloud(self, cost_fn):
    """Test computation of primal and dual costs for PointCouds."""
    geom = pointcloud.PointCloud(self.x, self.y, cost_fn=cost_fn, epsilon=1e-3)

    lin_prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)
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

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_f_potential_is_zero_centered(self, lse_mode: bool):
    geom = pointcloud.PointCloud(self.x, self.y)
    prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)
    assert prob.is_balanced
    solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, recenter_potentials=True)

    f = solver(prob).f
    f_mean = jnp.mean(jnp.where(jnp.isfinite(f), f, 0.))

    np.testing.assert_allclose(f_mean, 0., rtol=1e-6, atol=1e-6)

  @pytest.mark.fast.with_args("num_iterations", [30, 60])
  def test_callback_fn(self, num_iterations: int):
    """Check that the callback function is actually called."""

    def progress_fn(
        status: Tuple[np.ndarray, np.ndarray, np.ndarray,
                      sinkhorn.SinkhornState], *args: Any
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

    geom = pointcloud.PointCloud(self.x, self.y, epsilon=1e-3)
    lin_prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)

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
