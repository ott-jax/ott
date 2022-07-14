# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for the Policy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import linear_problems, sinkhorn
from ott.geometry import costs, geometry, pointcloud


class TestSinkhorn:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
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
      "lse_mode,momentum,chg_momentum_from,inner_iterations,norm_error",
      [(True, 1.0, 29, 10, 1), (False, 1.0, 30, 10, 1), (True, 1.0, 60, 1, 2),
       (True, 1.0, 12, 24, 4)],
      ids=["lse-Leh-mom", "scal-Leh-mom", "lse-Leh-1", "lse-Leh-24"],
      only_fast=[0, -1],
  )
  def test_euclidean_point_cloud(
      self, lse_mode, momentum, chg_momentum_from, inner_iterations, norm_error
  ):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1)
    out = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        momentum=momentum,
        chg_momentum_from=chg_momentum_from,
        inner_iterations=inner_iterations,
        norm_error=norm_error,
        lse_mode=lse_mode
    )
    errors = out.errors
    err = errors[errors > -1][-1]
    assert threshold > err

    other_geom = pointcloud.PointCloud(self.x, self.y + 0.3, epsilon=0.1)
    cost_other = out.cost_at_geom(other_geom)
    assert not jnp.isnan(cost_other)

  def test_autoepsilon(self):
    """Check that with auto-epsilon, dual potentials scale."""
    scale = 2.77
    # First geom specifies explicitly relative_epsilon to be True. This is not
    # needed in principle, but introduced here to test logic.
    geom_1 = pointcloud.PointCloud(self.x, self.y, relative_epsilon=True)
    # jit first with jit inside sinkhorn call.
    f_1 = sinkhorn.sinkhorn(
        geom_1, a=self.a, b=self.b, tau_a=.99, tau_b=.97, jit=True
    ).f

    # Second geom does not provide whether epsilon is relative.
    geom_2 = pointcloud.PointCloud(scale * self.x, scale * self.y)
    # jit now with jit outside sinkhorn call.
    compute_f = jax.jit(
        lambda g, a, b: sinkhorn.sinkhorn(g, a, b, tau_a=.99, tau_b=.97).f
    )
    f_2 = compute_f(geom_2, self.a, self.b)

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
    geom = pointcloud.PointCloud(self.x, self.y, init=init, decay=decay)
    out_1 = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        tau_a=tau_a,
        tau_b=tau_b,
        jit=True,
        lse_mode=lse_mode,
        threshold=1e-5
    )

    geom = pointcloud.PointCloud(self.x, self.y)
    out_2 = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        tau_a=tau_a,
        tau_b=tau_b,
        jit=True,
        lse_mode=lse_mode,
        threshold=1e-5
    )
    # recenter if problem is balanced, since in that case solution is only
    # valid up to additive constant.
    unb = (tau_a < 1.0 or tau_b < 1.0)
    np.testing.assert_allclose(
        out_1.f if unb else out_1.f - jnp.mean(out_1.f[jnp.isfinite(out_1.f)]),
        out_2.f if unb else out_2.f - jnp.mean(out_2.f[jnp.isfinite(out_2.f)]),
        rtol=1e-4,
        atol=1e-4
    )

  @pytest.mark.fast
  def test_euclidean_point_cloud_min_iter(self):
    """Testing the min_iterations parameter."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        min_iterations=34,
        implicit_differentiation=False
    ).errors
    err = errors[jnp.logical_and(errors > -1, jnp.isfinite(errors))][-1]
    assert threshold > err
    assert errors[0] == jnp.inf
    assert errors[1] == jnp.inf
    assert errors[2] == jnp.inf
    assert errors[3] > 0

  def test_geom_vs_point_cloud(self):
    """Two point clouds vs. simple cost_matrix execution of sinkorn."""
    geom_1 = pointcloud.PointCloud(self.x, self.y)
    geom_2 = geometry.Geometry(geom_1.cost_matrix)

    f_1 = sinkhorn.sinkhorn(geom_1, a=self.a, b=self.b).f
    f_2 = sinkhorn.sinkhorn(geom_2, a=self.a, b=self.b).f
    # recentering to remove ambiguity on equality up to additive constant.
    f_1 -= jnp.mean(f_1[jnp.isfinite(f_1)])
    f_2 -= jnp.mean(f_2[jnp.isfinite(f_2)])

    np.testing.assert_allclose(f_1, f_2, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_online_euclidean_point_cloud(self, lse_mode: bool):
    """Testing the online way to handle geometry."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1, batch_size=5)
    errors = sinkhorn.sinkhorn(
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
        self.x, self.y, cost_fn=costs.Euclidean(), epsilon=eps, batch_size=10
    )

    batch_geom = pointcloud.PointCloud(self.x, self.y, epsilon=eps)
    batch_geom_euc = pointcloud.PointCloud(
        self.x, self.y, cost_fn=costs.Euclidean(), epsilon=eps
    )

    out_online = sinkhorn.sinkhorn(
        online_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode
    )
    out_batch = sinkhorn.sinkhorn(
        batch_geom, a=self.a, b=self.b, threshold=threshold, lse_mode=lse_mode
    )
    out_online_euc = sinkhorn.sinkhorn(
        online_geom_euc,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode
    )
    out_batch_euc = sinkhorn.sinkhorn(
        batch_geom_euc,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode
    )

    # Checks regularized transport costs match.
    np.testing.assert_allclose(out_online.reg_ot_cost, out_batch.reg_ot_cost)
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
    keys = jax.random.split(self.rng, 6)
    x = jax.random.uniform(keys[0], (n, d))
    y = jax.random.uniform(keys[1], (m, d))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    transport_t_vec_a = [None, None, None, None]
    transport_vec_b = [None, None, None, None]

    batch_b = 8

    vec_a = jax.random.normal(keys[4], (n,))
    vec_b = jax.random.normal(keys[5], (batch_b, m))

    # test with lse_mode and online = True / False
    for j, lse_mode in enumerate([True, False]):
      for i, batch_size in enumerate([16, None]):
        geom = pointcloud.PointCloud(x, y, batch_size=batch_size, epsilon=0.2)
        sink = sinkhorn.sinkhorn(geom, a, b, lse_mode=lse_mode)

        transport_t_vec_a[i + 2 * j] = geom.apply_transport_from_potentials(
            sink.f, sink.g, vec_a, axis=0
        )
        transport_vec_b[i + 2 * j] = geom.apply_transport_from_potentials(
            sink.f, sink.g, vec_b, axis=1
        )

        transport = geom.transport_from_potentials(sink.f, sink.g)

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
    keys = jax.random.split(self.rng, 6)
    x = jax.random.uniform(keys[0], (n, d))
    y = jax.random.uniform(keys[1], (m, d))
    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    transport_t_vec_a = [None, None, None, None]
    transport_vec_b = [None, None, None, None]

    batch_b = 8

    vec_a = jax.random.normal(keys[4], (n,))
    vec_b = jax.random.normal(keys[5], (batch_b, m))

    # test with lse_mode and online = True / False
    for j, lse_mode in enumerate([True, False]):
      for i, batch_size in enumerate([64, None]):
        geom = pointcloud.PointCloud(x, y, batch_size=batch_size, epsilon=0.2)
        sink = sinkhorn.sinkhorn(geom, a, b, lse_mode=lse_mode)

        u = geom.scaling_from_potential(sink.f)
        v = geom.scaling_from_potential(sink.g)

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
    out = sinkhorn.sinkhorn(
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

    out_restarted = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=lse_mode,
        init_dual_a=init_dual_a,
        init_dual_b=init_dual_b,
        inner_iterations=1
    )

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

  @pytest.mark.limit_memory("90 MB")
  @pytest.mark.fast.with_args("batch_size", [500, 1000], only_fast=0)
  def test_sinkhorn_online_memory(self, batch_size: int):
    # offline: Total memory allocated: 240.1MiB
    # online (500): Total memory allocated: 33.4MiB
    # online (1000): Total memory allocated: 45.6MiB
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    n, m = 5000, 4000
    x = jax.random.uniform(rngs[0], (n, 2))
    y = jax.random.uniform(rngs[1], (m, 2))
    geom = pointcloud.PointCloud(x, y, batch_size=batch_size, epsilon=1)
    problem = linear_problems.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()

    out = solver(problem)
    assert out.converged
