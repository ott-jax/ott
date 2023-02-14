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
"""Tests for the Fused Gromov Wasserstein."""
from typing import Literal, Tuple, Union

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import geometry, low_rank, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from ott.solvers.quadratic import gromov_wasserstein as gw_solver


class TestFusedGromovWasserstein:

  # TODO(michalk8): refactor me in the future
  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    d_x = 2
    d_y = 3
    d_xy = 4
    self.n, self.m = 5, 6
    keys = jax.random.split(rng, 7)
    self.x = jax.random.uniform(keys[0], (self.n, d_x))
    self.y = jax.random.uniform(keys[1], (self.m, d_y))
    self.x_2 = jax.random.uniform(keys[0], (self.n, d_xy))
    self.y_2 = jax.random.uniform(keys[1], (self.m, d_xy))
    self.fused_penalty = 2.0
    self.fused_penalty_2 = 0.05
    a = jax.random.uniform(keys[2], (self.n,)) + 0.1
    b = jax.random.uniform(keys[3], (self.m,)) + 0.1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(keys[4], (self.n, self.n))
    self.cy = jax.random.uniform(keys[5], (self.m, self.m))
    self.cxy = jax.random.uniform(keys[6], (self.n, self.m))

  @pytest.mark.fast.with_args("jit", [False, True], only_fast=0)
  def test_gradient_marginals_fgw_solver(self, jit: bool):
    """Test gradient w.r.t. probability weights."""
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    geom_xy = pointcloud.PointCloud(self.x_2, self.y_2)

    def reg_gw(a: jnp.ndarray, b: jnp.ndarray, implicit: bool):
      prob = quadratic_problem.QuadraticProblem(
          geom_x, geom_y, geom_xy, fused_penalty=self.fused_penalty, a=a, b=b
      )

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      linear_solver = sinkhorn.Sinkhorn(
          implicit_diff=implicit_diff, max_iterations=1000
      )
      solver = gromov_wasserstein.GromovWasserstein(
          linear_ot_solver=linear_solver, epsilon=1.0
      )

      out = solver(prob)

      return out.reg_gw_cost, (out.linear_state.f, out.linear_state.g)

    grad_matrices = [None, None]
    reg_fgw_grad = jax.grad(reg_gw, has_aux=True, argnums=(0, 1))
    if jit:
      reg_fgw_grad = jax.jit(reg_fgw_grad, static_argnames="implicit")

    for i, implicit in enumerate([True, False]):
      (g_a, g_b), aux = reg_fgw_grad(self.a, self.b, implicit)
      grad_matrices[i] = (g_a, g_b)
      grad_manual_a = aux[0] - jnp.log(self.a)
      grad_manual_b = aux[1] - jnp.log(self.b)
      assert not jnp.any(jnp.isnan(g_a))
      assert not jnp.any(jnp.isnan(g_b))
      np.testing.assert_allclose(grad_manual_a, g_a, rtol=1e-2, atol=1e-2)
      np.testing.assert_allclose(grad_manual_b, g_b, rtol=1e-2, atol=1e-2)

    gi_a, gi_b = grad_matrices[0]
    g_a, g_b = grad_matrices[1]

    np.testing.assert_allclose(g_a, gi_a, rtol=1e-02, atol=1e-02)
    np.testing.assert_allclose(g_b, gi_b, rtol=1e-02, atol=1e-02)

  @pytest.mark.parametrize(
      "lse_mode,is_cost", [(True, False), (False, True)],
      ids=["lse-pc", "kernel-cost-mat"]
  )
  def test_gradient_fgw_solver_geometry(self, lse_mode: bool, is_cost: bool):
    """Test gradient w.r.t. the geometries."""

    def reg_gw(
        x: jnp.ndarray, y: jnp.ndarray,
        xy: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
        fused_penalty: float, a: jnp.ndarray, b: jnp.ndarray, implicit: bool
    ):
      if is_cost:
        geom_x = geometry.Geometry(cost_matrix=x)
        geom_y = geometry.Geometry(cost_matrix=y)
        geom_xy = geometry.Geometry(cost_matrix=xy)
      else:
        geom_x = pointcloud.PointCloud(x)
        geom_y = pointcloud.PointCloud(y)
        geom_xy = pointcloud.PointCloud(xy[0], xy[1])
      prob = quadratic_problem.QuadraticProblem(
          geom_x, geom_y, geom_xy, fused_penalty=fused_penalty, a=a, b=b
      )

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      linear_solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode, implicit_diff=implicit_diff, max_iterations=1000
      )
      solver = gromov_wasserstein.GromovWasserstein(
          linear_ot_solver=linear_solver, epsilon=1.0, max_iterations=10
      )

      return solver(prob).reg_gw_cost

    if is_cost:
      x, y, xy = self.cx, self.cy, self.cxy
    else:
      x, y, xy = self.x, self.y, (self.x_2, self.y_2)
    grad_matrices = [None, None]
    reg_fgw_grad = jax.grad(reg_gw, argnums=(0, 1, 2))

    for i, implicit in enumerate([True, False]):
      grad_matrices[i] = reg_fgw_grad(
          x, y, xy, self.fused_penalty, self.a, self.b, implicit
      )
      assert not jnp.any(jnp.isnan(grad_matrices[i][0]))
      assert not jnp.any(jnp.isnan(grad_matrices[i][1]))

    gi_x, gi_y, gi_xy = grad_matrices[0]
    g_x, g_y, g_xy = grad_matrices[1]

    np.testing.assert_allclose(g_x, gi_x, rtol=1e-02, atol=1e-02)
    np.testing.assert_allclose(g_y, gi_y, rtol=1e-02, atol=1e-02)
    if is_cost:
      np.testing.assert_allclose(g_xy, gi_xy, rtol=1e-02, atol=1e-02)
    else:
      np.testing.assert_allclose(g_xy[0], gi_xy[0], rtol=1e-02, atol=1e-02)
      np.testing.assert_allclose(g_xy[1], gi_xy[1], rtol=1e-02, atol=1e-02)

  def test_fgw_adaptive_threshold(self):
    """Checking solution is improved with smaller threshold for convergence."""
    geom_x = pointcloud.PointCloud(self.x, self.x)
    geom_y = pointcloud.PointCloud(self.y, self.y)
    geom_xy = pointcloud.PointCloud(self.x_2, self.y_2)

    # without warm start for calls to sinkhorn
    def loss_thre(threshold: float) -> float:
      prob = quadratic_problem.QuadraticProblem(
          geom_x,
          geom_y,
          geom_xy,
          a=self.a,
          b=self.b,
          fused_penalty=self.fused_penalty_2
      )
      solver = gromov_wasserstein.GromovWasserstein(
          threshold=threshold, epsilon=1e-1
      )

      return solver(prob).reg_gw_cost

    assert loss_thre(1e-1) > loss_thre(1e-4)
    assert loss_thre(1e-3) > loss_thre(1e-5)

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_gradient_fgw_solver_penalty(self, lse_mode: bool):
    """Test gradient w.r.t. penalty."""

    def reg_gw(
        cx: jnp.ndarray, cy: jnp.ndarray, cxy: jnp.ndarray,
        fused_penalty: float, a: jnp.ndarray, b: jnp.ndarray, implicit: bool
    ) -> float:
      geom_x = geometry.Geometry(cost_matrix=cx)
      geom_y = geometry.Geometry(cost_matrix=cy)
      geom_xy = geometry.Geometry(cost_matrix=cxy)
      prob = quadratic_problem.QuadraticProblem(
          geom_x, geom_y, geom_xy, a=a, b=b, fused_penalty=fused_penalty
      )

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      linear_solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode, implicit_diff=implicit_diff, max_iterations=1000
      )
      solver = gromov_wasserstein.GromovWasserstein(
          epsilon=1.0, max_iterations=10, linear_ot_solver=linear_solver
      )
      return solver(prob).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_fgw_grad = jax.grad(reg_gw, argnums=(3,))
      grad_matrices[i] = reg_fgw_grad(
          self.cx, self.cy, self.cxy, self.fused_penalty, self.a, self.b,
          implicit
      )
      assert not jnp.any(jnp.isnan(grad_matrices[i][0]))

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )

  @pytest.mark.limit_memory("400 MB")
  @pytest.mark.parametrize("jit", [False, True])
  def test_fgw_lr_memory(self, rng: jnp.ndarray, jit: bool):
    # Total memory allocated on CI: 342.5MiB (32bit)
    rngs = jax.random.split(rng, 4)
    n, m, d1, d2 = 15_000, 10_000, 2, 3
    x = jax.random.uniform(rngs[0], (n, d1))
    y = jax.random.uniform(rngs[1], (m, d2))
    xx = jax.random.uniform(rngs[2], (n, d2))
    yy = jax.random.uniform(rngs[3], (m, d2))
    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    geom_xy = pointcloud.PointCloud(xx, yy)
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, geom_xy)

    solver = gromov_wasserstein.GromovWasserstein(rank=5)
    if jit:
      solver = jax.jit(solver, static_argnames="rank")

    ot_gwlr = solver(prob)

    res0 = ot_gwlr.apply(x.T, axis=0)
    res1 = ot_gwlr.apply(y.T, axis=1)

    assert ot_gwlr.converged
    assert res0.shape == (d1, m)
    assert res1.shape == (d2, n)

  @pytest.mark.parametrize("cost_rank", [4, (2, 3, 4)])
  def test_fgw_lr_generic_cost_matrix(
      self, rng: jnp.ndarray, cost_rank: Union[int, Tuple[int, int, int]]
  ):
    n, m = 70, 100
    key1, key2, key3, key4 = jax.random.split(rng, 4)
    x = jax.random.normal(key1, shape=(n, 7))
    y = jax.random.normal(key2, shape=(m, 6))
    xx = jax.random.normal(key3, shape=(n, 5))
    yy = jax.random.normal(key4, shape=(m, 5))

    geom_x = geometry.Geometry(cost_matrix=x @ x.T)
    geom_y = geometry.Geometry(cost_matrix=y @ y.T)
    geom_xy = geometry.Geometry(cost_matrix=xx @ yy.T)

    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, geom_xy, ranks=cost_rank, tolerances=5e-1
    )
    assert prob._is_low_rank_convertible
    lr_prob = prob.to_low_rank()
    assert lr_prob.is_low_rank

    solver = gw_solver.GromovWasserstein(rank=5, epsilon=1.0)
    out = solver(prob)

    assert solver.rank == 5
    # make sure we don't modify the problem in-place
    for geom in [prob.geom_xx, prob.geom_yy, prob.geom_xy]:
      assert not isinstance(geom, low_rank.LRCGeometry)
    ranks = (cost_rank,) * 3 if isinstance(cost_rank, int) else cost_rank
    for rank, geom in zip(
        ranks, [lr_prob.geom_xx, lr_prob.geom_yy, lr_prob.geom_xy]
    ):
      assert geom.cost_rank == rank

    assert out.converged
    assert out.reg_gw_cost > 0
    np.testing.assert_array_equal(jnp.isfinite(out.costs), True)

  @pytest.mark.parametrize("scale_cost", ["mean", "max_cost"])
  def test_fgw_scale_cost(self, scale_cost: Literal["mean", "max_cost"]):
    epsilon = 0.1
    fused_penalty = 1
    geom_x = pointcloud.PointCloud(self.x, scale_cost=1.)
    geom_y = pointcloud.PointCloud(self.y, scale_cost=1.)
    geom_xy = pointcloud.PointCloud(self.x_2, self.y_2, scale_cost=1.)
    geom_x_scaled = pointcloud.PointCloud(self.x, scale_cost=scale_cost)
    geom_y_scaled = pointcloud.PointCloud(self.y, scale_cost=scale_cost)
    geom_xy_scaled = pointcloud.PointCloud(
        self.x_2, self.y_2, scale_cost=scale_cost
    )

    prob_no_scale = quadratic_problem.QuadraticProblem(
        geom_x_scaled,
        geom_y_scaled,
        geom_xy_scaled,
        fused_penalty=fused_penalty,
        scale_cost=False
    )
    prob_scale = quadratic_problem.QuadraticProblem(
        geom_x,
        geom_y,
        geom_xy,
        fused_penalty=fused_penalty,
        scale_cost=scale_cost
    )
    solver = gromov_wasserstein.GromovWasserstein(epsilon=epsilon)

    gt = solver(prob_scale)
    pred = solver(prob_no_scale)

    np.testing.assert_allclose(pred.matrix, gt.matrix)
    np.testing.assert_allclose(pred.costs, gt.costs)
