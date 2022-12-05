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
"""Tests for the Fused Gromov Wasserstein."""
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.geometry import geometry, low_rank, pointcloud
from ott.problems.quadratic import quadratic_problem
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

  def test_fgw_flag_store_errors_fused(self):
    """Tests whether errors are properly stored if requested."""
    threshold_sinkhorn = 1e-2
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    geom_xy = pointcloud.PointCloud(self.x_2, self.y_2)
    out = gw_solver.gromov_wasserstein(
        geom_xx=geom_x,
        geom_yy=geom_y,
        geom_xy=geom_xy,
        fused_penalty=self.fused_penalty,
        a=self.a,
        b=self.b,
        epsilon=.1
    ).errors
    assert out is None

    out = gw_solver.gromov_wasserstein(
        geom_xx=geom_x,
        geom_yy=geom_y,
        geom_xy=geom_xy,
        fused_penalty=self.fused_penalty,
        a=self.a,
        b=self.b,
        epsilon=.1,
        store_inner_errors=True,
        sinkhorn_kwargs={
            'threshold': threshold_sinkhorn
        }
    ).errors
    out = out[jnp.sum(out > 0, axis=1) > 0, :]
    last_errors = out[-1, :]

    assert threshold_sinkhorn > last_errors[last_errors > -1][-1]
    assert out.ndim == 2

  @pytest.mark.fast.with_args("jit", [False, True], only_fast=0)
  def test_gradient_marginals_fgw_solver(self, jit: bool):
    """Test gradient w.r.t. probability weights."""
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    geom_xy = pointcloud.PointCloud(self.x_2, self.y_2)
    fused_penalty = self.fused_penalty

    def reg_gw(a: jnp.ndarray, b: jnp.ndarray, implicit: bool):
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001
      }
      out = gw_solver.gromov_wasserstein(
          geom_x,
          geom_y,
          geom_xy=geom_xy,
          fused_penalty=fused_penalty,
          a=a,
          b=b,
          epsilon=1.0,
          loss='sqeucl',
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      )
      return out.reg_gw_cost, (out.linear_state.f, out.linear_state.g)

    if jit:
      reg_gw = jax.jit(reg_gw, static_argnames="implicit")

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(reg_gw, has_aux=True, argnums=(0, 1))
      (_, aux), grad_reg_gw = reg_gw_and_grad(self.a, self.b, implicit)
      grad_matrices[i] = grad_reg_gw
      grad_manual_a = aux[0] - jnp.log(self.a)
      grad_manual_b = aux[1] - jnp.log(self.b)
      assert not jnp.any(jnp.isnan(grad_reg_gw[0]))
      assert not jnp.any(jnp.isnan(grad_reg_gw[1]))
      np.testing.assert_allclose(
          grad_manual_a, grad_reg_gw[0], rtol=1e-2, atol=1e-2
      )
      np.testing.assert_allclose(
          grad_manual_b, grad_reg_gw[1], rtol=1e-2, atol=1e-2
      )
    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-02, atol=1e-02
    )

  @pytest.mark.fast.with_args(lse_mode=[False, True], only_fast=1)
  def test_fgw_solver_pointcloud(self, lse_mode: bool):
    """Test basic computations pointclouds."""

    def reg_gw(x, y, x_2, y_2, fused_penalty, a, b):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      geom_xy = pointcloud.PointCloud(x_2, y_2)
      return gw_solver.gromov_wasserstein(
          geom_x,
          geom_y,
          geom_xy=geom_xy,
          fused_penalty=fused_penalty,
          a=a,
          b=b,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs={
              "lse_mode": lse_mode
          },
      ).reg_gw_cost

    cost = reg_gw(
        self.x, self.y, self.x_2, self.y_2, self.fused_penalty, self.a, self.b
    )
    assert cost is not None

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_gradient_fgw_solver_pointcloud(self, lse_mode: bool):
    """Test gradient w.r.t. pointclouds."""

    def reg_gw(x, y, x_2, y_2, fused_penalty, a, b, implicit):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      geom_xy = pointcloud.PointCloud(x_2, y_2)
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001,
          'lse_mode': lse_mode
      }
      return gw_solver.gromov_wasserstein(
          geom_x,
          geom_y,
          geom_xy=geom_xy,
          fused_penalty=fused_penalty,
          a=a,
          b=b,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      ).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(reg_gw, argnums=(0, 1))
      _, grad_reg_gw = reg_gw_and_grad(
          self.x, self.y, self.x_2, self.y_2, self.fused_penalty, self.a,
          self.b, implicit
      )
      grad_matrices[i] = grad_reg_gw
      assert not jnp.any(jnp.isnan(grad_reg_gw[0]))
      assert not jnp.any(jnp.isnan(grad_reg_gw[1]))

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-02, atol=1e-02
    )

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_gradient_fgw_solver_geometry(self, lse_mode: bool):
    """Test gradient w.r.t. cost matrices."""

    def reg_gw(cx, cy, cxy, fused_penalty, a, b, implicit):
      geom_x = geometry.Geometry(cost_matrix=cx)
      geom_y = geometry.Geometry(cost_matrix=cy)
      geom_xy = geometry.Geometry(cost_matrix=cxy)
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001,
          'lse_mode': lse_mode
      }
      return gw_solver.gromov_wasserstein(
          geom_x,
          geom_y,
          geom_xy=geom_xy,
          fused_penalty=fused_penalty,
          a=a,
          b=b,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      ).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(reg_gw, argnums=(0, 1, 2))
      _, grad_reg_gw = reg_gw_and_grad(
          self.cx, self.cy, self.cxy, self.fused_penalty, self.a, self.b,
          implicit
      )
      grad_matrices[i] = grad_reg_gw
      assert not jnp.any(jnp.isnan(grad_reg_gw[0]))
      assert not jnp.any(jnp.isnan(grad_reg_gw[1]))

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][2], grad_matrices[1][2], rtol=1e-02, atol=1e-02
    )

  def test_fgw_adaptive_threshold(self):
    """Checking solution is improved with smaller threshold for convergence."""
    geom_x = pointcloud.PointCloud(self.x, self.x)
    geom_y = pointcloud.PointCloud(self.y, self.y)
    geom_xy = pointcloud.PointCloud(self.x_2, self.y_2)

    # without warm start for calls to sinkhorn
    def loss_thre(threshold: float) -> float:
      return gw_solver.gromov_wasserstein(
          geom_xx=geom_x,
          geom_yy=geom_y,
          geom_xy=geom_xy,
          fused_penalty=self.fused_penalty_2,
          a=self.a,
          b=self.b,
          epsilon=.1,
          threshold=threshold
      ).reg_gw_cost

    assert loss_thre(1e-1) > loss_thre(1e-3)
    assert loss_thre(1e-3) > loss_thre(1e-5)

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_gradient_fgw_solver_penalty(self, lse_mode: bool):
    """Test gradient w.r.t. penalty."""

    def reg_gw(cx, cy, cxy, fused_penalty, a, b, implicit):
      geom_x = geometry.Geometry(cost_matrix=cx)
      geom_y = geometry.Geometry(cost_matrix=cy)
      geom_xy = geometry.Geometry(cost_matrix=cxy)
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001,
          'lse_mode': lse_mode
      }
      return gw_solver.gromov_wasserstein(
          geom_x,
          geom_y,
          geom_xy=geom_xy,
          fused_penalty=fused_penalty,
          a=a,
          b=b,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      ).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(reg_gw, argnums=(3,))
      _, grad_reg_gw = reg_gw_and_grad(
          self.cx, self.cy, self.cxy, self.fused_penalty, self.a, self.b,
          implicit
      )
      grad_matrices[i] = grad_reg_gw
      assert not jnp.any(jnp.isnan(grad_reg_gw[0]))
    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )

  def test_effect_fused_penalty(self):

    def reg_fgw(x, y, x_2, y_2, fused_penalty, a, b):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      geom_xy = pointcloud.PointCloud(x_2, y_2)
      sinkhorn_kwargs = {'max_iterations': 1001}
      return gw_solver.gromov_wasserstein(
          geom_x,
          geom_y,
          geom_xy=geom_xy,
          fused_penalty=fused_penalty,
          a=a,
          b=b,
          epsilon=1.0,
          sinkhorn_kwargs=sinkhorn_kwargs
      )

    def reg_gw(x, y, a, b):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      sinkhorn_kwargs = {'max_iterations': 1001}
      return gw_solver.gromov_wasserstein(
          geom_x,
          geom_y,
          a=a,
          b=b,
          epsilon=1.0,
          sinkhorn_kwargs=sinkhorn_kwargs
      )

    fgw_output = reg_fgw(
        self.x, self.y, self.x_2, self.y_2, self.fused_penalty, self.a, self.b
    )
    gw_output = reg_gw(self.x, self.y, self.a, self.b)
    assert fgw_output.reg_gw_cost > gw_output.reg_gw_cost
    with pytest.raises(AssertionError):
      np.testing.assert_array_almost_equal(
          fgw_output.matrix[0, 0], gw_output.matrix[0, 0]
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

    solver = gw_solver.gromov_wasserstein
    if jit:
      solver = jax.jit(solver, static_argnames="rank")

    ot_gwlr = solver(
        geom_x,
        geom_y,
        geom_xy,
        rank=5,
    )
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

    problem = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, geom_xy, ranks=cost_rank, tolerances=5e-1
    )
    assert problem._is_low_rank_convertible
    lr_prob = problem.to_low_rank()
    assert lr_prob.is_low_rank

    solver = gw_solver.GromovWasserstein(rank=5, epsilon=1)
    out = solver(problem)

    assert solver.rank == 5
    # make sure we don't modify the problem in-place
    for geom in [problem.geom_xx, problem.geom_yy, problem.geom_xy]:
      assert not isinstance(geom, low_rank.LRCGeometry)
    ranks = (cost_rank,) * 3 if isinstance(cost_rank, int) else cost_rank
    for rank, geom in zip(
        ranks, [lr_prob.geom_xx, lr_prob.geom_yy, lr_prob.geom_xy]
    ):
      assert geom.cost_rank == rank

    assert out.converged
    assert out.reg_gw_cost > 0
    np.testing.assert_array_equal(jnp.isfinite(out.costs), True)
