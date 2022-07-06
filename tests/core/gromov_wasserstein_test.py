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
"""Tests for the Gromov Wasserstein."""
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import gromov_wasserstein, quad_problems
from ott.geometry import geometry, pointcloud


class TestGromovWasserstein:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    d_x = 2
    d_y = 3
    self.n, self.m = 5, 6
    keys = jax.random.split(rng, 8)
    self.x = jax.random.uniform(keys[0], (self.n, d_x))
    self.y = jax.random.uniform(keys[1], (self.m, d_y))
    a = jax.random.uniform(keys[2], (self.n,)) + 0.1
    b = jax.random.uniform(keys[3], (self.m,)) + 0.1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(keys[4], (self.n, self.n))
    self.cy = jax.random.uniform(keys[5], (self.m, self.m))
    self.xx = jax.random.uniform(keys[6], (self.n, d_x))
    self.yy = jax.random.uniform(keys[7], (self.m, d_x))

  def test_flag_store_errors(self):
    """Tests whether errors are properly stored if requested."""
    threshold_sinkhorn = 1e-2
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    out = gromov_wasserstein.gromov_wasserstein(
        geom_xx=geom_x, geom_yy=geom_y, a=self.a, b=self.b, epsilon=.1
    ).errors
    assert out is None

    out = gromov_wasserstein.gromov_wasserstein(
        geom_xx=geom_x,
        geom_yy=geom_y,
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

  @pytest.mark.parametrize("jit", [False, True])
  def test_gradient_marginals_gromov_wasserstein(self, jit: bool):
    """Test gradient w.r.t. probability weights."""
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)

    def reg_gw(a, b, implicit):
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001
      }
      out = gromov_wasserstein.gromov_wasserstein(
          geom_x,
          geom_y,
          a=a,
          b=b,
          epsilon=1.0,
          loss='sqeucl',
          max_iterations=10,
          jit=jit,
          sinkhorn_kwargs=sinkhorn_kwargs
      )
      return out.reg_gw_cost, (out.linear_state.f, out.linear_state.g)

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

  @pytest.mark.fast
  def test_gromov_wasserstein_pointcloud(self):
    """Test basic computations pointclouds."""

    def reg_gw(x, y, a, b):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      return gromov_wasserstein.gromov_wasserstein(
          geom_x, geom_y, a=a, b=b, epsilon=1.0, max_iterations=10
      ).reg_gw_cost

    assert not jnp.isnan(reg_gw(self.x, self.y, self.a, self.b))

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_gradient_gromov_wasserstein_pointcloud(self, lse_mode: bool):
    """Test gradient w.r.t. pointclouds."""

    def reg_gw(x, y, a, b, implicit):
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001,
          'lse_mode': lse_mode
      }
      return gromov_wasserstein.gromov_wasserstein(
          geom_x,
          geom_y,
          a=a,
          b=b,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      ).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(
          reg_gw, argnums=(
              0,
              1,
          )
      )
      _, grad_reg_gw = reg_gw_and_grad(self.x, self.y, self.a, self.b, implicit)
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
  def test_gradient_gromov_wasserstein_geometry(self, lse_mode: bool):
    """Test gradient w.r.t. cost matrices."""

    def reg_gw(cx, cy, a, b, implicit):
      geom_x = geometry.Geometry(cost_matrix=cx)
      geom_y = geometry.Geometry(cost_matrix=cy)
      sinkhorn_kwargs = {
          'implicit_differentiation': implicit,
          'max_iterations': 1001,
          'lse_mode': lse_mode
      }
      return gromov_wasserstein.gromov_wasserstein(
          geom_x,
          geom_y,
          a=a,
          b=b,
          epsilon=1.0,
          max_iterations=10,
          sinkhorn_kwargs=sinkhorn_kwargs
      ).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_and_grad = jax.value_and_grad(
          reg_gw, argnums=(
              0,
              1,
          )
      )
      _, grad_reg_gw = reg_gw_and_grad(
          self.cx, self.cy, self.a, self.b, implicit
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

  def test_adaptive_threshold(self):
    """Checking solution is improved with smaller threshold for convergence."""
    geom_x = pointcloud.PointCloud(self.x, self.x)
    geom_y = pointcloud.PointCloud(self.y, self.y)

    def loss_thre(threshold):
      return gromov_wasserstein.gromov_wasserstein(
          geom_xx=geom_x,
          geom_yy=geom_y,
          a=self.a,
          b=self.b,
          epsilon=.1,
          threshold=threshold
      ).reg_gw_cost

    assert loss_thre(1e-1), loss_thre(1e-3)
    assert loss_thre(1e-3), loss_thre(1e-5)

  @pytest.mark.fast
  def test_gw_lr(self, rng: jnp.ndarray):
    """Checking LR and Entropic have similar outputs on same problem."""
    rngs = jax.random.split(rng, 4)
    n, m, d1, d2 = 24, 17, 2, 3
    x = jax.random.uniform(rngs[0], (n, d1))
    y = jax.random.uniform(rngs[1], (m, d2))
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    geom_xx = pointcloud.PointCloud(x)
    geom_yy = pointcloud.PointCloud(y)
    prob = quad_problems.QuadraticProblem(geom_xx, geom_yy, a=a, b=b)
    solver = gromov_wasserstein.GromovWasserstein(rank=5)
    ot_gwlr = solver(prob)
    solver = gromov_wasserstein.GromovWasserstein(epsilon=0.2)
    ot_gw = solver(prob)
    np.testing.assert_allclose(ot_gwlr.costs, ot_gw.costs, rtol=5e-2)

  def test_gw_lr_fused(self, rng: jnp.ndarray):
    """Checking LR and Entropic have similar outputs on same fused problem."""
    rngs = jax.random.split(rng, 5)
    n, m, d1, d2 = 24, 17, 2, 3
    x = jax.random.uniform(rngs[0], (n, d1))
    y = jax.random.uniform(rngs[1], (m, d2))
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    z = jax.random.uniform(rngs[4], (m, d1))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    geom_xx = pointcloud.PointCloud(x)
    geom_yy = pointcloud.PointCloud(y)
    geom_xy = pointcloud.PointCloud(x, z)  # only used to compute n x m matrix
    prob = quad_problems.QuadraticProblem(
        geom_xx, geom_yy, geom_xy=geom_xy, fused_penalty=1.3, a=a, b=b
    )
    solver = gromov_wasserstein.GromovWasserstein(rank=6)
    ot_gwlr = solver(prob)
    solver = gromov_wasserstein.GromovWasserstein(rank=6, epsilon=1e-1)
    ot_gwlreps = solver(prob)
    solver = gromov_wasserstein.GromovWasserstein(epsilon=5e-2)
    ot_gw = solver(prob)

    # Test solutions look alike
    assert 0.1 > jnp.linalg.norm(ot_gwlr.matrix - ot_gw.matrix)
    assert 0.1 > jnp.linalg.norm(ot_gwlr.matrix - ot_gwlreps.matrix)
    # Test at least some difference when adding bigger entropic regularization
    assert jnp.linalg.norm(ot_gwlr.matrix - ot_gwlreps.matrix) > 1e-3

  @pytest.mark.parametrize("scale_cost", [True, "mean", "max_cost"])
  def test_gw_fused_scale_cost(self, scale_cost: Union[bool, str]):
    epsilon = 0.1
    fused_penalty = 1
    geom_x = pointcloud.PointCloud(self.x, scale_cost=None)
    geom_y = pointcloud.PointCloud(self.y, scale_cost=None)
    geom_xy = pointcloud.PointCloud(self.xx, self.yy, scale_cost=None)
    geom_x_scaled = pointcloud.PointCloud(self.x, scale_cost=scale_cost)
    geom_y_scaled = pointcloud.PointCloud(self.y, scale_cost=scale_cost)
    geom_xy_scaled = pointcloud.PointCloud(
        self.xx, self.yy, scale_cost=scale_cost
    )

    gt = gromov_wasserstein.gromov_wasserstein(
        geom_xx=geom_x_scaled,
        geom_yy=geom_y_scaled,
        geom_xy=geom_xy_scaled,
        fused_penalty=fused_penalty,
        epsilon=epsilon,
        scale_cost=False
    )
    pred = gromov_wasserstein.gromov_wasserstein(
        geom_xx=geom_x,
        geom_yy=geom_y,
        geom_xy=geom_xy,
        fused_penalty=fused_penalty,
        epsilon=epsilon,
        scale_cost=scale_cost
    )

    np.testing.assert_allclose(pred.matrix, gt.matrix)
    np.testing.assert_allclose(pred.costs, gt.costs)

  @pytest.mark.parametrize("axis", [0, 1])
  def test_gw_lr_apply(self, axis: int):
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    out = gromov_wasserstein.gromov_wasserstein(
        geom_xx=geom_x,
        geom_yy=geom_y,
        a=self.a,
        b=self.b,
        epsilon=.1,
        rank=2,
    )

    arr, matrix = (self.x, out.matrix) if axis == 0 else (self.y, out.matrix.T)
    res_apply = out.apply(arr.T, axis=axis)
    res_matrix = arr.T @ matrix

    np.testing.assert_allclose(res_apply, res_matrix, rtol=1e-5, atol=1e-5)
