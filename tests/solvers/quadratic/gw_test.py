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
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import geometry, low_rank, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein


@pytest.mark.fast()
class TestQuadraticProblem:

  @pytest.mark.parametrize("as_pc", [False, True])
  @pytest.mark.parametrize("rank", [-1, 5, (1, 2, 3), (2, 3, 5)])
  def test_quad_to_low_rank(
      self, rng: jax.random.PRNGKeyArray, as_pc: bool,
      rank: Union[int, Tuple[int, ...]]
  ):
    n, m, d1, d2, d = 200, 300, 20, 25, 30
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    x = jax.random.normal(rng1, (n, d1))
    y = jax.random.normal(rng2, (m, d2))
    xx = jax.random.normal(rng3, (n, d))
    yy = jax.random.normal(rng4, (m, d))

    geom_xx = pointcloud.PointCloud(x)
    geom_yy = pointcloud.PointCloud(y)
    geom_xy = pointcloud.PointCloud(xx, yy)
    if not as_pc:
      geom_xx = geometry.Geometry(geom_xx.cost_matrix)
      geom_yy = geometry.Geometry(geom_yy.cost_matrix)
      geom_xy = geometry.Geometry(geom_xy.cost_matrix)

    prob = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, geom_xy, ranks=rank
    )
    assert not prob.is_low_rank

    # point clouds are always converted, if possible
    if not as_pc and rank == -1:
      with pytest.raises(AssertionError, match=r"Rank must"):
        _ = prob.to_low_rank()
      return
    lr_prob = prob.to_low_rank()
    geoms = lr_prob.geom_xx, lr_prob.geom_yy, lr_prob.geom_xy

    if rank == -1:
      if as_pc:
        assert lr_prob.is_low_rank
      else:
        assert not lr_prob.is_low_rank
    else:
      rank = (rank,) * 3 if isinstance(rank, int) else rank
      for r, actual_geom, expected_geom in zip(
          rank, geoms, [geom_xx, geom_yy, geom_xy]
      ):
        if r == -1:
          assert actual_geom is expected_geom
        else:
          assert isinstance(actual_geom, low_rank.LRCGeometry)
          if as_pc:
            assert actual_geom.cost_rank == expected_geom.x.shape[1] + 2
          else:
            assert actual_geom.cost_rank == r

      if -1 in rank:
        assert not lr_prob.is_low_rank
      else:
        assert lr_prob.is_low_rank
        assert lr_prob._is_low_rank_convertible
        assert lr_prob.to_low_rank() is lr_prob

  def test_gw_implicit_conversion_mixed_input(
      self, rng: jax.random.PRNGKeyArray
  ):
    n, m, d1, d2 = 200, 300, 20, 25
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (n, d1))
    y = jax.random.normal(rng2, (m, d2))

    geom_xx = pointcloud.PointCloud(x)
    geom_yy = pointcloud.PointCloud(y).to_LRCGeometry()

    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, ranks=-1)
    lr_prob = prob.to_low_rank()

    assert prob._is_low_rank_convertible
    assert lr_prob.is_low_rank
    assert prob.geom_yy is lr_prob.geom_yy


class TestGromovWasserstein:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    d_x = 2
    d_y = 3
    self.n, self.m = 6, 7
    rngs = jax.random.split(rng, 6)
    self.x = jax.random.uniform(rngs[0], (self.n, d_x))
    self.y = jax.random.uniform(rngs[1], (self.m, d_y))
    a = jax.random.uniform(rngs[2], (self.n,)) + 1e-1
    b = jax.random.uniform(rngs[3], (self.m,)) + 1e-1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(rngs[4], (self.n, self.n))
    self.cy = jax.random.uniform(rngs[5], (self.m, self.m))
    self.tau_a = 0.8
    self.tau_b = 0.9

  def test_flag_store_errors(self):
    """Tests whether errors are properly stored if requested."""
    threshold_sinkhorn = 1e-2
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b
    )

    solver = gromov_wasserstein.GromovWasserstein(
        epsilon=1e-1, store_inner_errors=False
    )
    assert solver(prob).errors is None

    solver = gromov_wasserstein.GromovWasserstein(
        epsilon=1e-1, store_inner_errors=True
    )
    errors = solver(prob).errors

    assert errors.ndim == 2
    errors = errors[jnp.sum(errors > 0, axis=1) > 0, :]
    last_errors = errors[-1, :]
    assert threshold_sinkhorn > last_errors[last_errors > -1][-1]

  @pytest.mark.parametrize("jit", [False, True])
  def test_gradient_marginals_gw(self, jit: bool):
    """Test gradient w.r.t. probability weights."""

    def reg_gw(a: jnp.ndarray, b: jnp.ndarray,
               implicit: bool) -> Tuple[float, Tuple[jnp.ndarray, jnp.ndarray]]:
      prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a, b=b)
      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      linear_solver = sinkhorn.Sinkhorn(
          implicit_diff=implicit_diff, max_iterations=1000
      )
      solver = gromov_wasserstein.GromovWasserstein(
          epsilon=1.0, max_iterations=10, linear_ot_solver=linear_solver
      )
      out = solver(prob)
      return out.reg_gw_cost, (out.linear_state.f, out.linear_state.g)

    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_grad = jax.grad(reg_gw, has_aux=True, argnums=(0, 1))
      if jit:
        reg_gw_grad = jax.jit(reg_gw_grad, static_argnames="implicit")

      grad_reg_gw, aux = reg_gw_grad(self.a, self.b, implicit)
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

  @pytest.mark.fast()
  @pytest.mark.parametrize(("balanced", "rank"), [(True, -1), (False, -1),
                                                  (True, 3)])
  def test_gw_pointcloud(self, balanced: bool, rank: int):
    """Test basic computations pointclouds."""
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    tau_a, tau_b = (1.0, 1.0) if balanced else (self.tau_a, self.tau_b)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b, tau_a=tau_a, tau_b=tau_b
    )
    solver = gromov_wasserstein.GromovWasserstein(
        rank=rank, epsilon=0.0 if rank > 0 else 1.0, max_iterations=10
    )

    out = solver(prob)
    # TODO(cuturi): test primal cost for un-balanced case as well.
    if balanced:
      u = geom_x.apply_square_cost(out.matrix.sum(axis=-1)).squeeze()
      v = geom_y.apply_square_cost(out.matrix.sum(axis=0)).squeeze()
      c = (geom_x.cost_matrix @ out.matrix) @ geom_y.cost_matrix
      c = (u[:, None] + v[None, :] - 2 * c)

      np.testing.assert_allclose(
          out.primal_cost, jnp.sum(c * out.matrix), rtol=1e-3
      )

    assert not jnp.isnan(out.reg_gw_cost)

  @pytest.mark.parametrize(("unbalanced", "unbalanced_correction"),
                           [(False, False), (True, False), (True, True)],
                           ids=["bal", "unbal-nocorr", "unbal-corr"])
  @pytest.mark.parametrize(("lse_mode", "is_cost"), [(True, False),
                                                     (False, True)],
                           ids=["lse-pc", "kernel-cost-mat"])
  def test_gradient_gw_geometry(
      self, lse_mode: bool, is_cost: bool, unbalanced: bool,
      unbalanced_correction: bool
  ):
    """Test gradient w.r.t. the geometries."""

    def reg_gw(
        x: jnp.ndarray, y: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray,
        implicit: bool
    ) -> float:
      if is_cost:
        geom_x = geometry.Geometry(cost_matrix=x)
        geom_y = geometry.Geometry(cost_matrix=y)
      else:
        geom_x = pointcloud.PointCloud(x)
        geom_y = pointcloud.PointCloud(y)
      tau_a, tau_b = (self.tau_a, self.tau_b) if unbalanced else (1.0, 1.0)
      prob = quadratic_problem.QuadraticProblem(
          geom_x,
          geom_y,
          a=a,
          b=b,
          tau_a=tau_a,
          tau_b=tau_b,
          gw_unbalanced_correction=unbalanced_correction
      )

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      lin_solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode, max_iterations=1000, implicit_diff=implicit_diff
      )
      solver = gromov_wasserstein.GromovWasserstein(
          epsilon=1.0, max_iterations=10, linear_ot_solver=lin_solver
      )

      return solver(prob).reg_gw_cost

    grad_matrices = [None, None]
    x, y = (self.cx, self.cy) if is_cost else (self.x, self.y)
    reg_gw_grad = jax.grad(reg_gw, argnums=(0, 1))

    for i, implicit in enumerate([True, False]):
      grad_matrices[i] = reg_gw_grad(x, y, self.a, self.b, implicit)
      assert not jnp.any(jnp.isnan(grad_matrices[i][0]))
      assert not jnp.any(jnp.isnan(grad_matrices[i][1]))

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-02, atol=1e-02
    )

  def test_gw_adaptive_threshold(self):
    """Checking solution is improved with smaller threshold for convergence."""
    geom_x = pointcloud.PointCloud(self.x, self.x)
    geom_y = pointcloud.PointCloud(self.y, self.y)

    def loss_thre(threshold: float) -> float:
      prob = quadratic_problem.QuadraticProblem(
          geom_x, geom_y, a=self.a, b=self.b
      )
      solver = gromov_wasserstein.GromovWasserstein(
          threshold=threshold, epsilon=1e-1
      )

      return solver(prob).reg_gw_cost

    assert loss_thre(1e-1) >= loss_thre(1e-4)
    assert loss_thre(1e-3) >= loss_thre(1e-5)

  @pytest.mark.fast()
  def test_gw_lr(self, rng: jax.random.PRNGKeyArray):
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
    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a=a, b=b)
    solver = gromov_wasserstein.GromovWasserstein(rank=5, epsilon=0.2)
    ot_gwlr = solver(prob)
    solver = gromov_wasserstein.GromovWasserstein(epsilon=0.2)
    ot_gw = solver(prob)
    np.testing.assert_allclose(ot_gwlr.costs, ot_gw.costs, rtol=5e-2)

  def test_gw_lr_matches_fused(self, rng: jax.random.PRNGKeyArray):
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
    prob = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, geom_xy=geom_xy, fused_penalty=1.3, a=a, b=b
    )
    solver = gromov_wasserstein.GromovWasserstein(rank=6)
    ot_gwlr = solver(prob)
    solver = gromov_wasserstein.GromovWasserstein(rank=6, epsilon=1e-1)
    ot_gwlreps = solver(prob)
    solver = gromov_wasserstein.GromovWasserstein(epsilon=5e-2)
    ot_gw = solver(prob)

    # Test solutions look alike
    assert jnp.linalg.norm(ot_gwlr.matrix - ot_gw.matrix) < 0.11
    assert jnp.linalg.norm(ot_gwlr.matrix - ot_gwlreps.matrix) < 0.15
    # Test at least some difference when adding bigger entropic regularization
    assert jnp.linalg.norm(ot_gwlr.matrix - ot_gwlreps.matrix) > 1e-3

  @pytest.mark.parametrize("axis", [0, 1])
  def test_gw_lr_apply(self, axis: int):
    geom_x = pointcloud.PointCloud(self.x)
    geom_y = pointcloud.PointCloud(self.y)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b
    )
    solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-1, rank=2)
    out = solver(prob)

    arr, matrix = (self.x, out.matrix) if axis == 0 else (self.y, out.matrix.T)
    res_apply = out.apply(arr.T, axis=axis)
    res_matrix = arr.T @ matrix

    np.testing.assert_allclose(res_apply, res_matrix, rtol=1e-5, atol=1e-5)

  def test_gw_lr_warm_start_helps(self, rng: jax.random.PRNGKeyArray):
    rank = 3
    rng1, rng2 = jax.random.split(rng, 2)
    geom_x = pointcloud.PointCloud(jax.random.normal(rng1, (100, 5)))
    geom_y = pointcloud.PointCloud(jax.random.normal(rng2, (110, 6)))
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y)

    solver_cold = gromov_wasserstein.GromovWasserstein(
        rank=rank, warm_start=False
    )
    solver_warm = gromov_wasserstein.GromovWasserstein(
        rank=rank, warm_start=True
    )

    out_cold = solver_cold(prob)
    out_warm = solver_warm(prob)

    cost = out_cold.reg_gw_cost
    cost_warm_start = out_warm.reg_gw_cost
    assert (cost_warm_start + 5.0) < cost
    with pytest.raises(AssertionError):
      np.testing.assert_allclose(out_cold.matrix, out_warm.matrix)
