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

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott import utils
from ott.geometry import geometry, low_rank, pointcloud
from ott.math import utils as mu
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from tests import _utils


@pytest.mark.fast()
class TestQuadraticProblem:

  @pytest.mark.parametrize("as_pc", [False, True])
  @pytest.mark.parametrize("rank", [-1, 5, (1, 2, 3), (2, 3, 5)])
  def test_quad_to_low_rank(
      self, clouds: _utils.QuadClouds, rng: jax.Array, as_pc: bool,
      rank: Union[int, Tuple[int, ...]]
  ):
    n, m, d1, d2, d = 100, 120, 4, 6, 10
    rng1, rng2, rng3, rng4 = jr.split(rng, 4)
    x = jr.normal(rng1, (n, d1))
    y = jr.normal(rng2, (m, d2))
    xx = jr.normal(rng3, (n, d))
    yy = jr.normal(rng4, (m, d))

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

  def test_to_low_rank_small_point_clouds(self, rng: jax.Array):
    geom = pointcloud.PointCloud(jr.normal(rng, (5, 10)))
    prob = quadratic_problem.QuadraticProblem(geom, geom)

    assert not prob._is_low_rank_convertible
    assert prob.to_low_rank().is_low_rank

  def test_update_lr_geom_fused_penalty(self, rng: jax.Array):
    rng_x, rng_y, rng_z = jr.split(rng, 3)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 3))
    z = jr.normal(rng_z, (12, 2))
    prob = quadratic_problem.QuadraticProblem(
        pointcloud.PointCloud(x),
        pointcloud.PointCloud(y),
        geom_xy=pointcloud.PointCloud(x, z),
        fused_penalty=3.0,
    )
    lr_sink = sinkhorn_lr.LRSinkhorn(rank=2)(
        linear_problem.LinearProblem(prob.geom_xy)
    )

    geom = prob.update_lr_geom(lr_sink)
    geom_lr = prob.to_low_rank().update_lr_geom(lr_sink)

    np.testing.assert_allclose(
        geom_lr.cost_matrix, geom.cost_matrix, rtol=1e-5, atol=1e-5
    )

  def test_gw_implicit_conversion_mixed_input(self, rng: jax.Array):
    n, m, d1, d2 = 13, 77, 3, 4
    rng1, rng2 = jr.split(rng, 2)
    x = jr.normal(rng1, (n, d1))
    y = jr.normal(rng2, (m, d2))

    geom_xx = pointcloud.PointCloud(x)
    geom_yy = pointcloud.PointCloud(y).to_LRCGeometry()

    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, ranks=-1)
    lr_prob = prob.to_low_rank()

    assert prob._is_low_rank_convertible
    assert lr_prob.is_low_rank
    assert prob.geom_yy is lr_prob.geom_yy


TAU_A, TAU_B = 0.8, 0.9


@pytest.fixture(scope="module")
def clouds() -> _utils.QuadClouds:
  """Clouds in different ambient dimensions."""
  n, m, d_x, d_y = 6, 7, 2, 3
  rngs = jr.split(jr.key(0), 6)
  return _utils.QuadClouds(
      x=jr.uniform(rngs[0], (n, d_x)),
      y=jr.uniform(rngs[1], (m, d_y)),
      a=_utils.random_weights(rngs[2], n),
      b=_utils.random_weights(rngs[3], m),
      cx=jr.uniform(rngs[4], (n, n)),
      cy=jr.uniform(rngs[5], (m, m)),
  )


class TestGromovWasserstein:

  def test_flag_store_errors(self, clouds: _utils.QuadClouds):
    """Tests whether errors are properly stored if requested."""
    threshold_sinkhorn = 1e-2
    prob = clouds.quad_problem()

    linear_solver = sinkhorn.Sinkhorn()
    solver = gromov_wasserstein.GromovWasserstein(
        linear_solver, epsilon=1e-1, store_inner_errors=False
    )
    np.testing.assert_equal(solver(prob).errors, None)

    solver = gromov_wasserstein.GromovWasserstein(
        linear_solver, epsilon=1e-1, store_inner_errors=True
    )
    out = solver(prob)
    assert 1 < out.n_iters < 12
    errors = out.errors

    np.testing.assert_array_equal(errors.ndim, 2)
    errors = errors[jnp.sum(errors > 0, axis=1) > 0, :]
    last_errors = errors[-1, :]
    np.testing.assert_array_less(
        last_errors[last_errors > -1][-1], threshold_sinkhorn
    )

  @pytest.mark.parametrize("jit", [False, True])
  def test_gradient_marginals_gw(self, clouds: _utils.QuadClouds, jit: bool):
    """Test gradient w.r.t. probability weights."""

    def reg_gw(a: jnp.ndarray, b: jnp.ndarray,
               implicit: bool) -> Tuple[float, Tuple[jnp.ndarray, jnp.ndarray]]:
      prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a, b=b)
      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      linear_solver = sinkhorn.Sinkhorn(
          implicit_diff=implicit_diff, max_iterations=1000
      )
      solver = gromov_wasserstein.GromovWasserstein(
          linear_solver,
          epsilon=1.0,
          max_iterations=10,
      )
      out = solver(prob)
      return out.reg_gw_cost, (out.linear_state.f, out.linear_state.g)

    geom_x = pointcloud.PointCloud(clouds.x)
    geom_y = pointcloud.PointCloud(clouds.y)

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_gw_grad = jax.grad(reg_gw, has_aux=True, argnums=(0, 1))
      if jit:
        reg_gw_grad = jax.jit(reg_gw_grad, static_argnames="implicit")

      grad_reg_gw, aux = reg_gw_grad(clouds.a, clouds.b, implicit)
      grad_matrices[i] = grad_reg_gw
      grad_manual_a = aux[0] - jnp.log(clouds.a)
      grad_manual_b = aux[1] - jnp.log(clouds.b)
      assert not jnp.any(jnp.isnan(grad_reg_gw[0]))
      assert not jnp.any(jnp.isnan(grad_reg_gw[1]))
      np.testing.assert_allclose(
          grad_manual_a, grad_reg_gw[0], rtol=1e-2, atol=1e-2
      )
      np.testing.assert_allclose(
          grad_manual_b, grad_reg_gw[1], rtol=1e-2, atol=1e-2
      )

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-2, atol=1e-2
    )

  @pytest.mark.fast()
  @pytest.mark.parametrize(("balanced", "rank"), [(True, -1), (False, -1),
                                                  (True, 3)])
  def test_gw_pointcloud(
      self, clouds: _utils.QuadClouds, balanced: bool, rank: int
  ):
    """Test basic computations point clouds."""
    tau_a, tau_b = (1.0, 1.0) if balanced else (TAU_A, TAU_B)
    prob = clouds.quad_problem(tau_a=tau_a, tau_b=tau_b)
    if rank > 0:
      solver = gromov_wasserstein_lr.LRGromovWasserstein(
          rank=rank,
          epsilon=0.0,
          max_iterations=10,
      )
    else:
      linear_solver = sinkhorn.Sinkhorn()
      solver = gromov_wasserstein.GromovWasserstein(
          linear_solver, epsilon=1.0, max_iterations=10
      )

    out = solver(prob)
    # TODO(cuturi): test primal cost for un-balanced case as well.
    if balanced:
      geom_x, geom_y = prob.geom_xx, prob.geom_yy
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
      self, clouds: _utils.QuadClouds, lse_mode: bool, is_cost: bool,
      unbalanced: bool, unbalanced_correction: bool
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
      tau_a, tau_b = (TAU_A, TAU_B) if unbalanced else (1.0, 1.0)
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
      linear_solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode, max_iterations=1000, implicit_diff=implicit_diff
      )
      solver = gromov_wasserstein.GromovWasserstein(
          linear_solver,
          epsilon=1.0,
          max_iterations=10,
      )

      return solver(prob).reg_gw_cost

    grad_matrices = [None, None]
    x, y = (clouds.cx, clouds.cy) if is_cost else (clouds.x, clouds.y)
    reg_gw_grad = jax.grad(reg_gw, argnums=(0, 1))

    for i, implicit in enumerate([True, False]):
      grad_matrices[i] = reg_gw_grad(x, y, clouds.a, clouds.b, implicit)
      assert not jnp.any(jnp.isnan(grad_matrices[i][0]))
      assert not jnp.any(jnp.isnan(grad_matrices[i][1]))

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        grad_matrices[0][1], grad_matrices[1][1], rtol=1e-2, atol=1e-2
    )

  @pytest.mark.parametrize("fused", [False, True])
  def test_reg_gw_cost(self, rng: jax.Array, fused: bool):

    def solve(a: jnp.ndarray) -> gromov_wasserstein.GWOutput:
      prob = quadratic_problem.QuadraticProblem(
          geometry.Geometry(cost_matrix=cx),
          geometry.Geometry(cost_matrix=cy),
          geom_xy if fused else None,
          a=a,
      )
      return solver(prob)

    rngs = jr.split(rng, 4)
    x, y = jr.normal(rngs[0], (7, 2)), jr.normal(rngs[1], (8, 3))
    cx = pointcloud.PointCloud(x).cost_matrix
    cy = pointcloud.PointCloud(y).cost_matrix
    geom_xy = pointcloud.PointCloud(
        jr.normal(rngs[2], (7, 2)), jr.normal(rngs[3], (8, 2))
    )
    solver = gromov_wasserstein.GromovWasserstein(
        sinkhorn.Sinkhorn(threshold=1e-6, max_iterations=5000),
        epsilon=5.0,
        threshold=1e-6,
        min_iterations=20,
        max_iterations=200,
    )
    a, h = jnp.full(7, 1.0 / 7), 1e-2
    d = jnp.zeros(7).at[0].set(1.0).at[1].set(-1.0)

    out = solve(a)
    p, ab = out.matrix, jnp.outer(a, jnp.full(8, 1.0 / 8))
    loss = (cx[:, None, :, None] - cy[None, :, None, :]) ** 2
    quad = jnp.einsum("ijkl,ij,kl->", loss, p, p)
    lin = jnp.vdot(geom_xy.cost_matrix, p) if fused else 0.0
    grad = jnp.vdot(jax.grad(lambda a: solve(a).reg_gw_cost)(a), d)
    grad_fd = (solve(a + h * d).reg_gw_cost -
               solve(a - h * d).reg_gw_cost) / (2.0 * h)

    np.testing.assert_allclose(
        out.reg_gw_cost, quad + lin + 5.0 * mu.gen_kl(p, ab), rtol=1e-5
    )
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-2)

  def test_gw_adaptive_threshold(self, clouds: _utils.QuadClouds):
    """Checking solution is improved with smaller threshold for convergence."""
    geom_x = pointcloud.PointCloud(clouds.x, clouds.x)
    geom_y = pointcloud.PointCloud(clouds.y, clouds.y)

    def loss_thre(threshold: float) -> float:
      prob = quadratic_problem.QuadraticProblem(
          geom_x, geom_y, a=clouds.a, b=clouds.b
      )
      linear_solver = sinkhorn.Sinkhorn()
      solver = gromov_wasserstein.GromovWasserstein(
          linear_solver, threshold=threshold, epsilon=1e-1
      )
      out = solver(prob)

      ab = jnp.outer(clouds.a, clouds.b)
      return out.primal_cost + 1e-1 * mu.gen_kl(out.matrix, ab)

    assert loss_thre(1e-1) > loss_thre(1e-5)

  @pytest.mark.usefixtures("enable_x64")
  def test_unbalanced_split_point(self, rng: jax.Array):
    rng_x, rng_y = jr.split(rng, 2)
    x, y = jr.normal(rng_x, (6, 2)), jr.normal(rng_y, (7, 3))
    a, b = jnp.full(6, 1.0 / 6), jnp.full(7, 1.0 / 7)
    x_split = jnp.concatenate([x[:1], x])
    a_split = jnp.concatenate([a[:1] / 2, a[:1] / 2, a[1:]])

    def reg_gw_cost(x: jnp.ndarray, a: jnp.ndarray) -> float:
      prob = quadratic_problem.QuadraticProblem(
          pointcloud.PointCloud(x),
          pointcloud.PointCloud(y),
          a=a,
          b=b,
          tau_a=0.8,
          tau_b=0.9
      )
      solver = gromov_wasserstein.GromovWasserstein(
          sinkhorn.Sinkhorn(threshold=1e-10), epsilon=1.0
      )
      return solver(prob).reg_gw_cost

    np.testing.assert_allclose(
        reg_gw_cost(x_split, a_split), reg_gw_cost(x, a), rtol=1e-6
    )

  @pytest.mark.fast()
  def test_gw_lr(self, rng: jax.Array):
    """Checking LR and Entropic have similar outputs on same problem."""
    rngs = jr.split(rng, 4)
    n, m, d1, d2 = 24, 17, 2, 3
    x = jr.uniform(rngs[0], (n, d1))
    y = jr.uniform(rngs[1], (m, d2))
    a = jr.uniform(rngs[2], (n,))
    b = jr.uniform(rngs[3], (m,))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    geom_xx = pointcloud.PointCloud(x)
    geom_yy = pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, a=a, b=b)

    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=5,
        epsilon=0.2,
        min_iterations=0,
        inner_iterations=10,
        max_iterations=2000
    )
    ot_gwlr = solver(prob)
    linear_solver = sinkhorn.Sinkhorn()
    solver = gromov_wasserstein.GromovWasserstein(linear_solver, epsilon=0.4)
    ot_gw = solver(prob)

    np.testing.assert_allclose(
        ot_gwlr.primal_cost, ot_gw.primal_cost, rtol=5e-2
    )

  def test_gw_lr_matches_fused(self, rng: jax.Array):
    """Checking LR and Entropic have similar outputs on same fused problem."""
    rngs = jr.split(rng, 5)
    n, m, d1, d2 = 24, 17, 2, 3
    x = jr.uniform(rngs[0], (n, d1))
    y = jr.uniform(rngs[1], (m, d2))
    a = jr.uniform(rngs[2], (n,))
    b = jr.uniform(rngs[3], (m,))
    z = jr.uniform(rngs[4], (m, d1))
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    geom_xx = pointcloud.PointCloud(x)
    geom_yy = pointcloud.PointCloud(y)
    geom_xy = pointcloud.PointCloud(x, z)  # only used to compute n x m matrix
    prob = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, geom_xy=geom_xy, fused_penalty=1.3, a=a, b=b
    )

    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=6, min_iterations=0, inner_iterations=10, max_iterations=2000
    )
    ot_gwlr = solver(prob)
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=6,
        epsilon=1e-1,
        min_iterations=0,
        inner_iterations=10,
        max_iterations=2000
    )
    ot_gwlreps = solver(prob)
    linear_solver = sinkhorn.Sinkhorn()
    solver = gromov_wasserstein.GromovWasserstein(linear_solver, epsilon=5e-2)
    ot_gw = solver(prob)

    # Test solutions look alike
    assert jnp.linalg.norm(ot_gwlr.matrix - ot_gw.matrix) < 0.11
    assert jnp.linalg.norm(ot_gwlr.matrix - ot_gwlreps.matrix) < 0.15
    # Test at least some difference when adding bigger entropic regularization
    assert jnp.linalg.norm(ot_gwlr.matrix - ot_gwlreps.matrix) > 1e-3

  @pytest.mark.parametrize("axis", [0, 1])
  def test_gw_lr_apply(self, clouds: _utils.QuadClouds, axis: int):
    prob = clouds.quad_problem()
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=2,
        epsilon=1e-1,
        min_iterations=0,
        inner_iterations=10,
        max_iterations=2000
    )
    out = solver(prob)

    arr, matrix = (clouds.x,
                   out.matrix) if axis == 0 else (clouds.y, out.matrix.T)
    res_apply = out.apply(arr.T, axis=axis)
    res_matrix = arr.T @ matrix

    np.testing.assert_allclose(res_apply, res_matrix, rtol=1e-5, atol=1e-5)

  def test_gw_lr_kl_loss(self, rng: jax.Array):
    rng_x, rng_y = jr.split(rng, 2)
    x, y = jr.uniform(rng_x, (10, 2)), jr.uniform(rng_y, (12, 3))
    prob = quadratic_problem.QuadraticProblem(
        pointcloud.PointCloud(x), pointcloud.PointCloud(y), loss="kl"
    )
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=2, min_iterations=0, inner_iterations=10, max_iterations=100
    )

    with pytest.raises(AssertionError, match=r"squared Euclidean"):
      _ = solver(prob)

  def test_gw_lr_unbalanced_correction(self, rng: jax.Array):
    rng_x, rng_y = jr.split(rng, 2)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 3))
    geom_xx, geom_yy = pointcloud.PointCloud(x), pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, tau_a=0.9, tau_b=0.9
    )
    prob_no_corr = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, tau_a=0.9, tau_b=0.9, gw_unbalanced_correction=False
    )
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=2, min_iterations=0, inner_iterations=10, max_iterations=100
    )

    out = solver(prob)
    out_no_corr = solver(prob_no_corr)

    np.testing.assert_allclose(
        out_no_corr.reg_gw_cost, out.reg_gw_cost, rtol=1e-6
    )

  @pytest.mark.parametrize("fused_penalty", [1.0, 3.0])
  def test_gw_lr_fused_costs(self, rng: jax.Array, fused_penalty: float):

    def gw_cost(q: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray) -> float:
      p = (q / g) @ r.T
      quad = (cx ** 2 @ a) @ a + (cy ** 2 @ b) @ b
      quad -= 2.0 * jnp.vdot(cx @ p @ cy, p)
      return quad + fused_penalty * g.sum() * jnp.vdot(cxy, p)

    rng_x, rng_y, rng_z = jr.split(rng, 3)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 3))
    z = jr.normal(rng_z, (12, 2))
    prob = quadratic_problem.QuadraticProblem(
        pointcloud.PointCloud(x),
        pointcloud.PointCloud(y),
        geom_xy=pointcloud.PointCloud(x, z),
        fused_penalty=fused_penalty,
    )
    cx, cy = prob.geom_xx.cost_matrix, prob.geom_yy.cost_matrix
    cxy, a, b = prob.geom_xy.cost_matrix, prob.a, prob.b
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=2, gamma=1.0, gamma_rescale=False
    )
    q, r, g = solver.initializer(prob)
    state = solver.init_state(prob, (q, r, g))

    out = solver.output_from_state(prob, state)
    costs = solver._get_costs(prob, state)[:3]
    grads = jax.grad(gw_cost, argnums=(0, 1, 2))(q, r, g)

    np.testing.assert_allclose(out.primal_cost, gw_cost(q, r, g), rtol=1e-4)
    for arr, cost, grad in zip((q, r, g), costs, grads):
      np.testing.assert_allclose(
          jnp.log(arr) - cost, grad, rtol=1e-4, atol=1e-4
      )

  @pytest.mark.parametrize("loss", ["sqeucl", "kl"])
  def test_gw_lr_loss_costs(self, rng: jax.Array, loss: str):

    def gw_cost(q: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray) -> float:
      p = (q / g) @ r.T
      mu, nu = q.sum(1), r.sum(1)
      quad = jnp.einsum("ik,jl,ij,kl->", h1.func(cx), h2.func(cy), p, p)
      return mu @ f1.func(cx) @ mu + nu @ f2.func(cy) @ nu - quad

    rng_x, rng_y = jr.split(rng, 2)
    cx, cy = jr.uniform(rng_x, (10, 10)), jr.uniform(rng_y, (12, 12))
    cx, cy = cx + cx.T, cy + cy.T
    prob = quadratic_problem.QuadraticProblem(
        geometry.Geometry(cost_matrix=cx),
        geometry.Geometry(cost_matrix=cy),
        loss=loss,
        tau_a=0.9,
        tau_b=0.9,
    )
    (f1, f2), (h1, h2) = prob.linear_loss, prob.quad_loss
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=2, gamma=1.0, gamma_rescale=False
    )
    q, r, g = solver.initializer(prob)
    state = solver.init_state(prob, (q, r, g))

    out = solver.output_from_state(prob, state)
    costs = solver._get_costs(prob, state)[:3]
    grads = jax.grad(gw_cost, argnums=(0, 1, 2))(q, r, g)

    np.testing.assert_allclose(out.primal_cost, gw_cost(q, r, g), rtol=1e-5)
    for x, cost, grad in zip((q, r, g), costs, grads):
      np.testing.assert_allclose(jnp.log(x) - cost, grad, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_gw_lr_zero_weights_unrolled_grad(
      self, rng: jax.Array, lse_mode: bool
  ):

    def loss(x: jnp.ndarray, n_pad: int) -> float:
      x = jnp.concatenate([x, jnp.full((n_pad, 2), 5.0)])
      a = jnp.concatenate([jnp.full(10, 0.1), jnp.zeros(n_pad)])
      q_pad = jnp.concatenate([q, jnp.zeros((n_pad, 2))])
      prob = quadratic_problem.QuadraticProblem(
          pointcloud.PointCloud(x), pointcloud.PointCloud(y), a=a
      )
      return solver(prob, init=(q_pad, r, g)).reg_gw_cost

    rng_x, rng_y = jr.split(rng, 2)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 3))
    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=2,
        epsilon=1e-1,
        lse_mode=lse_mode,
        use_danskin=False,
        min_iterations=0,
        inner_iterations=10,
        max_iterations=200,
    )
    q, r, g = solver.initializer(
        quadratic_problem.QuadraticProblem(
            pointcloud.PointCloud(x), pointcloud.PointCloud(y)
        )
    )

    grad = jax.grad(loss)(x, 0)
    grad_pad = jax.grad(loss)(x, 3)

    np.testing.assert_allclose(grad_pad, grad, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("scale_cost", [1.0, "mean"])
  def test_relative_epsilon(
      self,
      rng: jax.Array,
      scale_cost: Union[float, str],
  ):
    eps = 1e-2
    rng1, rng2 = jr.split(rng, 2)
    geom_x = pointcloud.PointCloud(
        jr.normal(rng1, (49, 5)), scale_cost=scale_cost
    )
    geom_y = pointcloud.PointCloud(
        jr.normal(rng2, (78, 6)), scale_cost=scale_cost
    )
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y)

    linear_solver = sinkhorn.Sinkhorn()
    solver = gromov_wasserstein.GromovWasserstein(
        linear_solver,
        epsilon=eps,
        relative_epsilon="std",
    )

    out = solver(prob)

    if scale_cost == 1.0:
      assert out.reg_gw_cost < 37
      assert out.primal_cost < 32
    else:
      assert out.reg_gw_cost < 0.25
      assert out.primal_cost < 0.22

  @pytest.mark.parametrize(("tau_a", "tau_b", "eps", "ti"),
                           [(0.99, 0.95, 0.0, True), (0.9, 0.8, 1e-3, False),
                            (1.0, 0.999, 0.0, True), (0.5, 1.0, 1e-2, False)])
  def test_gwlr_unbalanced(
      self, clouds: _utils.QuadClouds, tau_a: float, tau_b: float, eps: float,
      ti: bool
  ):
    geom_x = pointcloud.PointCloud(clouds.x)
    geom_y = pointcloud.PointCloud(clouds.y)
    a = clouds.a.at[:2].set(0.0)
    b = clouds.b.at[15:20].set(0.0)
    prob = quadratic_problem.QuadraticProblem(
        geom_x,
        geom_y,
        a=a,
        b=b,
        tau_a=tau_a,
        tau_b=tau_b,
    )
    solver = jax.jit(
        gromov_wasserstein_lr.LRGromovWasserstein(
            rank=4,
            epsilon=eps,
            kwargs_dys={"translation_invariant": ti},
            min_iterations=0,
            inner_iterations=10,
            max_iterations=2000
        )
    )

    res = solver(prob)

    np.testing.assert_array_equal(jnp.isfinite(res.errors), True)
    np.testing.assert_array_equal(jnp.isfinite(res.costs), True)

  @pytest.mark.parametrize(("rank", "eps"), [(5, 0.0), (10, 1e-3), (15, 1e-2)])
  @pytest.mark.usefixtures("enable_x64")
  def test_gwlr_unbalanced_matches_balanced(
      self, clouds: _utils.QuadClouds, rank: int, eps: float
  ):

    geom_x = pointcloud.PointCloud(clouds.x)
    geom_y = pointcloud.PointCloud(clouds.y)
    prob = quadratic_problem.QuadraticProblem(
        geom_x,
        geom_y,
        a=clouds.a,
        b=clouds.b,
        tau_a=1.0,
        tau_b=1.0,
    )
    prob_unbal = quadratic_problem.QuadraticProblem(
        geom_x,
        geom_y,
        a=clouds.a,
        b=clouds.b,
        tau_a=0.9999,
        tau_b=0.9999,
    )
    solver = jax.jit(
        gromov_wasserstein_lr.LRGromovWasserstein(
            rank=rank,
            epsilon=eps,
            max_iterations=100,
        )
    )

    res = solver(prob)
    res_unbal = solver(prob_unbal)

    np.testing.assert_allclose(res.transport_mass, 1.0, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        res.transport_mass, res_unbal.transport_mass, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        res.primal_cost, res_unbal.primal_cost, rtol=1e-2, atol=1e-2
    )

  @pytest.mark.parametrize("grad", [False, True])
  def test_gw_progress_fn(self, clouds: _utils.QuadClouds, grad: bool):

    def callback(x: jnp.ndarray, y: jnp.ndarray):
      geom_xx = pointcloud.PointCloud(x)
      geom_yy = pointcloud.PointCloud(y)
      prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy)

      # TODO(michalk8): passing `progress_fn` in linear/quadratic solver
      # raises: ValueError: Reverse-mode differentiation does not work for
      # lax.while_loop or lax.fori_loop with dynamic start/stop values.
      linear_solver = sinkhorn.Sinkhorn(progress_fn=utils.default_progress_fn())
      quad_solver = gromov_wasserstein.GromovWasserstein(
          linear_solver,
          progress_fn=utils.default_progress_fn(),
          min_iterations=5,
          max_iterations=5,
          # needs to be explicitly set
          store_inner_errors=True,
      )

      return quad_solver(prob).reg_gw_cost

    fn = jax.grad(callback) if grad else callback
    fn = jax.jit(fn)
    res = fn(clouds.x, clouds.y)

    np.testing.assert_array_equal(jnp.isfinite(res), True)
