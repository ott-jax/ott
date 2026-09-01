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
import dataclasses
from typing import Literal, Tuple, Union

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import geometry, low_rank, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers import quadratic
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from tests import _utils

FUSED_PENALTY = 2.0


@dataclasses.dataclass(frozen=True)
class FusedClouds(_utils.QuadClouds):
  """:class:`~tests._utils.QuadClouds` plus the fused, inter-domain data."""
  x_2: jnp.ndarray  # (n, d_xy), source of the fused term
  y_2: jnp.ndarray  # (m, d_xy), target of the fused term
  cxy: jnp.ndarray  # (n, m), inter-domain cost


@pytest.fixture(scope="module")
def clouds() -> FusedClouds:
  """Clouds drawn exactly as this module always has."""
  n, m, d_x, d_y, d_xy = 5, 6, 2, 3, 4
  rngs = jr.split(_utils.root_key(), 9)
  return FusedClouds(
      x=jr.uniform(rngs[0], (n, d_x)),
      y=jr.uniform(rngs[1], (m, d_y)),
      a=_utils.random_probs(rngs[2], n),
      b=_utils.random_probs(rngs[3], m),
      cx=jr.uniform(rngs[4], (n, n)),
      cy=jr.uniform(rngs[5], (m, m)),
      x_2=jr.uniform(rngs[7], (n, d_xy)),
      y_2=jr.uniform(rngs[8], (m, d_xy)),
      cxy=jr.uniform(rngs[6], (n, m)),
  )


class TestFusedGromovWasserstein:

  @pytest.mark.fast.with_args("jit", [False, True], only_fast=0)
  def test_gradient_marginals_fgw_solver(self, clouds: FusedClouds, jit: bool):
    """Test gradient w.r.t. probability weights."""
    geom_x = pointcloud.PointCloud(clouds.x)
    geom_y = pointcloud.PointCloud(clouds.y)
    geom_xy = pointcloud.PointCloud(clouds.x_2, clouds.y_2)

    def reg_gw(a: jnp.ndarray, b: jnp.ndarray, implicit: bool):
      prob = quadratic_problem.QuadraticProblem(
          geom_x, geom_y, geom_xy, fused_penalty=FUSED_PENALTY, a=a, b=b
      )

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      linear_solver = sinkhorn.Sinkhorn(
          implicit_diff=implicit_diff, max_iterations=1000
      )
      solver = gromov_wasserstein.GromovWasserstein(linear_solver, epsilon=1.0)

      out = solver(prob)

      return out.reg_gw_cost, (out.linear_state.f, out.linear_state.g)

    grad_matrices = [None, None]
    reg_fgw_grad = jax.grad(reg_gw, has_aux=True, argnums=(0, 1))
    if jit:
      reg_fgw_grad = jax.jit(reg_fgw_grad, static_argnames="implicit")

    for i, implicit in enumerate([True, False]):
      (g_a, g_b), aux = reg_fgw_grad(clouds.a, clouds.b, implicit)
      grad_matrices[i] = (g_a, g_b)
      grad_manual_a = aux[0] - jnp.log(clouds.a)
      grad_manual_b = aux[1] - jnp.log(clouds.b)
      assert not jnp.any(jnp.isnan(g_a))
      assert not jnp.any(jnp.isnan(g_b))
      np.testing.assert_allclose(grad_manual_a, g_a, rtol=1e-2, atol=1e-2)
      np.testing.assert_allclose(grad_manual_b, g_b, rtol=1e-2, atol=1e-2)

    gi_a, gi_b = grad_matrices[0]
    g_a, g_b = grad_matrices[1]

    np.testing.assert_allclose(g_a, gi_a, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(g_b, gi_b, rtol=1e-2, atol=1e-2)

  @pytest.mark.parametrize(("lse_mode", "is_cost"), [(True, False),
                                                     (False, True)],
                           ids=["lse-pc", "kernel-cost-mat"])
  def test_gradient_fgw_solver_geometry(
      self, clouds: FusedClouds, lse_mode: bool, is_cost: bool
  ):
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
          linear_solver, epsilon=1.0, max_iterations=10
      )

      return solver(prob).reg_gw_cost

    if is_cost:
      x, y, xy = clouds.cx, clouds.cy, clouds.cxy
    else:
      x, y, xy = clouds.x, clouds.y, (clouds.x_2, clouds.y_2)
    grad_matrices = [None, None]
    reg_fgw_grad = jax.grad(reg_gw, argnums=(0, 1, 2))

    for i, implicit in enumerate([True, False]):
      grad_matrices[i] = reg_fgw_grad(
          x, y, xy, FUSED_PENALTY, clouds.a, clouds.b, implicit
      )
      assert not jnp.any(jnp.isnan(grad_matrices[i][0]))
      assert not jnp.any(jnp.isnan(grad_matrices[i][1]))

    gi_x, gi_y, gi_xy = grad_matrices[0]
    g_x, g_y, g_xy = grad_matrices[1]

    np.testing.assert_allclose(g_x, gi_x, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(g_y, gi_y, rtol=1e-2, atol=1e-2)
    if is_cost:
      np.testing.assert_allclose(g_xy, gi_xy, rtol=1e-2, atol=1e-2)
    else:
      np.testing.assert_allclose(g_xy[0], gi_xy[0], rtol=1e-2, atol=1e-2)
      np.testing.assert_allclose(g_xy[1], gi_xy[1], rtol=1e-2, atol=1e-2)

  def test_fgw_adaptive_threshold(self, clouds: FusedClouds):
    """Checking solution is improved with smaller threshold for convergence."""
    geom_x = pointcloud.PointCloud(clouds.x, clouds.x)
    geom_y = pointcloud.PointCloud(clouds.y, clouds.y)
    geom_xy = pointcloud.PointCloud(clouds.x_2, clouds.y_2)

    # without warm start for calls to sinkhorn
    def loss_thre(threshold: float) -> float:
      prob = quadratic_problem.QuadraticProblem(
          geom_x, geom_y, geom_xy, a=clouds.a, b=clouds.b, fused_penalty=0.05
      )
      linear_solver = sinkhorn.Sinkhorn()
      solver = gromov_wasserstein.GromovWasserstein(
          linear_solver, threshold=threshold, epsilon=1e-1
      )

      return solver(prob).reg_gw_cost

    assert loss_thre(1e-3) > loss_thre(1e-5)

  def test_gradient_fgw_solver_penalty(self, clouds: FusedClouds):
    """Test gradient w.r.t. penalty."""

    lse_mode = True

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
          lse_mode=lse_mode, implicit_diff=implicit_diff, max_iterations=200
      )
      solver = gromov_wasserstein.GromovWasserstein(
          linear_solver,
          epsilon=1.0,
          max_iterations=10,
      )
      return solver(prob).reg_gw_cost

    grad_matrices = [None, None]
    for i, implicit in enumerate([True, False]):
      reg_fgw_grad = jax.grad(reg_gw, argnums=(3,))
      grad_matrices[i] = reg_fgw_grad(
          clouds.cx, clouds.cy, clouds.cxy, FUSED_PENALTY, clouds.a, clouds.b,
          implicit
      )
      assert not jnp.any(jnp.isnan(grad_matrices[i][0]))

    np.testing.assert_allclose(
        grad_matrices[0][0], grad_matrices[1][0], rtol=1e-2, atol=1e-2
    )

  @pytest.mark.limit_memory("250 MB")
  @pytest.mark.parametrize("jit", [False, True])
  def test_fgw_lr_memory(self, rng: jax.Array, jit: bool):
    rngs = jr.split(rng, 4)
    n, m, d1, d2 = 5_000, 2_500, 1, 2
    x = jr.uniform(rngs[0], (n, d1))
    y = jr.uniform(rngs[1], (m, d2))
    xx = jr.uniform(rngs[2], (n, d2))
    yy = jr.uniform(rngs[3], (m, d2))
    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    geom_xy = pointcloud.PointCloud(xx, yy)
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, geom_xy)

    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=2, min_iterations=0, inner_iterations=10, max_iterations=2000
    )
    if jit:
      solver = jax.jit(solver)

    ot_gwlr = solver(prob)

    res0 = ot_gwlr.apply(x.T, axis=0)
    res1 = ot_gwlr.apply(y.T, axis=1)

    assert ot_gwlr.converged
    assert res0.shape == (d1, m)
    assert res1.shape == (d2, n)

  @pytest.mark.parametrize("cost_rank", [4, (2, 3, 4)])
  def test_fgw_lr_generic_cost_matrix(
      self, rng: jax.Array, cost_rank: Union[int, Tuple[int, int, int]]
  ):
    n, m = 20, 30
    rng1, rng2, rng3, rng4 = jr.split(rng, 4)
    x = jr.normal(rng1, shape=(n, 7))
    y = jr.normal(rng2, shape=(m, 6))
    xx = jr.normal(rng3, shape=(n, 5))
    yy = jr.normal(rng4, shape=(m, 5))

    geom_x = geometry.Geometry(cost_matrix=x @ x.T)
    geom_y = geometry.Geometry(cost_matrix=y @ y.T)
    geom_xy = geometry.Geometry(cost_matrix=xx @ yy.T)

    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, geom_xy, ranks=cost_rank, tolerances=5e-1
    )
    assert prob._is_low_rank_convertible
    lr_prob = prob.to_low_rank()
    assert lr_prob.is_low_rank

    solver = gromov_wasserstein_lr.LRGromovWasserstein(
        rank=5,
        epsilon=10.0,
        min_iterations=0,
        inner_iterations=10,
        max_iterations=2000
    )
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
    np.testing.assert_array_equal(jnp.isfinite(out.costs), True)

  @pytest.mark.parametrize("scale_cost", ["mean", "max_cost"])
  def test_fgw_scale_cost(
      self, clouds: FusedClouds, scale_cost: Literal["mean", "max_cost"]
  ):
    epsilon = 0.1
    fused_penalty = 1
    geom_x = pointcloud.PointCloud(clouds.x, scale_cost=1.0)
    geom_y = pointcloud.PointCloud(clouds.y, scale_cost=1.0)
    geom_xy = pointcloud.PointCloud(clouds.x_2, clouds.y_2, scale_cost=1.0)
    geom_x_scaled = pointcloud.PointCloud(clouds.x, scale_cost=scale_cost)
    geom_y_scaled = pointcloud.PointCloud(clouds.y, scale_cost=scale_cost)
    geom_xy_scaled = pointcloud.PointCloud(
        clouds.x_2, clouds.y_2, scale_cost=scale_cost
    )

    prob_no_scale = quadratic_problem.QuadraticProblem(
        geom_x_scaled,
        geom_y_scaled,
        geom_xy_scaled,
        fused_penalty=fused_penalty,
        scale_cost=None,
    )
    prob_scale = quadratic_problem.QuadraticProblem(
        geom_x,
        geom_y,
        geom_xy,
        fused_penalty=fused_penalty,
        scale_cost=scale_cost
    )
    linear_solver = sinkhorn.Sinkhorn()
    solver = gromov_wasserstein.GromovWasserstein(
        linear_solver, epsilon=epsilon
    )

    gt = solver(prob_scale)
    pred = solver(prob_no_scale)

    np.testing.assert_allclose(pred.matrix, gt.matrix)
    np.testing.assert_allclose(pred.costs, gt.costs)

  @pytest.mark.parametrize("fused_penalty", [0.3, 5.1])
  def test_fgw_fused_penalty(self, rng: jax.Array, fused_penalty: float):
    rtol = atol = 1e-5
    n, m, d = 21, 32, 2
    rngs = jr.split(rng, 4)
    xx = jr.normal(rngs[0], (n, d))
    yy = jr.normal(rngs[1], (m, d))
    x = jr.normal(rngs[2], (n, d))
    y = jr.normal(rngs[3], (m, d))

    geom_xy = pointcloud.PointCloud(x, y, scale_cost=1.0)
    geom_xy_fp = pointcloud.PointCloud(x, y, scale_cost=1.0 / fused_penalty)
    geom_xx = pointcloud.PointCloud(xx)
    geom_yy = pointcloud.PointCloud(yy)

    out = quadratic.solve(
        geom_xx,
        geom_yy,
        geom_xy=geom_xy,
        fused_penalty=fused_penalty,
        store_inner_errors=True
    )
    out_fp = quadratic.solve(
        geom_xx,
        geom_yy,
        geom_xy=geom_xy_fp,
        fused_penalty=1.0,
        store_inner_errors=True
    )

    np.testing.assert_allclose(out.costs, out_fp.costs, rtol=rtol, atol=atol)
    np.testing.assert_allclose(out.errors, out_fp.errors, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        out.primal_cost, out_fp.primal_cost, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        out.reg_gw_cost, out_fp.reg_gw_cost, rtol=rtol, atol=atol
    )
