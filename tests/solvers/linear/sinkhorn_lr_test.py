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
from typing import Any, Type

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import low_rank, pointcloud
from ott.initializers.linear import initializers_lr
from ott.problems.linear import linear_problem
from ott.solvers.linear import lr_utils, sinkhorn_lr
from tests import _utils


@pytest.fixture(scope="module")
def clouds() -> _utils.PointClouds:
  """Override with marginals containing zeros, to test their handling."""
  return _utils.random_clouds(
      jr.key(0), n=23, m=27, offset=0.0, zero_a=(0,), zero_b=(3,)
  )


class TestLRSinkhorn:

  @pytest.mark.fast.with_args(
      "use_lrcgeom,initializer_class,gamma_rescale,lse_mode", (
          (True, initializers_lr.Rank2Initializer, False, True),
          (False, initializers_lr.RandomInitializer, True, False),
          (True, initializers_lr.KMeansInitializer, False, True),
      ),
      only_fast=0
  )
  def test_euclidean_point_cloud_lr(
      self, clouds: _utils.PointClouds, use_lrcgeom: bool,
      initializer_class: Type[initializers_lr.LRInitializer],
      gamma_rescale: bool, lse_mode: bool
  ):
    """Two point clouds, tested with 3 different initializations."""
    rank, threshold = 6, 1e-3
    geom = pointcloud.PointCloud(clouds.x, clouds.y)
    # This test to check LR can work both with LRCGeometries and regular ones
    if use_lrcgeom:
      geom = geom.to_LRCGeometry()
      assert isinstance(geom, low_rank.LRCGeometry)
    prob = linear_problem.LinearProblem(geom, clouds.a, clouds.b)

    # Start with a low rank parameter
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=0.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer_class(rank)
    )
    out = solver(prob)

    criterions = out.errors
    criterions = criterions[criterions > -1]

    # Check convergence
    if out.converged:
      assert criterions[-1] < threshold
    np.testing.assert_allclose(out.transport_mass, 1.0, rtol=5e-4, atol=5e-4)

    # Store cost value.
    cost_1 = out.primal_cost

    # Try with higher rank
    rank = 14
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=0.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer_class(rank)
    )
    out = solver(prob)

    np.testing.assert_allclose(out.transport_mass, 1.0, rtol=5e-4, atol=5e-4)

    cost_2 = out.primal_cost
    # Ensure solution with more rank budget has lower cost (not guaranteed)
    try:
      assert cost_1 > cost_2
    except AssertionError:
      # at least test whether the values are close
      np.testing.assert_allclose(cost_1, cost_2, rtol=1e-4, atol=1e-4)

    # Ensure cost can still be computed on different geometry.
    other_geom = pointcloud.PointCloud(clouds.x, clouds.y + 0.3)
    cost_other = out.transport_cost_at_geom(other_geom)
    cost_other_lr = out.transport_cost_at_geom(other_geom.to_LRCGeometry())
    assert cost_other > 0.0
    np.testing.assert_allclose(cost_other, cost_other_lr, rtol=1e-6, atol=1e-6)

    # Ensure cost is higher when using high entropy.
    # (Note that for small entropy regularizers, this can be the opposite
    # due to non-convexity of problem and benefit of adding regularizer)
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=5e-1,
        gamma=1.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer_class(rank),
    )
    out = solver(prob)

    cost_3 = out.primal_cost
    try:
      assert cost_3 > cost_2
    except AssertionError:
      np.testing.assert_allclose(cost_3, cost_2, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("axis", [0, 1])
  def test_output_apply_batch_size(self, clouds: _utils.PointClouds, axis: int):
    n_stack, threshold = 3, 1e-3
    data = clouds.a if axis == 0 else clouds.b

    prob = clouds.problem()
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=10,
        epsilon=0.0,
    )
    out = solver(prob)

    gt = out.apply(data, axis=axis)
    pred = out.apply(jnp.stack([data] * n_stack), axis=axis)

    shape = prob.geom.shape[1 - axis]
    np.testing.assert_array_equal(gt.shape, (shape,))
    np.testing.assert_array_equal(pred.shape, (n_stack, shape))
    np.testing.assert_allclose(
        pred, jnp.stack([gt] * n_stack), rtol=1e-6, atol=1e-6
    )

  @pytest.mark.fast()
  def test_diverged_after_min_iterations(self, clouds: _utils.PointClouds):
    x = clouds.x.at[1, 0].set(jnp.nan)
    geom = pointcloud.PointCloud(x, clouds.y)
    prob = linear_problem.LinearProblem(geom, a=clouds.a, b=clouds.b)
    solver = sinkhorn_lr.LRSinkhorn(
        rank=2, min_iterations=10, inner_iterations=10
    )

    out = solver(prob)

    assert out.n_iters == 10

  def test_min_iterations(self, clouds: _utils.PointClouds):
    prob = clouds.problem()
    solver = sinkhorn_lr.LRSinkhorn(
        rank=2, min_iterations=20, max_iterations=20, inner_iterations=10
    )

    out = solver(prob)

    np.testing.assert_array_equal(jnp.isfinite(out.errors), [False, True])

  @pytest.mark.parametrize("lse_mode", [True, False])
  @pytest.mark.parametrize("tau", [1.0, 0.9])
  def test_dykstra_min_iter(
      self, clouds: _utils.PointClouds, lse_mode: bool, tau: float
  ):
    prob = clouds.problem(tau_a=tau, tau_b=tau)

    def solve(**kwargs_dys: Any) -> jnp.ndarray:
      solver = sinkhorn_lr.LRSinkhorn(
          rank=2,
          lse_mode=lse_mode,
          min_iterations=20,
          max_iterations=20,
          kwargs_dys=kwargs_dys,
      )
      return solver(prob).matrix

    np.testing.assert_allclose(
        solve(min_iter=10, max_iter=20, tolerance=1e10),
        solve(min_iter=10, max_iter=10),
        rtol=1e-5,
        atol=1e-6,
    )

  def test_crossed_threshold_recorded_errors(self, clouds: _utils.PointClouds):
    prob = clouds.problem()
    solver = sinkhorn_lr.LRSinkhorn(rank=2, threshold=1e3)
    state = solver.init_state(prob, solver.initializer(prob))
    state = state.set(errors=state.errors.at[0].set(1e4))

    state = solver.one_iteration(
        prob, state, iteration=solver.inner_iterations, compute_error=False
    )

    assert not state.crossed_threshold

  @pytest.mark.usefixtures("enable_x64")
  @pytest.mark.parametrize("lse_mode", [True, False])
  @pytest.mark.parametrize("tau", [1.0, 0.9])
  def test_grad(self, rng: jax.Array, lse_mode: bool, tau: float):
    rng_x, rng_y, rng_w, rng_v = jr.split(rng, 4)
    x = jr.normal(rng_x, (8, 2))
    y = jr.normal(rng_y, (9, 2))
    w = jr.normal(rng_w, (8, 9))
    v = jr.normal(rng_v, x.shape)
    v = 1e-4 * v / jnp.linalg.norm(v)
    solver = sinkhorn_lr.LRSinkhorn(
        rank=2,
        lse_mode=lse_mode,
        min_iterations=20,
        max_iterations=20,
        kwargs_dys={
            "min_iter": 20,
            "max_iter": 20
        },
    )

    def loss(x: jnp.ndarray) -> float:
      geom = pointcloud.PointCloud(x, y)
      prob = linear_problem.LinearProblem(geom, tau_a=tau, tau_b=tau)
      return jnp.vdot(w, solver(prob).matrix)

    grad = jax.grad(loss)(x)

    expected = loss(x + v) - loss(x - v)
    actual = 2.0 * jnp.vdot(v, grad)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)

  @pytest.mark.usefixtures("enable_x64")
  @pytest.mark.parametrize("lse_mode", [True, False])
  @pytest.mark.parametrize("tau", [1.0, 0.9])
  def test_float32_with_x64(self, rng: jax.Array, lse_mode: bool, tau: float):
    x = jr.normal(rng, (10, 2), dtype=jnp.float32)
    prob = linear_problem.LinearProblem(
        pointcloud.PointCloud(x), tau_a=tau, tau_b=tau
    )
    solver = sinkhorn_lr.LRSinkhorn(rank=2, lse_mode=lse_mode)

    out = solver(prob)

    for arr in (out.q, out.r, out.g, out.costs, out.errors, out.reg_ot_cost):
      assert arr.dtype == jnp.float32

  def test_zero_weights_epsilon(self, rng: jax.Array):
    rng_x, rng_y = jr.split(rng, 2)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 2))
    x_pad = jnp.concatenate([x, jnp.full((3, 2), 5.0)])
    a_pad = jnp.concatenate([jnp.full(10, 0.1), jnp.zeros(3)])
    prob = linear_problem.LinearProblem(pointcloud.PointCloud(x, y))
    prob_pad = linear_problem.LinearProblem(
        pointcloud.PointCloud(x_pad, y), a=a_pad
    )
    solver = sinkhorn_lr.LRSinkhorn(rank=2, epsilon=1e-1)
    q, r, g = solver.initializer(prob)
    q_pad = jnp.concatenate([q, jnp.zeros((3, 2))])

    out = solver(prob, init=(q, r, g))
    out_pad = solver(prob_pad, init=(q_pad, r, g))

    np.testing.assert_allclose(out_pad.reg_ot_cost, out.reg_ot_cost, rtol=1e-5)

  @pytest.mark.parametrize("tau", [1.0, 0.9])
  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_zero_weights_unrolled_grad(
      self, rng: jax.Array, lse_mode: bool, tau: float
  ):

    def loss(x: jnp.ndarray, n_pad: int) -> float:
      x = jnp.concatenate([x, jnp.full((n_pad, 2), 5.0)])
      a = jnp.concatenate([jnp.full(10, 0.1), jnp.zeros(n_pad)])
      q_pad = jnp.concatenate([q, jnp.zeros((n_pad, 2))])
      prob = linear_problem.LinearProblem(
          pointcloud.PointCloud(x, y), a=a, tau_a=tau, tau_b=tau
      )
      return solver(prob, init=(q_pad, r, g)).reg_ot_cost

    rng_x, rng_y = jr.split(rng, 2)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 2))
    solver = sinkhorn_lr.LRSinkhorn(
        rank=2, epsilon=1e-1, lse_mode=lse_mode, use_danskin=False
    )
    q, r, g = solver.initializer(
        linear_problem.LinearProblem(pointcloud.PointCloud(x, y))
    )

    grad = jax.grad(loss)(x, 0)
    grad_pad = jax.grad(loss)(x, 3)

    np.testing.assert_allclose(grad_pad, grad, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize(("tau_a", "tau_b"), [(0.9, 1.0), (1.0, 0.9)])
  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_one_sided_unbalanced_unrolled_grad(
      self, rng: jax.Array, lse_mode: bool, tau_a: float, tau_b: float
  ):

    def loss(x: jnp.ndarray) -> float:
      prob = linear_problem.LinearProblem(
          pointcloud.PointCloud(x, y), tau_a=tau_a, tau_b=tau_b
      )
      return solver(prob).reg_ot_cost

    rng_x, rng_y, rng_v = jr.split(rng, 3)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 2))
    v = jr.normal(rng_v, x.shape)
    v = 1e-2 * v / jnp.linalg.norm(v)
    solver = sinkhorn_lr.LRSinkhorn(
        rank=2,
        epsilon=1e-1,
        lse_mode=lse_mode,
        use_danskin=False,
        min_iterations=20,
        max_iterations=20,
        kwargs_dys={
            "min_iter": 20,
            "max_iter": 20
        },
    )

    grad = jax.grad(loss)(x)

    expected = loss(x + v) - loss(x - v)
    actual = 2.0 * jnp.vdot(v, grad)
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-5)

  @pytest.mark.parametrize("translation_invariant", [False, True])
  def test_unbalanced_dykstra_kernel_zero_weights(
      self, rng: jax.Array, translation_invariant: bool
  ):
    rng_x, rng_y, rng_q, rng_r = jr.split(rng, 4)
    x, y = jr.normal(rng_x, (10, 2)), jr.normal(rng_y, (12, 2))
    k_q, k_r = jr.uniform(rng_q, (10, 2)), jr.uniform(rng_r, (12, 2))
    x_pad = jnp.concatenate([x, jnp.zeros((3, 2))])
    a_pad = jnp.concatenate([jnp.full(10, 0.1), jnp.zeros(3)])
    k_q_pad = jnp.concatenate([k_q, jnp.zeros((3, 2))])
    prob = linear_problem.LinearProblem(
        pointcloud.PointCloud(x, y), tau_a=0.9, tau_b=0.9
    )
    prob_pad = linear_problem.LinearProblem(
        pointcloud.PointCloud(x_pad, y), a=a_pad, tau_a=0.9, tau_b=0.9
    )
    kwargs = {
        "translation_invariant": translation_invariant,
        "inner_iter": 1,
        "tolerance": 1e-5,
    }

    _, _, g = lr_utils.unbalanced_dykstra_kernel(
        k_q, k_r, jnp.ones(2), 1.0, prob, **kwargs
    )
    _, _, g_pad = lr_utils.unbalanced_dykstra_kernel(
        k_q_pad, k_r, jnp.ones(2), 1.0, prob_pad, **kwargs
    )

    np.testing.assert_allclose(g_pad, g, rtol=1e-5)

  def test_progress_fn(self, clouds: _utils.PointClouds):
    """Check that the callback function is actually called."""
    num_iterations = 37

    traced_values, progress_fn = _utils.tracing_progress_fn()

    prob = clouds.problem(epsilon=1e-3)

    rank = 2
    inner_iterations = 10

    _ = sinkhorn_lr.LRSinkhorn(
        rank,
        progress_fn=progress_fn,
        max_iterations=num_iterations,
        inner_iterations=inner_iterations
    )(
        prob
    )

    iters = traced_values["iters"]
    assert iters, "progress function was never called"
    assert iters == [inner_iterations * (v + 1) - 1 for v in range(len(iters))]
    assert traced_values["total"] == [num_iterations] * len(iters)

  @pytest.mark.fast.with_args(eps=[0.0, 1e-1], only_fast=0)
  def test_lse_matches_kernel_mode(
      self, clouds: _utils.PointClouds, eps: float
  ):
    threshold = 1e-3
    tol = 1e-5
    rank = 5
    prob = clouds.problem()

    out_lse = sinkhorn_lr.LRSinkhorn(
        lse_mode=True,
        threshold=threshold,
        rank=rank,
        epsilon=eps,
    )(
        prob
    )

    out_kernel = sinkhorn_lr.LRSinkhorn(
        lse_mode=False,
        threshold=threshold,
        rank=rank,
        epsilon=eps,
    )(
        prob
    )

    assert out_lse.converged
    assert out_kernel.converged
    np.testing.assert_allclose(
        out_lse.reg_ot_cost, out_kernel.reg_ot_cost, rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        out_lse.matrix, out_kernel.matrix, rtol=tol, atol=tol
    )

  @pytest.mark.fast.with_args("ti", [False, True], only_fast=0)
  @pytest.mark.parametrize(("tau_a", "tau_b"), [(0.9, 0.95), (0.89, 1.0),
                                                (1.0, 0.85)])
  def test_lr_unbalanced_lse(
      self, clouds: _utils.PointClouds, tau_a: float, tau_b: float, ti: bool
  ):
    rank, epsilon, threshold = 10, 0.0, 1e-4
    prob = clouds.problem(tau_a=tau_a, tau_b=tau_b)

    out_lse = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=epsilon,
        lse_mode=True,
        kwargs_dys={"translation_invariant": ti},
    )(
        prob
    )
    out_kernel = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=epsilon,
        lse_mode=False,
        kwargs_dys={"translation_invariant": ti},
    )(
        prob
    )

    assert out_lse.converged
    assert out_kernel.converged
    np.testing.assert_allclose(
        out_lse.reg_ot_cost, out_kernel.reg_ot_cost, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out_lse.matrix, out_kernel.matrix, rtol=1e-5, atol=1e-5
    )

  @pytest.mark.parametrize("lse_mode", [False, True])
  @pytest.mark.fast.with_args(("tau_a", "tau_b", "epsilon"),
                              [(0.92, 0.99, 1e-3), (0.75, 1.0, 0.0),
                               (1.0, 0.5, 0.0)],
                              only_fast=1)
  def test_lr_unbalanced_ti(
      self, clouds: _utils.PointClouds, tau_a: float, tau_b: float,
      epsilon: float, lse_mode: bool
  ):
    rank, threshold = 8, 1e-4
    prob = clouds.problem(tau_a=tau_a, tau_b=tau_b)

    out = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=epsilon,
        lse_mode=lse_mode,
        kwargs_dys={"translation_invariant": False},
    )(
        prob
    )
    out_ti = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=epsilon,
        lse_mode=lse_mode,
        kwargs_dys={"translation_invariant": True},
    )(
        prob
    )

    assert out.converged
    assert out_ti.converged
    np.testing.assert_allclose(out.errors, out_ti.errors, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        out.reg_ot_cost, out_ti.reg_ot_cost, rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(out.matrix, out_ti.matrix, rtol=1e-2, atol=1e-2)

  @pytest.mark.parametrize("epsilon", [5e-2, 1e-1])
  def test_epsilon_stationary_point(self, rng: jax.Array, epsilon: float):
    prob = _utils.random_clouds(rng).problem()
    solver = sinkhorn_lr.LRSinkhorn(rank=3, epsilon=epsilon, threshold=1e-7)

    out = solver(prob)
    grad_fn = jax.grad(sinkhorn_lr.compute_reg_ot_cost, argnums=(0, 1))
    grads = grad_fn(
        out.q, out.r, out.g, prob, epsilon=epsilon, use_danskin=False
    )

    assert out.converged
    # at a stationary point, the gradients are sums of row and column duals
    for grad in grads:
      separable = (
          grad.mean(axis=1, keepdims=True) + grad.mean(axis=0) - grad.mean()
      )
      np.testing.assert_allclose(grad, separable, rtol=1e-3, atol=1e-3)
