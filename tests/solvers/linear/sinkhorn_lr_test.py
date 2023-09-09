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
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import low_rank, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr


class TestLRSinkhorn:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    self.dim = 4
    self.n = 23
    self.m = 27
    self.rng, *rngs = jax.random.split(rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))

    # adding zero weights to test proper handling:
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @pytest.mark.fast.with_args(
      "use_lrcgeom,initializer,gamma_rescale,lse_mode", (
          (True, "rank2", False, True),
          (False, "random", True, False),
          (True, "k-means", False, True),
      ),
      only_fast=0
  )
  def test_euclidean_point_cloud_lr(
      self, use_lrcgeom: bool, initializer: str, gamma_rescale: bool,
      lse_mode: bool
  ):
    """Two point clouds, tested with 3 different initializations."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y)
    # This test to check LR can work both with LRCGeometries and regular ones
    if use_lrcgeom:
      geom = geom.to_LRCGeometry()
      assert isinstance(geom, low_rank.LRCGeometry)
    ot_prob = linear_problem.LinearProblem(geom, self.a, self.b)

    # Start with a low rank parameter
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=6,
        epsilon=0.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer
    )
    out = solver(ot_prob)

    criterions = out.errors
    criterions = criterions[criterions > -1]

    # Check convergence
    if out.converged:
      assert criterions[-1] < threshold
    np.testing.assert_allclose(out.transport_mass, 1.0, rtol=5e-4, atol=5e-4)

    # Store cost value.
    cost_1 = out.primal_cost

    # Try with higher rank
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=14,
        epsilon=0.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer,
    )
    out = solver(ot_prob)

    np.testing.assert_allclose(out.transport_mass, 1.0, rtol=5e-4, atol=5e-4)

    cost_2 = out.primal_cost
    # Ensure solution with more rank budget has lower cost (not guaranteed)
    try:
      assert cost_1 > cost_2
    except AssertionError:
      # at least test whether the values are close
      np.testing.assert_allclose(cost_1, cost_2, rtol=1e-4, atol=1e-4)

    # Ensure cost can still be computed on different geometry.
    other_geom = pointcloud.PointCloud(self.x, self.y + 0.3)
    cost_other = out.transport_cost_at_geom(other_geom)
    cost_other_lr = out.transport_cost_at_geom(other_geom.to_LRCGeometry())
    assert cost_other > 0.0
    np.testing.assert_allclose(cost_other, cost_other_lr, rtol=1e-6, atol=1e-6)

    # Ensure cost is higher when using high entropy.
    # (Note that for small entropy regularizers, this can be the opposite
    # due to non-convexity of problem and benefit of adding regularizer)
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=14,
        epsilon=5e-1,
        gamma=1.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer,
    )
    out = solver(ot_prob)

    cost_3 = out.primal_cost
    try:
      assert cost_3 > cost_2
    except AssertionError:
      np.testing.assert_allclose(cost_3, cost_2, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("axis", [0, 1])
  def test_output_apply_batch_size(self, axis: int):
    n_stack, threshold = 3, 1e-3
    data = self.a if axis == 0 else self.b

    geom = pointcloud.PointCloud(self.x, self.y)
    ot_prob = linear_problem.LinearProblem(geom, self.a, self.b)
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=10,
        epsilon=0.0,
    )
    out = solver(ot_prob)

    gt = out.apply(data, axis=axis)
    pred = out.apply(jnp.stack([data] * n_stack), axis=axis)

    np.testing.assert_array_equal(gt.shape, (geom.shape[1 - axis],))
    np.testing.assert_array_equal(pred.shape, (n_stack, geom.shape[1 - axis]))
    np.testing.assert_allclose(
        pred, jnp.stack([gt] * n_stack), rtol=1e-6, atol=1e-6
    )

  @pytest.mark.fast()
  @pytest.mark.skipif(
      jax.__version_info__ < (0, 4, 0),
      reason="`jax.experimental.io_callback` doesn't exist"
  )
  def test_progress_fn(self):
    """Check that the callback function is actually called."""
    num_iterations = 37

    def progress_fn(
        status: Tuple[np.ndarray, np.ndarray, np.ndarray,
                      sinkhorn_lr.LRSinkhornState], *args: Any
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

    rank = 2
    inner_iterations = 10

    _ = sinkhorn_lr.LRSinkhorn(
        rank,
        progress_fn=progress_fn,
        max_iterations=num_iterations,
        inner_iterations=inner_iterations
    )(
        lin_prob
    )

    # check that the function is called on the 10th iteration (iter #9), the
    # 20th iteration (iter #19).
    assert traced_values["iters"] == [9, 19]

    # check that error decreases
    np.testing.assert_array_equal(np.diff(traced_values["error"]) < 0, True)

    # check that max iterations is provided each time: [30, 30]
    assert traced_values["total"] == [num_iterations] * 2

  @pytest.mark.fast.with_args(eps=[0.0, 1e-1])
  def test_lse_matches_kernel_mode(self, eps: float):
    threshold = 1e-3
    tol = 1e-5
    rank = 5
    geom = pointcloud.PointCloud(self.x, self.y)
    ot_prob = linear_problem.LinearProblem(geom, self.a, self.b)

    out_lse = sinkhorn_lr.LRSinkhorn(
        lse_mode=True,
        threshold=threshold,
        rank=rank,
        epsilon=eps,
    )(
        ot_prob
    )

    out_kernel = sinkhorn_lr.LRSinkhorn(
        lse_mode=False,
        threshold=threshold,
        rank=rank,
        epsilon=eps,
    )(
        ot_prob
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
  def test_lr_unbalanced_lse(self, tau_a: float, tau_b: float, ti: bool):
    rank, epsilon, threshold = 10, 0.0, 1e-4
    geom = pointcloud.PointCloud(self.x, self.y)
    prob = linear_problem.LinearProblem(
        geom, self.a, self.b, tau_a=tau_a, tau_b=tau_b
    )

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
      self, tau_a: float, tau_b: float, epsilon: float, lse_mode: bool
  ):
    rank, threshold = 8, 1e-4
    geom = pointcloud.PointCloud(self.x, self.y)
    prob = linear_problem.LinearProblem(
        geom, self.a, self.b, tau_a=tau_a, tau_b=tau_b
    )

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
    np.testing.assert_allclose(out.errors, out_ti.errors, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out.reg_ot_cost, out_ti.reg_ot_cost, rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(out.matrix, out_ti.matrix, rtol=1e-2, atol=1e-2)
