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
from typing import Any, Tuple, Type

import pytest

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import low_rank, pointcloud
from ott.initializers.linear import initializers_lr
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr
from tests import _utils


@pytest.fixture(scope="module")
def clouds() -> _utils.PointClouds:
  """Point clouds with zero weights, drawn exactly as this module always has.

  ``test_lr_unbalanced_ti`` compares two dual-update schemes at ``rtol=5e-4``,
  which only holds for this particular sample, so the draw is spelled out
  rather than delegated to :func:`tests._utils.random_clouds`.
  """
  n, m, dim = 23, 27, 4
  _, rng_x, rng_y, rng_a, rng_b = jr.split(_utils.root_key(), 5)
  return _utils.PointClouds(
      x=jr.uniform(rng_x, (n, dim)),
      y=jr.uniform(rng_y, (m, dim)),
      a=_utils.random_probs(rng_a, n, offset=0.0, zero_at=(0,)),
      b=_utils.random_probs(rng_b, m, offset=0.0, zero_at=(3,)),
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
    ot_prob = linear_problem.LinearProblem(geom, clouds.a, clouds.b)

    # Start with a low rank parameter
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=0.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer_class(rank)
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
    rank = 14
    solver = sinkhorn_lr.LRSinkhorn(
        threshold=threshold,
        rank=rank,
        epsilon=0.0,
        gamma_rescale=gamma_rescale,
        lse_mode=lse_mode,
        initializer=initializer_class(rank)
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
    out = solver(ot_prob)

    cost_3 = out.primal_cost
    try:
      assert cost_3 > cost_2
    except AssertionError:
      np.testing.assert_allclose(cost_3, cost_2, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("axis", [0, 1])
  def test_output_apply_batch_size(self, clouds: _utils.PointClouds, axis: int):
    n_stack, threshold = 3, 1e-3
    data = clouds.a if axis == 0 else clouds.b

    geom = pointcloud.PointCloud(clouds.x, clouds.y)
    ot_prob = linear_problem.LinearProblem(geom, clouds.a, clouds.b)
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
  def test_progress_fn(self, clouds: _utils.PointClouds):
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

    geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=1e-3)
    lin_prob = linear_problem.LinearProblem(geom, a=clouds.a, b=clouds.b)

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

    assert traced_values["iters"] == [9, 19, 29]
    assert traced_values["total"] == [num_iterations
                                     ] * len(traced_values["total"])

  @pytest.mark.fast.with_args(eps=[0.0, 1e-1])
  def test_lse_matches_kernel_mode(
      self, clouds: _utils.PointClouds, eps: float
  ):
    threshold = 1e-3
    tol = 1e-5
    rank = 5
    geom = pointcloud.PointCloud(clouds.x, clouds.y)
    ot_prob = linear_problem.LinearProblem(geom, clouds.a, clouds.b)

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
  def test_lr_unbalanced_lse(
      self, clouds: _utils.PointClouds, tau_a: float, tau_b: float, ti: bool
  ):
    rank, epsilon, threshold = 10, 0.0, 1e-4
    geom = pointcloud.PointCloud(clouds.x, clouds.y)
    prob = linear_problem.LinearProblem(
        geom, clouds.a, clouds.b, tau_a=tau_a, tau_b=tau_b
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
      self, clouds: _utils.PointClouds, tau_a: float, tau_b: float,
      epsilon: float, lse_mode: bool
  ):
    rank, threshold = 8, 1e-4
    geom = pointcloud.PointCloud(clouds.x, clouds.y)
    prob = linear_problem.LinearProblem(
        geom, clouds.a, clouds.b, tau_a=tau_a, tau_b=tau_b
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
    np.testing.assert_allclose(out.errors, out_ti.errors, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        out.reg_ot_cost, out_ti.reg_ot_cost, rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(out.matrix, out_ti.matrix, rtol=1e-2, atol=1e-2)
