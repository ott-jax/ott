from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, low_rank, pointcloud
from ott.solvers import linear


@pytest.mark.fast()
class TestLRCGeometry:

  @pytest.mark.parametrize("std", [1e-2, 1.0, 1e2])
  @pytest.mark.parametrize("kernel", ["gaussian", "arccos"])
  def test_positive_features(
      self, rng: jax.Array, kernel: Literal["gaussian", "arccos"], std: float
  ):
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (10, 2))
    y = jax.random.normal(rng2, (12, 2))
    rank = 5

    geom = low_rank.LRKGeometry.from_pointcloud(
        x, y, kernel=kernel, std=std, rank=rank
    )

    if kernel == "gaussian":
      assert geom.rank == rank
    else:
      assert geom.rank == rank + 1
    np.testing.assert_array_equal(geom.k1 >= 0.0, True)
    np.testing.assert_array_equal(geom.k2 >= 0.0, True)

  @pytest.mark.parametrize("n", [0, 1, 2])
  def test_arccos_j_function(self, rng: jax.Array, n: int):

    def j(theta: float) -> float:
      if n == 0:
        return jnp.pi - theta
      if n == 1:
        return jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)
      if n == 2:
        return 3.0 * jnp.sin(theta) * jnp.cos(theta) + (jnp.pi - theta) * (
            1.0 + 2.0 * jnp.cos(theta) ** 2
        )
      raise NotImplementedError(n)

    x = jnp.abs(jax.random.normal(rng, (32,)))
    cost_fn = costs.Arccos(n)

    gt = jax.vmap(j)(x)
    pred = jax.vmap(cost_fn._j)(x)

    np.testing.assert_allclose(gt, pred, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("std", [1e-2, 1e-1, 1.0])
  @pytest.mark.parametrize("kernel", ["gaussian", "arccos"])
  def test_kernel_approximation(
      self, rng: jax.Array, kernel: Literal["gaussian", "arccos"], std: float
  ):
    rng, rng1, rng2 = jax.random.split(rng, 3)
    x = jax.random.normal(rng1, (230, 5))
    y = jax.random.normal(rng2, (260, 5))
    n = 1

    cost_fn = costs.SqEuclidean() if kernel == "gaussian" else costs.Arccos(n)
    pc = pointcloud.PointCloud(x, y, epsilon=std, cost_fn=cost_fn)
    gt_cost = pc.cost_matrix

    max_abs_diff = []
    for rank in [10, 50, 100, 200]:
      rng, rng_approx = jax.random.split(rng, 2)
      geom = low_rank.LRKGeometry.from_pointcloud(
          x, y, rank=rank, kernel=kernel, std=std, n=n, rng=rng_approx
      )
      pred_cost = geom.cost_matrix
      max_abs_diff.append(np.max(np.abs(gt_cost - pred_cost)))

    # test higher rank better approximates the cost
    np.testing.assert_array_equal(np.diff(max_abs_diff) <= 0.0, True)

  @pytest.mark.parametrize("std", [1e-2, 1e-1, 1.0])
  @pytest.mark.parametrize(("kernel", "n"), [("gaussian", 0), ("arccos", 0),
                                             ("arccos", 1), ("arccos", 2)])
  def test_sinkhorn_approximation(
      self,
      rng: jax.Array,
      kernel: Literal["gaussian", "arccos"],
      std: float,
      n: int,
  ):
    rng, rng1, rng2 = jax.random.split(rng, 3)
    x = jax.random.normal(rng1, (83, 5))
    y = jax.random.normal(rng2, (96, 5))
    solve_fn = jax.jit(linear.solve, static_argnames="lse_mode")

    cost_fn = costs.SqEuclidean() if kernel == "gaussian" else costs.Arccos(n)
    geom = pointcloud.PointCloud(x, y, epsilon=std, cost_fn=cost_fn)
    gt_out = solve_fn(geom, lse_mode=False)

    primal_costs_diff = []
    for rank in [3, 5, 20]:
      rng, rng_approx = jax.random.split(rng, 2)
      geom = low_rank.LRKGeometry.from_pointcloud(
          x, y, rank=rank, kernel=kernel, std=std, n=n, rng=rng_approx
      )

      pred_out = solve_fn(geom, lse_mode=False)
      primal_costs_diff.append(
          np.abs(gt_out.primal_cost - pred_out.primal_cost)
      )

    diff = np.diff(primal_costs_diff)
    try:
      # test higher rank better approximates the Sinkhorn solution
      np.testing.assert_array_equal(diff <= 0.0, True)
    except AssertionError:
      # arccos-0-1.0:
      # diff: array([1.072884e-06, 3.576279e-07], dtype=float32)
      np.testing.assert_allclose(diff, 0.0, rtol=1e-4, atol=1e-5)
