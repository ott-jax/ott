from typing import Literal

import jax
import numpy as np
import pytest
from ott.geometry import costs, low_rank, pointcloud


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

  @pytest.mark.parametrize("std", [1e-2, 1e-1, 5e-1, 1.0])
  @pytest.mark.parametrize("kernel", ["gaussian", "arccos"])
  def test_kernel_approximation(
      self, rng: jax.Array, kernel: Literal["gaussian", "arccos"], std: float
  ):
    rng, rng1, rng2 = jax.random.split(rng, 3)
    s = 1
    x = jax.random.normal(rng1, (230, 5))
    y = jax.random.normal(rng2, (260, 5))

    cost_fn = costs.SqEuclidean() if kernel == "gaussian" else costs.Arccos(s)
    pc = pointcloud.PointCloud(x, y, epsilon=std, cost_fn=cost_fn)
    gt_cost = pc.cost_matrix

    max_abs_diff = []
    for rank in [10, 50, 100, 200]:
      rng, rng_approx = jax.random.split(rng, 2)
      geom = low_rank.LRKGeometry.from_pointcloud(
          x, y, rank=rank, kernel=kernel, std=std, s=s, rng=rng_approx
      )
      pred_cost = geom.cost_matrix
      max_abs_diff.append(np.max(np.abs(gt_cost - pred_cost)))

    # test that higher rank better approximates the cost
    np.testing.assert_array_equal(np.diff(max_abs_diff) <= 0.0, True)

  def test_sinkhorn_approximation(self, rng: jax.Array):
    pass

  def test_sinkhorn_diff(self, rng: jax.Array):
    pass
