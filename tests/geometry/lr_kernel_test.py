from typing import Literal

import jax
import numpy as np
import pytest
from ott.geometry import low_rank, pointcloud


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

    assert geom.rank == rank
    np.testing.assert_array_equal(geom.k1 >= 0.0, True)
    np.testing.assert_array_equal(geom.k2 >= 0.0, True)

  @pytest.mark.parametrize("eps", [1e-3, 1e-2, 1e-1, 5e-1, 1.0])
  def test_gaussian_approximation(self, rng: jax.Array, eps: float):
    rng, rng1, rng2 = jax.random.split(rng, 3)
    x = jax.random.normal(rng1, (230, 5))
    y = jax.random.normal(rng2, (260, 5))

    pc = pointcloud.PointCloud(x, y, epsilon=eps, scale_cost=1.0)
    gt_cost = pc.cost_matrix

    max_abs_diff = []
    for rank in [25, 50, 100, 200]:
      rng, rng_approx = jax.random.split(rng, 2)
      geom = low_rank.LRKGeometry.from_pointcloud(
          x, y, rank=rank, kernel="gaussian", std=eps, rng=rng_approx
      )
      pred_cost = geom.cost_matrix

      max_abs_diff.append(np.max(np.abs(gt_cost - pred_cost)))

    # test that higher rank better approximates the cost
    np.testing.assert_array_equal(np.diff(max_abs_diff) <= 0.0, True)

  def test_arccos_approximation(self, rng: jax.Array):
    pass

  def test_sinkhorn_approximation(self, rng: jax.Array):
    pass

  def test_sinkhorn_diff(self, rng: jax.Array):
    pass
