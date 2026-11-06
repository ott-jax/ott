from typing import List, Literal, Optional

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import costs, low_rank, pointcloud
from ott.solvers import linear


@pytest.mark.fast()
class TestLRCGeometry:

  @pytest.mark.parametrize("std", [1e-1, 1.0, 1e2])
  @pytest.mark.parametrize("kernel", ["gaussian", "arccos"])
  def test_positive_features(
      self, rng: jax.Array, kernel: Literal["gaussian", "arccos"], std: float
  ):
    rng1, rng2 = jr.split(rng, 2)
    x = jr.normal(rng1, (10, 2))
    y = jr.normal(rng2, (12, 2))
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

  @pytest.mark.parametrize(("row_ixs", "col_ixs"), [
      ([0, 3], None),
      (None, [1, 2, 4]),
      ([0, 3], [1, 2, 4]),
  ])
  def test_subset(
      self, rng: jax.Array, row_ixs: Optional[List[int]],
      col_ixs: Optional[List[int]]
  ):
    rng1, rng2 = jr.split(rng, 2)
    k1 = jr.uniform(rng1, (7, 3))
    k2 = jr.uniform(rng2, (5, 3))
    geom = low_rank.LRKGeometry(k1, k2, epsilon=1.0)

    expected = geom.kernel_matrix
    if row_ixs is not None:
      row_ixs = jnp.asarray(row_ixs)
      expected = expected[row_ixs]
    if col_ixs is not None:
      col_ixs = jnp.asarray(col_ixs)
      expected = expected[:, col_ixs]
    actual = geom.subset(row_ixs=row_ixs, col_ixs=col_ixs).kernel_matrix

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  def test_tree_flatten(self):
    geom = low_rank.LRKGeometry(
        jnp.ones((2, 1)), jnp.ones((3, 1)), scale_cost=2.0
    )
    assert jax.tree.map(lambda x: x, geom)._scale_cost == 2.0

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

    x = jnp.abs(jr.normal(rng, (32,)))
    cost_fn = costs.Arccos(n)

    gt = jax.vmap(j)(x)
    pred = jax.vmap(cost_fn._j)(x)

    np.testing.assert_allclose(gt, pred, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("std", [1e-2, 1e-1, 1.0])
  @pytest.mark.parametrize("kernel", ["gaussian", "arccos"])
  def test_kernel_approximation(
      self, rng: jax.Array, kernel: Literal["gaussian", "arccos"], std: float
  ):
    rng, rng1, rng2 = jr.split(rng, 3)
    x = jr.normal(rng1, (230, 5))
    y = jr.normal(rng2, (260, 5))
    n = 1

    cost_fn = costs.SqEuclidean() if kernel == "gaussian" else costs.Arccos(n)
    pc = pointcloud.PointCloud(x, y, epsilon=std, cost_fn=cost_fn)
    gt_cost = pc.cost_matrix

    max_abs_diff = []
    for rank in [50, 100, 400]:
      rng, rng_approx = jr.split(rng, 2)
      geom = low_rank.LRKGeometry.from_pointcloud(
          x, y, rank=rank, kernel=kernel, std=std, n=n, rng=rng_approx
      )
      pred_cost = geom.cost_matrix
      max_abs_diff.append(np.max(np.abs(gt_cost - pred_cost)))

    # test higher rank better approximates the cost
    np.testing.assert_array_equal(np.diff(max_abs_diff) <= 0.0, True)

  @pytest.mark.parametrize(("kernel", "n", "std"), [("gaussian", None, 1e-2),
                                                    ("gaussian", None, 1e-1),
                                                    ("arccos", 0, 1.0001),
                                                    ("arccos", 1, 2.0),
                                                    ("arccos", 2, 1.05)])
  def test_sinkhorn_approximation(
      self,
      rng: jax.Array,
      kernel: Literal["gaussian", "arccos"],
      std: float,
      n: Optional[int],
  ):
    rng, rng1, rng2 = jr.split(rng, 3)
    x = jr.normal(rng1, (83, 5))
    x /= jnp.linalg.norm(x, keepdims=True)
    y = jr.normal(rng2, (96, 5))
    y /= jnp.linalg.norm(y, keepdims=True)
    solve_fn = jax.jit(lambda g: linear.solve(g, lse_mode=False))

    cost_fn = costs.SqEuclidean() if kernel == "gaussian" else costs.Arccos(n)
    geom = pointcloud.PointCloud(x, y, epsilon=std, cost_fn=cost_fn)
    gt_out = solve_fn(geom)

    cs = []
    for rank in [5, 40, 80]:
      rng, rng_approx = jr.split(rng, 2)
      geom = low_rank.LRKGeometry.from_pointcloud(
          x, y, rank=rank, kernel=kernel, std=std, n=n, rng=rng_approx
      )

      pred_out = solve_fn(geom)
      cs.append(pred_out.reg_ot_cost)

    diff = np.diff(np.abs(gt_out.reg_ot_cost - np.array(cs)))
    try:
      # test higher rank better approximates the Sinkhorn solution
      np.testing.assert_array_equal(diff <= 0.0, True)
    except AssertionError:
      np.testing.assert_allclose(diff, 0.0, rtol=1e-2, atol=1e-2)
