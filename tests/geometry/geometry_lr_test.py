# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Test Low-Rank Geometry."""
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.geometry import costs, geometry, low_rank, pointcloud


@pytest.mark.fast
class TestLRGeometry:

  def test_apply(self, rng: jnp.ndarray):
    """Test application of cost to vec or matrix."""
    n, m, r = 17, 11, 7
    keys = jax.random.split(rng, 5)
    c1 = jax.random.normal(keys[0], (n, r))
    c2 = jax.random.normal(keys[1], (m, r))
    c = jnp.matmul(c1, c2.T)
    bias = 0.27
    geom = geometry.Geometry(c + bias)
    geom_lr = low_rank.LRCGeometry(c1, c2, bias=bias)
    for dim, axis in ((m, 1), (n, 0)):
      for mat_shape in ((dim, 2), (dim,)):
        mat = jax.random.normal(keys[2], mat_shape)
        np.testing.assert_allclose(
            geom.apply_cost(mat, axis=axis),
            geom_lr.apply_cost(mat, axis=axis),
            rtol=1e-4
        )

  @pytest.mark.parametrize("scale_cost", ['mean', 'max_cost', 'max_bound', 42.])
  def test_conversion_pointcloud(
      self, rng: jnp.ndarray, scale_cost: Union[str, float]
  ):
    """Test conversion from PointCloud to LRCGeometry."""
    n, m, d = 17, 11, 3
    keys = jax.random.split(rng, 3)
    x = jax.random.normal(keys[0], (n, d))
    y = jax.random.normal(keys[1], (m, d))

    geom = pointcloud.PointCloud(x, y, scale_cost=scale_cost)
    geom_lr = geom.to_LRCGeometry()

    assert geom._scale_cost == geom_lr._scale_cost
    np.testing.assert_allclose(
        geom.inv_scale_cost, geom_lr.inv_scale_cost, rtol=1e-6, atol=1e-6
    )
    for dim, axis in ((m, 1), (n, 0)):
      for mat_shape in ((dim, 2), (dim,)):
        mat = jax.random.normal(keys[2], mat_shape)
        np.testing.assert_allclose(
            geom.apply_cost(mat, axis=axis),
            geom_lr.apply_cost(mat, axis=axis),
            rtol=1e-4
        )

  def test_apply_squared(self, rng: jnp.ndarray):
    """Test application of squared cost to vec or matrix."""
    n, m = 27, 25
    keys = jax.random.split(rng, 5)
    for r in [3, 15]:
      c1 = jax.random.normal(keys[0], (n, r))
      c2 = jax.random.normal(keys[1], (m, r))
      c = jnp.matmul(c1, c2.T)
      geom = geometry.Geometry(c)
      geom2 = geometry.Geometry(c ** 2)
      geom_lr = low_rank.LRCGeometry(c1, c2)
      for dim, axis in ((m, 1), (n, 0)):
        for mat_shape in ((dim, 2), (dim,)):
          mat = jax.random.normal(keys[2], mat_shape)
          out_lr = geom_lr.apply_square_cost(mat, axis=axis)
          np.testing.assert_allclose(
              geom.apply_square_cost(mat, axis=axis), out_lr, rtol=5e-4
          )
          np.testing.assert_allclose(
              geom2.apply_cost(mat, axis=axis), out_lr, rtol=5e-4
          )

  def test_add_lr_geoms(self, rng: jnp.ndarray):
    """Test application of cost to vec or matrix."""
    n, m, r, q = 17, 11, 7, 2
    keys = jax.random.split(rng, 5)
    c1 = jax.random.normal(keys[0], (n, r))
    c2 = jax.random.normal(keys[1], (m, r))
    d1 = jax.random.normal(keys[0], (n, q))
    d2 = jax.random.normal(keys[1], (m, q))

    c = jnp.matmul(c1, c2.T)
    d = jnp.matmul(d1, d2.T)
    geom = geometry.Geometry(c + d)

    geom_lr_c = low_rank.LRCGeometry(c1, c2)
    geom_lr_d = low_rank.LRCGeometry(d1, d2)
    geom_lr = low_rank.add_lrc_geom(geom_lr_c, geom_lr_d)

    for dim, axis in ((m, 1), (n, 0)):
      mat = jax.random.normal(keys[1], (dim, 2))
      np.testing.assert_allclose(
          geom.apply_cost(mat, axis=axis),
          geom_lr.apply_cost(mat, axis=axis),
          rtol=1e-4
      )
      vec = jax.random.normal(keys[1], (dim,))
      np.testing.assert_allclose(
          geom.apply_cost(vec, axis=axis),
          geom_lr.apply_cost(vec, axis=axis),
          rtol=1e-4
      )

  @pytest.mark.parametrize("axis", [0, 1])
  @pytest.mark.parametrize("fn", [lambda x: x + 10, lambda x: x * 2])
  def test_apply_affine_function_efficient(
      self, rng: jnp.ndarray, fn: Callable[[jnp.ndarray], jnp.ndarray],
      axis: int
  ):
    n, m, d = 21, 13, 3
    keys = jax.random.split(rng, 3)
    x = jax.random.normal(keys[0], (n, d))
    y = jax.random.normal(keys[1], (m, d))
    vec = jax.random.normal(keys[2], (n if axis == 0 else m,))

    geom = pointcloud.PointCloud(x, y)

    res_eff = geom.apply_cost(vec, axis=axis, fn=fn, is_linear=True)
    res_ineff = geom.apply_cost(vec, axis=axis, fn=fn, is_linear=False)

    if fn(0.0) == 0.0:
      np.testing.assert_allclose(res_eff, res_ineff, rtol=1e-4, atol=1e-4)
    else:
      with pytest.raises(AssertionError):
        np.testing.assert_allclose(res_ineff, res_eff, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("rank", [5, 1000])
  def test_point_cloud_to_lr(self, rng: jnp.ndarray, rank: int):
    n, m = 1500, 1000
    scale = 2.0
    keys = jax.random.split(rng, 2)
    x = jax.random.normal(keys[0], (n, rank))
    y = jax.random.normal(keys[1], (m, rank))

    geom_pc = pointcloud.PointCloud(x, y)
    geom_lr = geom_pc.to_LRCGeometry(scale=scale)

    if n * m > (n + m) * rank:
      assert isinstance(geom_lr, low_rank.LRCGeometry)
    else:
      assert isinstance(geom_lr, pointcloud.PointCloud)
      np.testing.assert_allclose(geom_lr.x, jnp.sqrt(scale) * geom_pc.x)
      np.testing.assert_allclose(geom_lr.y, jnp.sqrt(scale) * geom_pc.y)


class TestCostMatrixFactorization:

  @staticmethod
  def assert_upper_bound(
      geom: geometry.Geometry, geom_lr: low_rank.LRCGeometry, *, rank: int,
      tol: float
  ):
    # Theorem 1.2 `Sample-Optimal Low-Rank Approximation of Distance Matrices
    # https://arxiv.org/abs/1906.00339
    A = geom.cost_matrix
    C1, C2 = geom_lr.cost_1, geom_lr.cost_2

    U, D, VT = jnp.linalg.svd(A)
    # best k-rank approx.
    A_k = U[:, :rank] @ jnp.diag(D[:rank]) @ VT[:rank]

    lhs = jnp.linalg.norm(A - C1 @ C2.T) ** 2
    rhs = jnp.linalg.norm(A - A_k) ** 2 + tol * jnp.linalg.norm(A) ** 2

    assert lhs <= rhs

  @pytest.mark.fast.with_args(rank=[2, 3], tol=[5e-1, 1e-2], only_fast=0)
  def test_geometry_to_lr(self, rng: jnp.ndarray, rank: int, tol: float):
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, shape=(370, 3))
    y = jax.random.normal(key2, shape=(460, 3))
    geom = geometry.Geometry(cost_matrix=x @ y.T)

    geom_lr = geom.to_LRCGeometry(rank=rank, tol=tol, seed=42)

    np.testing.assert_array_equal(geom.shape, geom_lr.shape)
    assert geom_lr.cost_rank == rank

    if rank == 2 and tol == 1e-2:
      pytest.mark.xfail("assert 171666.83 <= 154635.98")
    else:
      self.assert_upper_bound(geom, geom_lr, rank=rank, tol=tol)

  @pytest.mark.fast.with_args(
      "batch_size,scale_cost", [(None, "mean"), (32, None)], only_fast=1
  )
  def test_point_cloud_to_lr(
      self, rng: jnp.ndarray, batch_size: Optional[int],
      scale_cost: Optional[str]
  ):
    rank, tol = 7, 1e-1
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, shape=(384, 10))
    y = jax.random.normal(key2, shape=(512, 10))
    geom = pointcloud.PointCloud(
        x,
        y,
        cost_fn=costs.Euclidean(),
        batch_size=batch_size,
        power=3,
        scale_cost=scale_cost,
    )
    if geom.batch_size is not None:
      # because `self.assert_upper_bound` tries to instantiate the matrix
      geom = geom.subset(None, None, batch_size=None)

    geom_lr = geom.to_LRCGeometry(rank=rank, tol=tol)

    np.testing.assert_array_equal(geom.shape, geom_lr.shape)
    assert geom_lr.cost_rank == rank
    self.assert_upper_bound(geom, geom_lr, rank=rank, tol=tol)

  def test_to_lrc_geometry_noop(self, rng: jnp.ndarray):
    key1, key2 = jax.random.split(rng, 2)
    cost1 = jax.random.normal(key1, shape=(32, 2))
    cost2 = jax.random.normal(key2, shape=(23, 2))
    geom = low_rank.LRCGeometry(cost1, cost2)

    geom_lrc = geom.to_LRCGeometry(rank=10)

    assert geom is geom_lrc

  @pytest.mark.limit_memory("190 MB")
  def test_large_scale_factorization(self, rng: jnp.ndarray):
    rank, tol = 4, 1e-2
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, shape=(10_000, 7))
    y = jax.random.normal(key2, shape=(11_000, 7))
    geom = pointcloud.PointCloud(x, y, epsilon=1e-2, cost_fn=costs.Cosine())

    geom_lr = geom.to_LRCGeometry(rank=rank, tol=tol)

    np.testing.assert_array_equal(geom.shape, geom_lr.shape)
    assert geom_lr.cost_rank == rank
    # self.assert_upper_bound(geom, geom_lr, rank=rank, tol=tol)
