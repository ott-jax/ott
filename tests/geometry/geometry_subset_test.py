from typing import Optional, Sequence, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.geometry import geometry, low_rank, pointcloud

Geom_t = Union[pointcloud.PointCloud, geometry.Geometry, low_rank.LRCGeometry]


@pytest.fixture()
def pc_masked(rng: jnp.ndarray) -> Tuple[pointcloud.PointCloud, Tuple]:
  n, m = 20, 30
  key1, key2 = jax.random.split(rng, 2)
  # x = jnp.full((n,), fill_value=1.)
  # y = jnp.full((m,), fill_value=2.)
  x = jax.random.normal(key1, shape=(n, 3))
  y = jax.random.normal(key1, shape=(m, 3))
  src_mask = jnp.asarray([0, 2, 1])
  tgt_mask = jnp.asarray([3, 5, 6])

  pc = pointcloud.PointCloud(x, y, src_mask=src_mask, tgt_mask=tgt_mask)
  masked = pointcloud.PointCloud(x[src_mask], y[tgt_mask])
  return pc, masked


@pytest.fixture(params=["geometry", "point_cloud", "low_rank"])
def geom_masked(request, pc_masked) -> Tuple[Geom_t, pointcloud.PointCloud]:
  pc, masked = pc_masked
  if request.param == "point_cloud":
    geom = pc
  elif request.param == "geometry":
    geom = geometry.Geometry(
        cost_matrix=pc.cost_matrix,
        src_mask=pc._src_mask,
        tgt_mask=pc._tgt_mask
    )
  elif request.param == "low_rank":
    geom = pc.to_LRCGeometry()
  else:
    raise NotImplementedError(request.param)
  return geom, masked


@pytest.mark.fast
class TestSubsetPointCloud:

  @pytest.mark.parametrize("tgt_ixs", [7, jnp.arange(5)])
  @pytest.mark.parametrize("src_ixs", [None, (3, 3)])
  @pytest.mark.parametrize(
      "clazz", [geometry.Geometry, pointcloud.PointCloud, low_rank.LRCGeometry]
  )
  def test_subset(
      self, rng: jnp.ndarray, clazz: Type[geometry.Geometry],
      src_ixs: Optional[Union[int, Sequence[int]]],
      tgt_ixs: Optional[Union[int, Sequence[int]]]
  ):
    key1, key2 = jax.random.split(rng, 2)
    new_batch_size = 7
    x = jax.random.normal(key1, shape=(10, 3))
    y = jax.random.normal(key2, shape=(20, 3))

    if clazz is geometry.Geometry:
      geom = clazz(cost_matrix=x @ y.T, scale_cost="mean")
    else:
      geom = clazz(x, y, scale_cost="max_cost", batch_size=5)
    n = geom.shape[0] if src_ixs is None else 1 if isinstance(
        src_ixs, int
    ) else len(src_ixs)
    m = geom.shape[1] if tgt_ixs is None else 1 if isinstance(
        tgt_ixs, int
    ) else len(tgt_ixs)

    geom_sub = geom.subset(src_ixs, tgt_ixs, batch_size=new_batch_size)

    assert type(geom_sub) == type(geom)
    np.testing.assert_array_equal(geom_sub.shape, (n, m))
    assert geom_sub._scale_cost == geom._scale_cost
    if clazz is pointcloud.PointCloud:
      # test overriding some argument
      assert geom_sub._batch_size == new_batch_size

  def test_masked_geometry_shape(
      self, pc_masked: Tuple[Geom_t, pointcloud.PointCloud]
  ):
    pc, masked = pc_masked

    assert masked._masked_geom is masked
    assert masked._masked_geom.shape == (3, 3)

  @pytest.mark.parametrize(
      "scale_cost", ["mean", "max_cost", "median", "max_norm", "max_bound"]
  )
  def test_masked_inverse_scaling(
      self, geom_masked: Tuple[Geom_t, pointcloud.PointCloud], scale_cost: str
  ):
    geom, masked = geom_masked
    geom = geom._set_scale_cost(scale_cost)
    masked = masked._set_scale_cost(scale_cost)

    try:
      desired = masked.inv_scale_cost
      actual = geom.inv_scale_cost
    except ValueError as e:
      if "not implemented" not in str(e):
        raise
      pytest.mark.xfail(str(e))
    else:
      np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("stat", ["mean", "median"])
  def test_masked_summary(
      self, geom_masked: Tuple[Geom_t, pointcloud.PointCloud], stat: str
  ):
    geom, masked = geom_masked
    if stat == "mean":
      np.testing.assert_allclose(geom.mean_cost_matrix, masked.mean_cost_matrix)
    else:
      np.testing.assert_allclose(
          geom.median_cost_matrix, masked.median_cost_matrix
      )
