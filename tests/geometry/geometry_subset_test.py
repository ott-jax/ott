from typing import Optional, Sequence, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.geometry import geometry, low_rank, pointcloud


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
