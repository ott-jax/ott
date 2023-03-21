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
from typing import Optional, Sequence, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import geometry, low_rank, pointcloud

Geom_t = Union[pointcloud.PointCloud, geometry.Geometry, low_rank.LRCGeometry]


@pytest.fixture()
def pc_masked(
    rng: jax.random.PRNGKeyArray
) -> Tuple[pointcloud.PointCloud, pointcloud.PointCloud]:
  n, m = 20, 30
  rng1, rng2 = jax.random.split(rng, 2)
  # x = jnp.full((n,), fill_value=1.)
  # y = jnp.full((m,), fill_value=2.)
  x = jax.random.normal(rng1, shape=(n, 3))
  y = jax.random.normal(rng1, shape=(m, 3))
  src_mask = jnp.asarray([0, 1, 2])
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
        cost_matrix=pc.cost_matrix, src_mask=pc.src_mask, tgt_mask=pc.tgt_mask
    )
  elif request.param == "low_rank":
    geom = pc.to_LRCGeometry()
  else:
    raise NotImplementedError(request.param)
  return geom, masked


@pytest.mark.fast()
class TestMaskPointCloud:

  @pytest.mark.parametrize("tgt_ixs", [7, jnp.arange(5)])
  @pytest.mark.parametrize("src_ixs", [None, (3, 3)])
  @pytest.mark.parametrize(
      "clazz", [geometry.Geometry, pointcloud.PointCloud, low_rank.LRCGeometry]
  )
  def test_mask(
      self, rng: jax.random.PRNGKeyArray, clazz: Type[geometry.Geometry],
      src_ixs: Optional[Union[int, Sequence[int]]],
      tgt_ixs: Optional[Union[int, Sequence[int]]]
  ):
    rng1, rng2 = jax.random.split(rng, 2)
    new_batch_size = 7
    x = jax.random.normal(rng1, shape=(10, 3))
    y = jax.random.normal(rng2, shape=(20, 3))

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

    if clazz is geometry.Geometry:
      geom_sub = geom.subset(src_ixs, tgt_ixs)
    else:
      geom_sub = geom.subset(src_ixs, tgt_ixs, batch_size=new_batch_size)

    assert type(geom_sub) == type(geom)
    np.testing.assert_array_equal(geom_sub.shape, (n, m))
    assert geom_sub._scale_cost == geom._scale_cost
    if clazz is pointcloud.PointCloud:
      # test overriding some argument
      assert geom_sub._batch_size == new_batch_size

  @pytest.mark.parametrize(
      "scale_cost", ["mean", "max_cost", "median", "max_norm", "max_bound"]
  )
  def test_mask_inverse_scaling(
      self, geom_masked: Tuple[Geom_t, pointcloud.PointCloud], scale_cost: str
  ):
    geom, masked = geom_masked
    geom = geom._set_scale_cost(scale_cost)
    masked = masked._set_scale_cost(scale_cost)

    try:
      actual = geom.inv_scale_cost
      desired = masked.inv_scale_cost
    except ValueError as e:
      if "not implemented" not in str(e):
        raise
      pytest.mark.xfail(str(e))
    else:
      np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-6)
      geom_subset = geom.subset(geom.src_mask, geom.tgt_mask)
      np.testing.assert_allclose(
          geom_subset.cost_matrix, masked.cost_matrix, rtol=1e-6, atol=1e-6
      )

  @pytest.mark.parametrize("stat", ["mean", "median"])
  def test_masked_summary(
      self, geom_masked: Tuple[Geom_t, pointcloud.PointCloud], stat: str
  ):
    geom, masked = geom_masked
    if stat == "mean":
      np.testing.assert_allclose(
          geom.mean_cost_matrix, masked.mean_cost_matrix, rtol=1e-6, atol=1e-6
      )
    else:
      np.testing.assert_allclose(
          geom.median_cost_matrix,
          masked.median_cost_matrix,
          rtol=1e-6,
          atol=1e-6,
      )

  def test_mask_permutation(
      self, geom_masked: Tuple[Geom_t, pointcloud.PointCloud],
      rng: jax.random.PRNGKeyArray
  ):
    rng1, rng2 = jax.random.split(rng)
    geom, _ = geom_masked
    n, m = geom.shape

    # nullify the mask
    geom._src_mask = None
    geom._tgt_mask = None
    assert geom._masked_geom() is geom
    children, aux_data = geom.tree_flatten()
    gt_geom = type(geom).tree_unflatten(aux_data, children)

    geom._src_mask = jax.random.permutation(rng1, jnp.arange(n))
    geom._tgt_mask = jax.random.permutation(rng2, jnp.arange(m))

    np.testing.assert_allclose(geom.mean_cost_matrix, gt_geom.mean_cost_matrix)
    np.testing.assert_allclose(
        geom.median_cost_matrix, gt_geom.median_cost_matrix
    )

  def test_boolean_mask(
      self, geom_masked: Tuple[Geom_t, pointcloud.PointCloud],
      rng: jax.random.PRNGKeyArray
  ):
    rng1, rng2 = jax.random.split(rng)
    p = jnp.array([0.5, 0.5])
    geom, _ = geom_masked
    n, m = geom.shape

    src_mask = jax.random.choice(rng1, jnp.array([False, True]), (n,), p=p)
    tgt_mask = jax.random.choice(rng1, jnp.array([False, True]), (m,), p=p)
    geom._src_mask = src_mask
    geom._tgt_mask = tgt_mask
    gt_cost = geom.cost_matrix[src_mask, :][:, tgt_mask]

    np.testing.assert_allclose(
        geom.mean_cost_matrix, jnp.mean(gt_cost), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        geom.median_cost_matrix, jnp.median(gt_cost), rtol=1e-6, atol=1e-6
    )

  def test_subset_mask(
      self,
      geom_masked: Tuple[Geom_t, pointcloud.PointCloud],
  ):
    geom, masked = geom_masked
    assert masked.shape < geom.shape
    geom = geom.subset(geom.src_mask, geom.tgt_mask)

    assert geom.shape == masked.shape
    assert geom.src_mask.shape == (geom.shape[0],)
    assert geom.tgt_mask.shape == (geom.shape[1],)

    np.testing.assert_allclose(
        geom.mean_cost_matrix, masked.mean_cost_matrix, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        geom.median_cost_matrix,
        masked.median_cost_matrix,
        rtol=1e-6,
        atol=1e-6
    )
    np.testing.assert_allclose(
        geom.cost_matrix, masked.cost_matrix, rtol=1e-6, atol=1e-6
    )

  def test_mask_as_nonunique_indices(
      self,
      geom_masked: Tuple[Geom_t, pointcloud.PointCloud],
  ):
    geom, _ = geom_masked
    n, m = geom.shape
    src_ixs, tgt_ixs = [0, 2], [3, 1]
    geom._src_mask = jnp.asarray(src_ixs * 11)  # numbers chosen arbitrarily
    geom._tgt_mask = jnp.asarray(tgt_ixs * 13)

    np.testing.assert_array_equal(
        geom.src_mask, jnp.isin(jnp.arange(n), jnp.asarray(src_ixs))
    )
    np.testing.assert_array_equal(
        geom.tgt_mask, jnp.isin(jnp.arange(m), jnp.asarray(tgt_ixs))
    )
