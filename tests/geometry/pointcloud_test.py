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
"""Tests for apply_cost and apply_kernel."""
from typing import Union

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import costs, geometry, pointcloud


@pytest.mark.fast
class TestPointCloudApply:

  def test_apply_cost_and_kernel(self, rng: jnp.ndarray):
    """Test consistency of cost/kernel apply to vec."""
    n, m, p, b = 5, 8, 10, 7
    keys = jax.random.split(rng, 5)
    x = jax.random.normal(keys[0], (n, p))
    y = jax.random.normal(keys[1], (m, p)) + 1
    cost = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    vec0 = jax.random.normal(keys[2], (n, b))
    vec1 = jax.random.normal(keys[3], (m, b))

    geom = pointcloud.PointCloud(x, y, batch_size=3)
    prod0_online = geom.apply_cost(vec0, axis=0)
    prod1_online = geom.apply_cost(vec1, axis=1)
    geom = pointcloud.PointCloud(x, y, batch_size=None)
    prod0 = geom.apply_cost(vec0, axis=0)
    prod1 = geom.apply_cost(vec1, axis=1)
    geom = geometry.Geometry(cost)
    prod0_geom = geom.apply_cost(vec0, axis=0)
    prod1_geom = geom.apply_cost(vec1, axis=1)
    np.testing.assert_allclose(prod0_online, prod0, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(prod1_online, prod1, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(prod0_geom, prod0, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(prod1_geom, prod1, rtol=1e-03, atol=1e-02)

    geom = pointcloud.PointCloud(x, y, cost_fn=costs.Euclidean(), batch_size=4)
    prod0_online = geom.apply_cost(vec0, axis=0)
    prod1_online = geom.apply_cost(vec1, axis=1)
    geom = pointcloud.PointCloud(
        x, y, cost_fn=costs.Euclidean(), batch_size=None
    )
    prod0 = geom.apply_cost(vec0, axis=0)
    prod1 = geom.apply_cost(vec1, axis=1)
    np.testing.assert_allclose(prod0_online, prod0, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(prod1_online, prod1, rtol=1e-03, atol=1e-02)

    geom = pointcloud.PointCloud(x, y, batch_size=5)
    prod0_online = geom.apply_kernel(vec0, axis=0)
    prod1_online = geom.apply_kernel(vec1, axis=1)
    geom = pointcloud.PointCloud(x, y, batch_size=None)
    prod0 = geom.apply_kernel(vec0, axis=0)
    prod1 = geom.apply_kernel(vec1, axis=1)
    np.testing.assert_allclose(prod0_online, prod0, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(prod1_online, prod1, rtol=1e-03, atol=1e-02)

  def test_general_cost_fn(self, rng: jnp.ndarray):
    """Test non-vec cost apply to vec."""
    n, m, p, b = 5, 8, 10, 7
    keys = jax.random.split(rng, 5)
    x = jax.random.normal(keys[0], (n, p))
    y = jax.random.normal(keys[1], (m, p)) + 1
    vec0 = jax.random.normal(keys[2], (n, b))
    vec1 = jax.random.normal(keys[3], (m, b))

    geom = pointcloud.PointCloud(x, y, cost_fn=costs.Cosine(), batch_size=None)
    cost = geom.cost_matrix
    prod0 = geom.apply_cost(vec0, axis=0)
    prod1 = geom.apply_cost(vec1, axis=1)

    geom = geometry.Geometry(cost)
    prod0_geom = geom.apply_cost(vec0, axis=0)
    prod1_geom = geom.apply_cost(vec1, axis=1)

    np.testing.assert_allclose(prod0_geom, prod0, rtol=1e-03, atol=1e-02)
    np.testing.assert_allclose(prod1_geom, prod1, rtol=1e-03, atol=1e-02)

  def test_correct_shape(self):
    n, m, d = 11, 12, 17
    x = jnp.zeros((n, d))
    y = jnp.zeros((m, d))
    pc = pointcloud.PointCloud(x=x, y=y)
    np.testing.assert_array_equal(pc.shape, (n, m))

  @pytest.mark.parametrize("axis", [0, 1])
  def test_apply_cost_without_norm(self, rng: jnp.ndarray, axis: 1):
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, shape=(17, 3))
    y = jax.random.normal(key2, shape=(12, 3))
    pc = pointcloud.PointCloud(x, y, cost_fn=costs.Cosine())
    arr = jnp.ones((pc.shape[0],)) if axis == 0 else jnp.ones((pc.shape[1],))

    assert pc.cost_fn.norm is None
    with pytest.raises(
        AssertionError, match=r"Cost matrix is not a squared Euclidean\."
    ):
      _ = pc.vec_apply_cost(arr, axis=axis)

    expected = pc.cost_matrix @ arr if axis == 1 else pc.cost_matrix.T @ arr
    actual = pc.apply_cost(arr, axis=axis).squeeze()

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class TestPointCloudCosineConversion:

  @pytest.mark.parametrize(
      "scale_cost", ["mean", "median", "max_cost", "max_norm", 41]
  )
  def test_cosine_to_sqeucl_conversion(
      self, rng: jnp.ndarray, scale_cost: Union[str, float]
  ):
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, shape=(101, 4))
    y = jax.random.normal(key2, shape=(123, 4))
    cosine = pointcloud.PointCloud(
        x, y, cost_fn=costs.Cosine(), scale_cost=scale_cost
    )

    eucl = cosine._cosine_to_sqeucl()
    assert eucl.is_squared_euclidean

    np.testing.assert_allclose(
        2. * eucl.inv_scale_cost, cosine.inv_scale_cost, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        eucl.mean_cost_matrix, cosine.mean_cost_matrix, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        eucl.median_cost_matrix,
        cosine.median_cost_matrix,
        rtol=1e-6,
        atol=1e-6
    )
    np.testing.assert_allclose(
        eucl.cost_matrix, cosine.cost_matrix, rtol=1e-6, atol=1e-6
    )

  @pytest.mark.parametrize(
      "scale_cost", ["mean", "median", "max_cost", "max_norm", 2.0]
  )
  @pytest.mark.parametrize("axis", [0, 1])
  def test_apply_cost_cosine_to_sqeucl(
      self, rng: jnp.ndarray, axis: int, scale_cost: Union[str, float]
  ):
    key1, key2 = jax.random.split(rng, 2)
    x = jax.random.normal(key1, shape=(17, 5))
    y = jax.random.normal(key2, shape=(12, 5))
    cosine = pointcloud.PointCloud(
        x, y, cost_fn=costs.Cosine(), scale_cost=scale_cost
    )
    eucl = cosine._cosine_to_sqeucl()
    arr = jnp.ones((x.shape[0],)) if axis == 0 else jnp.ones((y.shape[0],))

    expected = cosine.apply_cost(arr, axis=axis)
    actual = eucl.apply_cost(arr, axis=axis)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
