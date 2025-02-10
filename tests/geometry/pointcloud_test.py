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
from typing import Union

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import costs, geometry, pointcloud


class NonSymCost(costs.CostFn):

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    z = x - y
    return jnp.sum(z ** 2 * (jnp.sign(z) + 0.5) ** 2)


@pytest.mark.fast()
class TestPointCloudApply:

  def test_apply_cost_and_kernel(self, rng: jax.Array):
    """Test consistency of cost/kernel apply to vec."""
    n, m, p, b = 5, 8, 10, 7
    rngs = jax.random.split(rng, 5)
    x = jax.random.normal(rngs[0], (n, p))
    y = jax.random.normal(rngs[1], (m, p)) + 1
    cost = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    vec0 = jax.random.normal(rngs[2], (n, b))
    vec1 = jax.random.normal(rngs[3], (m, b))

    geom = pointcloud.PointCloud(x, y, batch_size=3)
    prod0_online = geom.apply_cost(vec0, axis=0)
    prod1_online = geom.apply_cost(vec1, axis=1)
    geom = pointcloud.PointCloud(x, y, batch_size=None)
    prod0 = geom.apply_cost(vec0, axis=0)
    prod1 = geom.apply_cost(vec1, axis=1)
    geom = geometry.Geometry(cost)
    prod0_geom = geom.apply_cost(vec0, axis=0)
    prod1_geom = geom.apply_cost(vec1, axis=1)
    np.testing.assert_allclose(prod0_online, prod0, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(prod1_online, prod1, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(prod0_geom, prod0, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(prod1_geom, prod1, rtol=1e-3, atol=1e-2)

    geom = pointcloud.PointCloud(x, y, cost_fn=costs.Euclidean(), batch_size=4)
    prod0_online = geom.apply_cost(vec0, axis=0)
    prod1_online = geom.apply_cost(vec1, axis=1)
    geom = pointcloud.PointCloud(
        x, y, cost_fn=costs.Euclidean(), batch_size=None
    )
    prod0 = geom.apply_cost(vec0, axis=0)
    prod1 = geom.apply_cost(vec1, axis=1)
    np.testing.assert_allclose(prod0_online, prod0, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(prod1_online, prod1, rtol=1e-3, atol=1e-2)

    geom = pointcloud.PointCloud(x, y, batch_size=5)
    prod0_online = geom.apply_kernel(vec0, axis=0)
    prod1_online = geom.apply_kernel(vec1, axis=1)
    geom = pointcloud.PointCloud(x, y, batch_size=None)
    prod0 = geom.apply_kernel(vec0, axis=0)
    prod1 = geom.apply_kernel(vec1, axis=1)
    np.testing.assert_allclose(prod0_online, prod0, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(prod1_online, prod1, rtol=1e-3, atol=1e-2)

  def test_general_cost_fn(self, rng: jax.Array):
    """Test non-vec cost apply to vec."""
    n, m, p, b = 5, 8, 10, 7
    rngs = jax.random.split(rng, 5)
    x = jax.random.normal(rngs[0], (n, p))
    y = jax.random.normal(rngs[1], (m, p)) + 1
    vec0 = jax.random.normal(rngs[2], (n, b))
    vec1 = jax.random.normal(rngs[3], (m, b))

    geom = pointcloud.PointCloud(x, y, cost_fn=costs.Cosine(), batch_size=None)
    cost = geom.cost_matrix
    prod0 = geom.apply_cost(vec0, axis=0)
    prod1 = geom.apply_cost(vec1, axis=1)

    geom = geometry.Geometry(cost)
    prod0_geom = geom.apply_cost(vec0, axis=0)
    prod1_geom = geom.apply_cost(vec1, axis=1)

    np.testing.assert_allclose(prod0_geom, prod0, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(prod1_geom, prod1, rtol=1e-3, atol=1e-2)

  def test_correct_shape(self):
    n, m, d = 11, 12, 17
    x = jnp.zeros((n, d))
    y = jnp.zeros((m, d))
    pc = pointcloud.PointCloud(x=x, y=y)
    np.testing.assert_array_equal(pc.shape, (n, m))

  @pytest.mark.parametrize("axis", [0, 1])
  def test_apply_cost_without_norm(self, rng: jax.Array, axis: 1):
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, shape=(17, 3))
    y = jax.random.normal(rng2, shape=(12, 3))
    pc = pointcloud.PointCloud(x, y, cost_fn=costs.Cosine())
    arr = jnp.ones((pc.shape[0],)) if axis == 0 else jnp.ones((pc.shape[1],))

    with pytest.raises(
        AssertionError, match=r"Cost matrix is not a squared Euclidean\."
    ):
      _ = pc._apply_sqeucl_cost(arr, axis=axis, scale_cost=1.0)

    expected = pc.cost_matrix @ arr if axis == 1 else pc.cost_matrix.T @ arr
    actual = pc.apply_cost(arr, axis=axis).squeeze()

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class TestPointCloudCosineConversion:

  @pytest.mark.parametrize("scale_cost", ["mean", "median", "max_cost", 41])
  def test_cosine_to_sqeucl_conversion(
      self, rng: jax.Array, scale_cost: Union[str, float]
  ):
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, shape=(101, 4))
    y = jax.random.normal(rng2, shape=(123, 4))
    cosine = pointcloud.PointCloud(
        x, y, cost_fn=costs.Cosine(), scale_cost=scale_cost
    )

    eucl = cosine._cosine_to_sqeucl()
    assert eucl.is_squared_euclidean

    np.testing.assert_allclose(
        2.0 * eucl.inv_scale_cost, cosine.inv_scale_cost, rtol=1e-6, atol=1e-6
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

  @pytest.mark.parametrize("scale_cost", ["mean", "median", "max_cost", 2.0])
  @pytest.mark.parametrize("axis", [0, 1])
  def test_apply_cost_cosine_to_sqeucl(
      self, rng: jax.Array, axis: int, scale_cost: Union[str, float]
  ):
    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, shape=(17, 5))
    y = jax.random.normal(rng2, shape=(12, 5))
    cosine = pointcloud.PointCloud(
        x, y, cost_fn=costs.Cosine(), scale_cost=scale_cost
    )
    eucl = cosine._cosine_to_sqeucl()
    arr = jnp.ones((x.shape[0],)) if axis == 0 else jnp.ones((y.shape[0],))

    expected = cosine.apply_cost(arr, axis=axis)
    actual = eucl.apply_cost(arr, axis=axis)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize(("n", "m"), [(20, 10), (9, 22)])
  def test_nonsym_cost_batched(self, rng: jax.Array, n: int, m: int):
    d, eps = 5, 1e-1
    rtol, atol = 1e-5, 1e-5
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    x = jax.random.normal(rng1, shape=(n, d))
    y = jax.random.normal(rng2, shape=(m, d))

    pc = pointcloud.PointCloud(x, y, cost_fn=NonSymCost())
    pc_batched = pointcloud.PointCloud(x, y, cost_fn=NonSymCost(), batch_size=4)

    f, g = jnp.zeros(n), jnp.zeros(m)
    u, v = jnp.ones(n), jnp.ones(m)
    arr0, arr1 = jax.random.normal(rng3, (n,)), jax.random.normal(rng4, (m,))

    # transport
    np.testing.assert_allclose(
        pc.transport_from_potentials(f, g),
        pc_batched.transport_from_potentials(f, g),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        pc.transport_from_scalings(u, v),
        pc_batched.transport_from_scalings(u, v),
        rtol=rtol,
        atol=atol,
    )

    # statistics
    np.testing.assert_allclose(
        pc.mean_cost_matrix,
        pc_batched._compute_summary_online("mean"),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        pc.cost_matrix.max(),
        pc_batched._compute_summary_online("max_cost"),
        rtol=rtol,
        atol=atol,
    )

    for axis, arr in zip([0, 1], [arr0, arr1]):
      # apply LSE
      gt, _ = pc.apply_lse_kernel(f, g, eps, axis=axis)
      pred, _ = pc_batched.apply_lse_kernel(f, g, eps, axis=axis)
      np.testing.assert_allclose(gt, pred, rtol=rtol, atol=atol)

      # apply cost
      gt = pc.apply_cost(arr, axis=axis)
      pred = pc_batched.apply_cost(arr, axis=axis)
      np.testing.assert_allclose(gt, pred, rtol=rtol, atol=atol)

      # apply kernel
      gt = pc.apply_kernel(arr, axis=axis)
      pred = pc_batched.apply_kernel(arr, axis=axis)
      np.testing.assert_allclose(gt, pred, rtol=rtol, atol=atol)
