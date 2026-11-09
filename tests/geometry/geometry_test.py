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

from typing import Optional

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import epsilon_scheduler, geometry, pointcloud


@pytest.mark.fast()
class TestCostMeanStd:

  @pytest.mark.parametrize("geom_type", ["pc", "geometry"])
  def test_cost_stdmean(self, rng: jax.Array, geom_type: str):
    """Test consistency of std evaluation."""
    n, m, d = 5, 18, 10
    default_scale = epsilon_scheduler.DEFAULT_EPSILON_SCALE
    rngs = jr.split(rng, 5)
    x = jr.normal(rngs[0], (n, d))
    y = jr.normal(rngs[1], (m, d)) + 1

    geom = pointcloud.PointCloud(x, y)
    if geom_type == "geometry":
      geom = geometry.Geometry(cost_matrix=geom.cost_matrix)

    std = jnp.std(geom.cost_matrix)
    mean = jnp.mean(geom.cost_matrix)
    np.testing.assert_allclose(geom.std_cost_matrix, std, rtol=1e-5, atol=1e-5)

    eps = pointcloud.PointCloud(x, y, relative_epsilon="mean").epsilon
    np.testing.assert_allclose(default_scale * mean, eps, rtol=1e-5, atol=1e-5)

    eps = pointcloud.PointCloud(x, y, relative_epsilon="std").epsilon
    np.testing.assert_allclose(default_scale * std, eps, rtol=1e-5, atol=1e-5)


@pytest.mark.fast()
class TestApplyKernel:

  def test_apply_kernel_eps(self, rng: jax.Array):
    cost = jr.uniform(rng, (4, 5), maxval=10.0)
    vec = jnp.ones(5)
    geom = geometry.Geometry(cost_matrix=cost, epsilon=1e-2)

    actual = geom.apply_kernel(vec, eps=1.0, axis=1)

    np.testing.assert_allclose(
        actual, jnp.exp(-geom.cost_matrix) @ vec, rtol=1e-5
    )


@pytest.mark.fast()
class TestKernelGeometry:

  @pytest.mark.parametrize(("epsilon", "relative_epsilon"), [
      (None, None),
      (0.5, None),
      (None, "mean"),
      (0.5, "std"),
  ])
  def test_cost_matches_kernel(
      self, rng: jax.Array, epsilon: Optional[float],
      relative_epsilon: Optional[str]
  ):
    kernel = jr.uniform(rng, (5, 7), minval=0.1, maxval=1.0)
    geom = geometry.Geometry(
        kernel_matrix=kernel,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon
    )

    np.testing.assert_allclose(geom.kernel_matrix, kernel)
    np.testing.assert_allclose(
        jnp.exp(-geom.cost_matrix / geom.epsilon), kernel, rtol=1e-5, atol=1e-5
    )

  def test_default_epsilon(self, rng: jax.Array):
    kernel = jr.uniform(rng, (5, 7), minval=0.1, maxval=1.0)
    geom = geometry.Geometry(kernel_matrix=kernel)

    std = jnp.std(-jnp.log(kernel))
    expected = epsilon_scheduler.DEFAULT_EPSILON_SCALE * std
    np.testing.assert_allclose(geom.epsilon, expected, rtol=1e-5, atol=1e-5)

  def test_scale_cost_requires_cost_matrix(self, rng: jax.Array):
    kernel = jr.uniform(rng, (5, 7), minval=0.1, maxval=1.0)
    geom = geometry.Geometry(kernel_matrix=kernel, scale_cost="mean")

    with pytest.raises(ValueError, match="requires a cost matrix"):
      _ = geom.cost_matrix
