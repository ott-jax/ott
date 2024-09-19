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

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import pointcloud


@pytest.mark.fast()
class TestCostStd:

  def test_coststd(self, rng: jax.Array):
    """Test consistency of std evaluation."""
    n, m, d = 5, 18, 10
    rngs = jax.random.split(rng, 5)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d)) + 1

    geom = pointcloud.PointCloud(x, y)
    std = jnp.std(geom.cost_matrix)
    mean = jnp.mean(geom.cost_matrix)
    np.testing.assert_allclose(geom.std_cost_matrix, std, rtol=1e-5, atol=1e-5)

    eps = pointcloud.PointCloud(x, y, relative_epsilon="mean").epsilon
    np.testing.assert_allclose(5e-2 * mean, eps, rtol=1e-5, atol=1e-5)

    eps = pointcloud.PointCloud(x, y, relative_epsilon="std").epsilon
    np.testing.assert_allclose(5e-2 * std, eps, rtol=1e-5, atol=1e-5)
