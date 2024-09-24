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

from ott.geometry import geometry, pointcloud


@pytest.mark.fast()
class TestCostMeanStd:

  @pytest.mark.parametrize("geom_type", ["pc", "geometry"])
  def test_coststdmeanpc(self, rng: jax.Array, geom_type: str):
    """Test consistency of std evaluation."""
    n, m, d = 5, 18, 10
    # should match that in the `DEFAULT_SCALE` in epsilon_scheduler.py
    default_scale = 5e-2
    rngs = jax.random.split(rng, 5)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d)) + 1

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
