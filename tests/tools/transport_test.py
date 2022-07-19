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
"""Tests for ott.tools.transport."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import linear_problems
from ott.geometry import pointcloud
from ott.tools import transport


@pytest.mark.fast
class TestTransport:
  """Tests for the Transport class."""

  def test_transport_from_point(self, rng: jnp.ndarray):
    rngs = jax.random.split(rng, 2)
    num_a, num_b = 23, 17
    x = jax.random.uniform(rngs[0], (num_a, 4))
    y = jax.random.uniform(rngs[1], (num_b, 4))
    ot = transport.solve(x, y, threshold=1e-2)

    np.testing.assert_array_equal(ot.matrix.shape, (num_a, num_b))
    np.testing.assert_allclose(jnp.sum(ot.matrix, axis=1), ot.a, atol=1e-3)
    np.testing.assert_allclose(jnp.sum(ot.matrix, axis=0), ot.b, atol=1e-3)

  def test_transport_from_geom(self, rng: jnp.ndarray):
    rngs = jax.random.split(rng, 3)
    num_a, num_b = 23, 17
    x = jax.random.uniform(rngs[0], (num_a, 4))
    y = jax.random.uniform(rngs[1], (num_b, 4))
    geom = pointcloud.PointCloud(x, y, epsilon=1e-2, batch_size=8)
    b = jax.random.uniform(rngs[2], (num_b,))
    b /= jnp.sum(b)
    ot = transport.solve(geom, b=b, threshold=1e-3)

    np.testing.assert_array_equal(ot.matrix.shape, (num_a, num_b))
    np.testing.assert_allclose(jnp.sum(ot.matrix, axis=1), ot.a, atol=1e-3)
    np.testing.assert_allclose(jnp.sum(ot.matrix, axis=0), ot.b, atol=1e-3)

  def test_transport_from_problem(self, rng: jnp.ndarray):
    rngs = jax.random.split(rng, 3)
    num_a, num_b = 23, 17
    x = jax.random.uniform(rngs[0], (num_a, 4))
    y = jax.random.uniform(rngs[1], (num_b, 4))
    geom = pointcloud.PointCloud(x, y, batch_size=9)
    b = jax.random.uniform(rngs[2], (num_b,))
    b /= jnp.sum(b)
    pb = linear_problems.LinearProblem(geom, b=b)
    ot = transport.solve(pb)

    np.testing.assert_array_equal(ot.matrix.shape, (num_a, num_b))
    np.testing.assert_allclose(jnp.sum(ot.matrix, axis=1), ot.a, atol=1e-3)
    np.testing.assert_allclose(jnp.sum(ot.matrix, axis=0), ot.b, atol=1e-3)

  def test_transport_wrong_init(self, rng: jnp.ndarray):
    rngs = jax.random.split(rng, 2)
    num_a, num_b = 23, 17
    x = jax.random.uniform(rngs[0], (num_a, 4))
    y = jax.random.uniform(rngs[1], (num_b, 4))
    geom = pointcloud.PointCloud(x, y, epsilon=1e-2, batch_size=10)
    with pytest.raises(AttributeError, match=r".*has no attribute.*'"):
      transport.solve(geom, x, threshold=1e-3)

    with pytest.raises(ValueError, match="Cannot instantiate a transport"):
      transport.solve('pointcloud', threshold=1e-3)
