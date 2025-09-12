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
from typing import Optional, Tuple

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import pointcloud
from ott.geometry import semidiscrete_pointcloud as sdpc


class TestSemidiscretePointCloud:

  @pytest.mark.parametrize("num_samples", [0, 12])
  def test_sample(self, rng: jax.Array, num_samples: int):
    m, d = 16, 3
    rng_data, rng_sample = jr.split(rng, 2)
    y = jr.normal(rng, (m, d))

    geom = sdpc.SemidiscretePointCloud(sampler=jr.normal, y=y)

    if num_samples <= 0:
      with pytest.raises(AssertionError, match=r"must be > 0"):
        _ = geom.sample(rng_sample, num_samples)
      return

    pc = geom.sample(rng_sample, num_samples)

    assert isinstance(pc, pointcloud.PointCloud), type(pc)
    assert pc.shape == (num_samples, m)

  @pytest.mark.parametrize("epsilon", [0.0, None, 1e-2])
  def epsilon(self, rng: jax.Array, epsilon: Optional[float]):
    n, d = 32, 5
    y = jr.normal(rng, (n, d))

    geom = sdpc.SemidiscretePointCloud(
        sampler=jr.normal,
        y=y,
        epsilon=epsilon,
    )

    if epsilon == 0.0:
      assert not geom.is_entropy_regularized, epsilon
      assert geom.epsilon == 0.0, geom.epsilon
    else:
      assert geom.is_entropy_regularized, epsilon
      assert isinstance(geom.epsilon, jax.Array), type(geom.epsilon)

  @pytest.mark.parametrize(("n", "m", "d"), [(11, 12, 4), (1, 32, 7)])
  def test_shape(self, rng: jax.Array, n: int, m: int, d: int):
    rng_data, rng_sample = jr.split(rng, 2)
    y = jr.normal(rng_data, (m, d))
    geom = sdpc.SemidiscretePointCloud(jr.normal, y=y)

    assert geom.shape == (float("inf"), m)
    pc = geom.sample(rng_sample, n)

    assert pc.x.shape == (n, d)
    assert pc.y.shape == (m, d)
    assert pc.shape == (n, m)

  @pytest.mark.parametrize(("epsilon", "dtype"), [(0.0, jnp.float16),
                                                  (None, jnp.bfloat16),
                                                  (0.2, jnp.float32)])
  def test_dtype(
      self, rng: jax.Array, epsilon: Optional[float], dtype: jnp.dtype
  ):
    rng_data, rng_sample = jr.split(rng, 2)
    m, d = 15, 1
    y = jr.normal(rng_data, (m, d), dtype=dtype)
    geom = sdpc.SemidiscretePointCloud(jr.normal, y=y, epsilon=epsilon)
    pc = geom.sample(rng_sample, 12)

    assert geom.dtype == dtype
    assert geom.epsilon.dtype == dtype

    assert pc.dtype == dtype
    assert pc.cost_matrix.dtype == dtype

  def test_jit(self, rng: jax.Array):

    @jax.jit
    def sample(
        geom: sdpc.SemidiscretePointCloud
    ) -> Tuple[sdpc.SemidiscretePointCloud, pointcloud.PointCloud, jax.Array]:
      pc = geom.sample(rng_sample, 32)
      return geom, pc, geom.epsilon

    rng_data, rng_sample = jr.split(rng, 2)
    y = jr.normal(rng_data, (11, 5))

    geom = sdpc.SemidiscretePointCloud(jr.normal, y=y)

    geom2, pc, epsilon = sample(geom)

    np.testing.assert_allclose(geom.epsilon, geom2.epsilon, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(geom.epsilon, epsilon, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(pc.epsilon, epsilon, rtol=1e-5, atol=0.0)
