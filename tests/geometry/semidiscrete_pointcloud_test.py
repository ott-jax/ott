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
import jax.random as jr

from ott.geometry import pointcloud
from ott.geometry import semidiscrete_pointcloud as sdpc


class TestSemidiscretePointCloud:

  @pytest.mark.parametrize("num_samples", [0, 12])
  def test_sample(self, rng: jax.Array, num_samples: int):
    n, d = 128, 3
    rng_data, rng_sample = jr.split(rng, 2)
    y = jr.normal(rng, (n, d))

    geom = sdpc.SemidiscretePointCloud(sampler=jr.normal, y=y)

    if num_samples <= 0:
      with pytest.raises(AssertionError, match=r"must be > 0"):
        _ = geom.sample(rng_sample, num_samples)
      return

    pc = geom.sample(rng_sample, num_samples)

    assert isinstance(pc, pointcloud.PointCloud), type(pc)
    assert pc.shape == (num_samples, n)

  @pytest.mark.parametrize("epsilon", [0.0, None, 1e-2])
  def epsilon(self, rng: jax.Array, epsilon: Optional[float]):
    n, d = 128, 3
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

  def test_shape(self, rng: jax.Array):
    pass

  def test_dtype(self, rng: jax.Array):
    pass

  def test_jit(self, rng: jax.Array):

    @jax.jit
    def fn(geom):

      return geom
