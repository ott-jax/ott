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
import numpy as np

from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.tools import unreg


class TestHungarian:

  @pytest.mark.parametrize("cost_fn", [costs.PNormP(1.3), None])
  def test_matches_sink(self, rng: jax.Array, cost_fn: Optional[costs.CostFn]):
    n, m, dim = 12, 12, 5
    rng1, rng2 = jax.random.split(rng, 2)
    x, y = gen_data(rng1, n, m, dim)
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=.0005)
    cost_hung, out_hung = jax.jit(unreg.hungarian)(geom)
    out_sink = jax.jit(linear.solve)(geom)
    np.testing.assert_allclose(
        out_sink.primal_cost, cost_hung, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        out_sink.matrix, out_hung.matrix.todense(), rtol=1e-3, atol=1e-3
    )

  @pytest.mark.parametrize("p", [1.3, 2.3])
  def test_wass(self, rng: jax.Array, p: float):
    n, m, dim = 12, 12, 5
    rng1, rng2 = jax.random.split(rng, 2)
    x, y = gen_data(rng1, n, m, dim)
    geom = pointcloud.PointCloud(x, y, cost_fn=costs.EuclideanP(p=p))
    cost_hung, _ = jax.jit(unreg.hungarian)(geom)
    w_p = jax.jit(unreg.wassdis_p)(x, y, p)
    np.testing.assert_allclose(w_p, cost_hung ** 1. / p, rtol=1e-3, atol=1e-3)


def gen_data(rng: jax.Array, n: int, m: int,
             dim: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  rngs = jax.random.split(rng, 4)
  x = jax.random.uniform(rngs[0], (n, dim))
  y = jax.random.uniform(rngs[1], (m, dim))
  return x, y
