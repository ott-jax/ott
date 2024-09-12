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
from typing import Callable, Optional, Tuple

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.tools import sliced

Projector = Callable[[jnp.ndarray, int, jax.Array], jnp.ndarray]


def custom_proj(
    x: jnp.ndarray,
    rng: Optional[jax.Array] = None,
    n_proj: int = 27
) -> jnp.ndarray:
  dim = x.shape[1]
  rng = jax.random.PRNGKey(42) if rng is None else rng
  proj_m = jax.random.uniform(rng, (n_proj, dim))
  return (x @ proj_m.T) ** 2


def gen_data(
    rng: jax.Array, n: int, m: int, dim: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  rngs = jax.random.split(rng, 4)
  x = jax.random.uniform(rngs[0], (n, dim))
  y = jax.random.uniform(rngs[1], (m, dim))
  a = jax.random.uniform(rngs[2], (n,))
  b = jax.random.uniform(rngs[3], (m,))
  a /= jnp.sum(a)
  b /= jnp.sum(b)
  return a, x, b, y


class TestSliced:

  @pytest.mark.parametrize("proj_fn", [None, custom_proj])
  @pytest.mark.parametrize("cost_fn", [costs.PNormP(1.3), None])
  def test_random_projs(
      self, rng: jax.Array, cost_fn: Optional[costs.CostFn],
      proj_fn: Optional[Projector]
  ):
    n, m, dim, n_proj = 12, 17, 5, 13
    rng1, rng2 = jax.random.split(rng, 2)
    a, x, b, y = gen_data(rng1, n, m, dim)

    # Test non-negative and returns output as needed.
    cost, out = sliced.sliced_wasserstein(
        x, y, a, b, cost_fn=cost_fn, proj_fn=proj_fn, n_proj=n_proj, rng=rng2
    )
    assert cost > 0.0
    np.testing.assert_array_equal(cost, jnp.sum(out.ot_costs))

  @pytest.mark.parametrize("cost_fn", [costs.SqPNorm(1.4), None])
  def test_consistency_with_id(
      self, rng: jax.Array, cost_fn: Optional[costs.CostFn]
  ):
    n, m, dim = 11, 12, 4
    a, x, b, y = gen_data(rng, n, m, dim)

    # Test matches standard implementation when using identity.
    cost, _ = sliced.sliced_wasserstein(
        x, y, proj_fn=lambda x: x, cost_fn=cost_fn
    )
    geom = pointcloud.PointCloud(x=x, y=y, cost_fn=cost_fn)
    out_lin = jnp.sum(linear.solve_univariate(geom).ot_costs)
    np.testing.assert_allclose(out_lin, cost, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("proj_fn", [None, custom_proj])
  def test_diff(self, rng: jax.Array, proj_fn: Optional[Projector]):
    eps = 1e-4
    n, m, dim = 13, 16, 7
    a, x, b, y = gen_data(rng, n, m, dim)

    # Test differentiability. We assume uniform samples because makes diff
    # more accurate (avoiding ties, making computations a lot more sensitive).
    dx = jax.random.uniform(rng, (n, dim)) - 0.5
    cost_p, _ = sliced.sliced_wasserstein(x + eps * dx, y)
    cost_m, _ = sliced.sliced_wasserstein(x - eps * dx, y)
    g, _ = jax.jit(jax.grad(sliced.sliced_wasserstein, has_aux=True))(x, y)

    np.testing.assert_allclose(
        jnp.sum(g * dx), (cost_p - cost_m) / (2 * eps), atol=1e-3, rtol=1e-3
    )
