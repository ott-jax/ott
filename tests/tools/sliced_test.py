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
import functools
from typing import Callable, Optional, Tuple

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.tools import sliced
from tests import _utils

Projector = Callable[[jax.Array, jnp.ndarray], jnp.ndarray]


def custom_proj(
    rng: jax.Array, x: jnp.ndarray, *, n_proj: int = 27
) -> jnp.ndarray:
  dim = x.shape[1]
  proj_m = jr.uniform(rng, (n_proj, dim))
  return (x @ proj_m.T) ** 2


def gen_data(
    rng: jax.Array, n: int, m: int, dim: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  c = _utils.random_clouds(rng, n=n, m=m, dim=dim, offset=0.0)
  return c.a, c.x, c.b, c.y


class TestSliced:

  @pytest.mark.parametrize("proj_fn", [None, custom_proj])
  @pytest.mark.parametrize("cost_fn", [costs.PNormP(1.3), None])
  def test_random_projs(
      self, rng: jax.Array, cost_fn: Optional[costs.CostFn],
      proj_fn: Optional[Projector]
  ):
    n, m, dim, n_proj = 12, 17, 5, 13
    rng_data, rng_w, rng_proj = jr.split(rng, 3)
    a, x, b, y = gen_data(rng_data, n, m, dim)
    weights = jr.uniform(rng_w, (n_proj,))

    if proj_fn is None:
      proj_fn = sliced.random_proj_sphere
    proj_fn = functools.partial(proj_fn, n_proj=n_proj)

    # Test non-negative and returns output as needed.
    cost, out = sliced.sliced_wasserstein(
        x,
        y,
        a,
        b,
        cost_fn=cost_fn,
        proj_fn=proj_fn,
        rng=rng_proj,
        weights=weights
    )
    assert cost > 0.0
    np.testing.assert_array_equal(
        cost, jnp.average(out.ot_costs, weights=weights)
    )

  @pytest.mark.parametrize("cost_fn", [costs.SqPNorm(1.4), None])
  def test_consistency_with_id(
      self, rng: jax.Array, cost_fn: Optional[costs.CostFn]
  ):
    n, m, dim = 11, 12, 4
    _, x, _, y = gen_data(rng, n, m, dim)

    # Test matches standard implementation when using identity.
    cost, _ = sliced.sliced_wasserstein(
        x, y, proj_fn=lambda _, x: x, cost_fn=cost_fn
    )
    geom = pointcloud.PointCloud(x=x, y=y, cost_fn=cost_fn)
    out_lin = jnp.mean(linear.solve_univariate(geom).ot_costs)
    np.testing.assert_allclose(out_lin, cost, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("proj_fn", [None, custom_proj])
  def test_diff(self, rng: jax.Array, proj_fn: Optional[Projector]):
    eps = 1e-4
    n, m, dim = 13, 16, 7
    rng_data, rng_dx = jr.split(rng, 2)
    _, x, _, y = gen_data(rng_data, n, m, dim)

    # Test differentiability. We assume uniform samples because makes diff
    # more accurate (avoiding ties, making computations a lot more sensitive).
    dx = jr.uniform(rng_dx, (n, dim)) - 0.5
    sw = functools.partial(sliced.sliced_wasserstein, proj_fn=proj_fn)
    cost_p, _ = sw(x + eps * dx, y)
    cost_m, _ = sw(x - eps * dx, y)
    g, _ = jax.jit(jax.grad(sw, has_aux=True))(x, y)

    np.testing.assert_allclose(
        jnp.sum(g * dx), (cost_p - cost_m) / (2 * eps), atol=1e-3, rtol=1e-3
    )
