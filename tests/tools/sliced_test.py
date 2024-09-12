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

from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.tools import sliced


class TestSliced:

  @pytest.mark.parametrize("cost_fn", [costs.PNormP(1.3), None])
  def test_random_projs(self, rng: jax.Array, cost_fn: costs.CostFn):

    n, m, dim, n_proj = 12, 17, 5, 13
    rngs = jax.random.split(rng, 5)
    x = jax.random.uniform(rngs[0], (n, dim))
    y = jax.random.uniform(rngs[1], (m, dim))
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    a /= jnp.sum(a)
    b /= jnp.sum(b)

    # Test non-negative and returns output as needed.
    out = sliced.sliced_wasserstein(
        x,
        y,
        a,
        b,
        cost_fn=cost_fn,
        n_proj=n_proj,
        return_univariate_output_obj=True
    )
    np.testing.assert_array_less(0.0, out[0])
    assert isinstance(out[1], linear.univariate.UnivariateOutput)

    # Test matches standard implementation when using identity for proj_matrix.
    out = sliced.sliced_wasserstein(x, y, proj_fn=lambda x: x, cost_fn=cost_fn)
    geom = pointcloud.PointCloud(x=x, y=y, cost_fn=cost_fn)
    out_lin = jnp.sum(linear.solve_univariate(geom).ot_costs)
    np.testing.assert_allclose(out_lin, out)

    # Test differentiability. We assume uniform samples because makes diff
    # more accurate (avoiding ties, making computations a lot more sensitive).
    def fn(x):
      return sliced.sliced_wasserstein(x, y, cost_fn=cost_fn)

    dx = jax.random.uniform(rngs[4], (n, dim)) - .5
    eps = 1e-4
    out_p = fn(x + eps * dx)
    out_m = fn(x - eps * dx)
    g = jax.grad(fn)(x)
    np.testing.assert_allclose(jnp.sum(g * dx), (out_p - out_m) / (2 * eps))
