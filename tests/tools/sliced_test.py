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

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.tools import sliced


class TestSliced:

  def test_random_projs(self, rng):
    cost_fn = costs.PNormP(1.3)
    n, m, dim = 12, 17, 5
    rngs = jax.random.split(rng, 4)
    x = jax.random.uniform(rngs[0], (n, dim))
    y = jax.random.uniform(rngs[1], (m, dim))
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    a /= jnp.sum(a)
    b /= jnp.sum(b)

    # Test non-negative
    out = sliced.sliced_w(
        x,
        y,
        a,
        b,
        rng=None,
        n_proj=13,
        random_direction_generator=jax.random.uniform,
        cost_fn=cost_fn
    )
    np.testing.assert_array_less(0.0, out)

    # Test matches standard implementation when using identity for proj_matrix.
    out = sliced.sliced_w(x, y, a, b, proj_matrix=jnp.eye(dim), cost_fn=cost_fn)
    geom = pointcloud.PointCloud(x=x, y=y, cost_fn=cost_fn)
    out_lin = jnp.sum(linear.solve_univariate(geom=geom, a=a, b=b).ot_costs)
    np.testing.assert_approx_equal(out_lin, out)
