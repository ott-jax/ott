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

from ott.experimental import mmsinkhorn
from ott.geometry import pointcloud
from ott.solvers import linear


class TestMMSinkhorn:

  @pytest.mark.fast.with_args(
      a_none=[True, False], b_none=[True, False], only_fast=0
  )
  def test_match_2sinkhorn(self, a_none: bool, b_none: bool, rng: jax.Array):
    """Test consistency of cost/kernel apply to vec."""
    n, m, d = 5, 10, 7
    rngs = jax.random.split(rng, 5)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d)) + 1
    if a_none:
      a = None
    else:
      a = jax.random.uniform(rngs[2], (n,))
      a /= jnp.sum(a)

    if b_none:
      b = None
    else:
      b = jax.random.uniform(rngs[3], (m,))
      b /= jnp.sum(b)
    geom = pointcloud.PointCloud(x, y)
    out = linear.solve(geom, a=a, b=b, threshold=1e-5)

    ab = None if a is None and b is None else [a, b]
    out_ms = jax.jit(mmsinkhorn.MMSinkhorn(threshold=1e-5))([x, y], ab)
    assert out.converged
    assert out_ms.converged

    for axis in [0, 1]:
      f = out.potentials[axis]
      f_ms = out_ms.potentials[axis]
      f -= jnp.mean(f)
      f_ms -= jnp.mean(f_ms)
      np.testing.assert_allclose(f, f_ms, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        out.ot_prob.geom.epsilon, out_ms.epsilon, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(out.matrix, out_ms.tensor, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(out.ent_reg_cost, out_ms.ent_reg_cost)
