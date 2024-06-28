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
from ott.geometry import costs, pointcloud
from ott.solvers import linear


class TestMMSinkhorn:

  @pytest.mark.fast.with_args(
      a_none=[True, False], b_none=[True, False], only_fast=0
  )
  def test_match_2sinkhorn(self, a_none: bool, b_none: bool, rng: jax.Array):
    """Test consistency of MMSinkhorn for 2 margins vs regular Sinkhorn."""
    n, m, d = 5, 10, 7
    rngs = jax.random.split(rng, 5)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d)) + 1
    if a_none:
      a = None
    else:
      a = jax.random.uniform(rngs[2], (n,))
      a = a.at[0].set(0.0)
      a /= jnp.sum(a)

    if b_none:
      b = None
    else:
      b = jax.random.uniform(rngs[3], (m,))
      b.at[2].set(0.0)
      b /= jnp.sum(b)
    cost_fn = costs.PNormP(1.8)
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
    out = linear.solve(geom, a=a, b=b, threshold=1e-5)

    ab = None if a is None and b is None else [a, b]
    solver = jax.jit(mmsinkhorn.MMSinkhorn(threshold=1e-5))
    out_ms = solver([x, y], ab, cost_fns=cost_fn)
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
    np.testing.assert_allclose(
        out.ent_reg_cost, out_ms.ent_reg_cost, rtol=1e-6, atol=1e-6
    )

  @pytest.mark.fast.with_args(
      a_s_none=[True, False], costs_none=[True, False], only_fast=0
  )
  def test_mm_sinkhorn(self, a_s_none: bool, costs_none: bool, rng: jax.Array):
    """Test correctness of MMSinkhorn for 4 marginals."""
    n_s, d = [13, 5, 10, 3], 7

    rngs = jax.random.split(rng, len(n_s))
    x_s = [jax.random.normal(rng, (n, d)) for rng, n in zip(rngs, n_s)]

    if a_s_none:
      a_s = None
    else:
      a_s = [jax.random.uniform(rng, (n,)) for rng, n in zip(rngs, n_s)]
      a_s = [a / jnp.sum(a) for a in a_s]

    if costs_none:
      cost_fns = None
    else:
      cost_fns = [costs.PNormP(1.5) for _ in range(3)]
      cost_fns += [costs.PNormP(1.1) for _ in range(3)]

    out_ms = jax.jit(mmsinkhorn.MMSinkhorn(norm_error=1.1)
                    )(x_s, a_s, cost_fns=cost_fns)
    assert out_ms.converged
    np.testing.assert_array_equal(out_ms.tensor.shape, n_s)
    for i in range(len(n_s)):
      np.testing.assert_allclose(
          out_ms.marginals[i], out_ms.a_s[i], rtol=1e-4, atol=1e-4
      )

  def test_mm_sinkhorn_diff(self, rng: jax.Array):
    """Test differentiability (Danskin) of MMSinkhorn's ent_reg_cost."""
    n_s, d = [13, 5, 7, 3], 2

    rngs = jax.random.split(rng, 2 * len(n_s) + 1)
    x_s = [
        jax.random.normal(rng, (n, d)) for rng, n in zip(rngs[:len(n_s)], n_s)
    ]

    deltas = [
        jax.random.normal(rng, (n, d)) for rng, n in zip(rngs[len(n_s):], n_s)
    ]
    eps = 1e-3
    x_s_p = [x + eps * delta for x, delta in zip(x_s, deltas)]
    x_s_m = [x - eps * delta for x, delta in zip(x_s, deltas)]

    solver = mmsinkhorn.MMSinkhorn(threshold=1e-5)
    ent_reg = jax.jit(lambda x_s: solver(x_s).ent_reg_cost)
    out_p = ent_reg(x_s_p)
    out_m = ent_reg(x_s_m)
    ent_g = jax.grad(ent_reg)
    g_s = ent_g(x_s)
    first_order = 0
    for g, delta in zip(g_s, deltas):
      first_order += jnp.sum(g * delta)

    np.testing.assert_allclose((out_p - out_m) / (2 * eps),
                               first_order,
                               rtol=1e-3,
                               atol=1e-3)
