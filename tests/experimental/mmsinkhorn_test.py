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
from typing import List, Optional

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
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
    rngs = jr.split(rng, 5)
    x = jr.normal(rngs[0], (n, d))
    y = jr.normal(rngs[1], (m, d)) + 1
    if a_none:
      a = None
    else:
      a = jr.uniform(rngs[2], (n,))
      a = a.at[0].set(0.0)
      a /= jnp.sum(a)

    if b_none:
      b = None
    else:
      b = jr.uniform(rngs[3], (m,))
      b = b.at[2].set(0.0)
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
      # the potentials are `-inf` at zero weights
      supp = jnp.isfinite(f)
      np.testing.assert_array_equal(jnp.isfinite(f_ms), supp)
      f, f_ms = f[supp], f_ms[supp]
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

    rngs = jr.split(rng, len(n_s))
    x_s = [jr.normal(rng, (n, d)) for rng, n in zip(rngs, n_s)]

    if a_s_none:
      a_s = None
    else:
      a_s = [jr.uniform(rng, (n,)) for rng, n in zip(rngs, n_s)]
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
          out_ms.marginals[i], out_ms.a_s[i], rtol=1e-3, atol=1e-3
      )

  def test_mm_sinkhorn_diff(self, rng: jax.Array):
    """Test differentiability (Danskin) of MMSinkhorn's ent_reg_cost."""
    n_s, d = [13, 5, 7, 3], 2

    rngs = jr.split(rng, 2 * len(n_s) + 1)
    x_s = [jr.normal(rng, (n, d)) for rng, n in zip(rngs[:len(n_s)], n_s)]

    deltas = [jr.normal(rng, (n, d)) for rng, n in zip(rngs[len(n_s):], n_s)]
    eps = 1e-3
    x_s_p = [x + eps * delta for x, delta in zip(x_s, deltas)]
    x_s_m = [x - eps * delta for x, delta in zip(x_s, deltas)]

    solver = mmsinkhorn.MMSinkhorn(threshold=1e-5)
    # the default epsilon is a constant when differentiating
    epsilon = solver(x_s).epsilon
    ent_reg = jax.jit(lambda x_s: solver(x_s).ent_reg_cost)
    ent_reg_fixed = jax.jit(
        lambda x_s: solver(x_s, epsilon=epsilon).ent_reg_cost
    )
    out_p = ent_reg_fixed(x_s_p)
    out_m = ent_reg_fixed(x_s_m)
    ent_g = jax.grad(ent_reg)
    g_s = ent_g(x_s)
    first_order = 0
    for g, delta in zip(g_s, deltas):
      first_order += jnp.sum(g * delta)

    np.testing.assert_allclose((out_p - out_m) / (2 * eps),
                               first_order,
                               rtol=1e-3,
                               atol=1e-3)

  @pytest.mark.parametrize("use_danskin", [True, False])
  def test_mm_sinkhorn_diff_zero_weights(
      self, rng: jax.Array, use_danskin: bool
  ):
    """Test gradients of MMSinkhorn's ent_reg_cost with zero weights."""
    n_s, d = [5, 6, 7], 2
    x_s = [jr.normal(rng, (n, d)) for rng, n in zip(jr.split(rng, 3), n_s)]
    a_s = [jnp.ones(n) / n for n in n_s]
    a_s[0] = jnp.array([0.0, 0.25, 0.25, 0.25, 0.25])
    solver = mmsinkhorn.MMSinkhorn(threshold=1e-5, use_danskin=use_danskin)
    grad_fn = jax.grad(
        lambda x_s, a_s, eps: solver(x_s, a_s, epsilon=eps).ent_reg_cost,
        argnums=(1, 2)
    )

    g_a_s, g_eps = grad_fn(x_s, a_s, 1.0)
    g_a_s_supp, g_eps_supp = grad_fn([x_s[0][1:], *x_s[1:]],
                                     [a_s[0][1:], *a_s[1:]], 1.0)

    assert jnp.all(jnp.isfinite(g_a_s[0]))
    np.testing.assert_allclose(g_a_s[0][1:], g_a_s_supp[0], rtol=1e-4)
    np.testing.assert_allclose(g_eps, g_eps_supp, rtol=1e-4)

  def test_tensor_grad_zero_weights(self, rng: jax.Array):
    n_s, d = [5, 6, 7], 2
    rng_x, rng_v, rng_w = jr.split(rng, 3)
    x_s = [jr.normal(rng, (n, d)) for rng, n in zip(jr.split(rng_x, 3), n_s)]
    v_s = [jr.normal(rng, (n, d)) for rng, n in zip(jr.split(rng_v, 3), n_s)]
    a_s = [jnp.ones(n) / n for n in n_s]
    a_s[0] = jnp.array([0.0, 0.25, 0.25, 0.25, 0.25])
    w = jr.normal(rng_w, n_s)
    solver = mmsinkhorn.MMSinkhorn(threshold=1e-6)

    def loss(x_s: List[jnp.ndarray], epsilon: Optional[float] = None) -> float:
      return jnp.vdot(w, solver(x_s, a_s, epsilon=epsilon).tensor)

    # the default epsilon is a constant when differentiating
    epsilon = solver(x_s, a_s).epsilon
    grad = jax.grad(loss)(x_s)
    h = 1e-3
    loss_p = loss([x + h * v for x, v in zip(x_s, v_s)], epsilon)
    loss_m = loss([x - h * v for x, v in zip(x_s, v_s)], epsilon)

    actual = sum(jnp.vdot(g, v) for g, v in zip(grad, v_s))
    np.testing.assert_allclose(actual, (loss_p - loss_m) / (2 * h), rtol=1e-3)

  @pytest.mark.parametrize("min_iterations", [30, 34])
  def test_min_iterations(self, rng: jax.Array, min_iterations: int):
    """Test that errors are recorded once ``min_iterations`` are done."""
    x_s = [jr.normal(rng, (7, 2)) for rng in jr.split(rng, 3)]

    out = mmsinkhorn.MMSinkhorn(min_iterations=min_iterations)(x_s)

    errors = out.errors[:, 0]
    first = (min_iterations - 1) // 10
    np.testing.assert_array_equal(errors[:first], jnp.inf)
    assert 0 < errors[first] < jnp.inf
    assert out.converged

  @pytest.mark.parametrize(("max_iterations", "converged"), [(15, False),
                                                             (100, True)])
  def test_scan_converged(
      self, rng: jax.Array, max_iterations: int, converged: bool
  ):
    """Test the converged flag when running a scan loop."""
    x_s = [jr.normal(rng, (7, 2)) for rng in jr.split(rng, 3)]
    solver = mmsinkhorn.MMSinkhorn(
        min_iterations=max_iterations, max_iterations=max_iterations
    )

    out = solver(x_s)

    assert out.converged == converged

  @pytest.mark.usefixtures("enable_x64")
  def test_float32_with_x64(self, rng: jax.Array):
    n_s, d = [5, 6, 7], 2
    x_s = [
        jr.normal(rng, (n, d), dtype=jnp.float32)
        for rng, n in zip(jr.split(rng, 3), n_s)
    ]

    out = mmsinkhorn.MMSinkhorn()(x_s)

    for arr in (*out.potentials, out.errors, out.ent_reg_cost, out.tensor):
      assert arr.dtype == jnp.float32
