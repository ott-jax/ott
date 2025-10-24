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
from typing import Dict, Literal, Optional, Tuple

import pytest

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import optax
from flax import nnx

from ott.neural.methods import flow_matching as fm
from ott.neural.networks.velocity_field import ema, mlp


def _prepare_batch(
    rng: jax.Array,
    *,
    shape: Tuple[int, ...],
    num_classes: Optional[int] = None,
) -> Dict[Literal["t", "x_t", "v_t", "cond", "x0", "x1"], jax.Array]:
  rng_t, rng_x0, rng_x1, rng_cond = jr.split(rng, 4)
  batch_size, *_ = shape
  x0 = jr.uniform(rng_x0, shape, minval=-1.0, maxval=1.0)
  x1 = jr.normal(rng_x1, shape) + 3
  t = jr.uniform(rng_t, (batch_size,))
  cond = jr.choice(
      rng_cond, num_classes, (batch_size,)
  ) if num_classes else None

  t_expanded = jnp.expand_dims(t, range(1, len(shape)))
  x_t = (1.0 - t_expanded) * x0 + t_expanded * x1
  v_t = x1 - x0

  return {"t": t, "x_t": x_t, "v_t": v_t, "cond": cond, "x0": x0, "x1": x1}


class TestFlowMatching:

  def test_fm_step(self, rng: jax.Array):
    batch_size, dim = 2, 5
    batch = _prepare_batch(rng, shape=(batch_size, dim))
    model = mlp.MLP(dim, rngs=nnx.Rngs(0), dropout_rate=0.1)

    optimizer = optax.adam(1e-3)
    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    fm_step = nnx.jit(chex.assert_max_traces(fm.flow_matching_step, 1))
    step_rngs = nnx.Rngs(0)

    for _ in range(5):
      metrics = fm_step(model, optimizer, batch, rngs=step_rngs)
      assert jnp.isfinite(metrics["loss"])
      assert jnp.isfinite(metrics["grad_norm"])

  def test_ema_callback(self, rng: jax.Array):
    batch_size, dim = 2, 5
    batch = _prepare_batch(rng, shape=(batch_size, dim))
    model = mlp.MLP(dim, rngs=nnx.Rngs(0), dropout_rate=0.1)
    model_ema = ema.EMA(model, decay=0.99)

    optimizer = optax.adam(1e-3)
    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    fm_step = nnx.jit(chex.assert_max_traces(fm.flow_matching_step, 1))
    step_rngs = nnx.Rngs(0)

    for _ in range(5):
      _ = fm_step(
          model, optimizer, batch, model_callback_fn=model_ema, rngs=step_rngs
      )

    ema_state = nnx.state(model_ema)
    ema_state = nnx.to_flat_state(ema_state)
    for k, v in ema_state:
      with pytest.raises(AssertionError):
        np.testing.assert_array_equal(v, 0.0, err_msg=str(k))

  @pytest.mark.parametrize(("num_steps", "reverse"), [(None, False), (4, True)])
  def test_evaluate_vf(
      self, rng: jax.Array, num_steps: Optional[int], reverse: bool
  ):
    batch_size, dim = 2, 3
    batch = _prepare_batch(rng, shape=(batch_size, dim))
    model = mlp.MLP(dim, hidden_dims=[dim] * 3, rngs=nnx.Rngs(0))
    x = batch["x1"] if reverse else batch["x1"]

    eval_vf = functools.partial(
        fm.evaluate_velocity_field,
        num_steps=num_steps,
        reverse=reverse,
        max_steps=8,
    )
    eval_vf = nnx.jit(jax.vmap(eval_vf, in_axes=[None, 0]))

    out = eval_vf(model, x)

    assert out.ys.shape == (2, 1, dim)

  @pytest.mark.parametrize("num_steps", [None, 3])
  def test_evaluate_vf_save_extra(
      self, rng: jax.Array, num_steps: Optional[int]
  ):
    batch_size, dim = 5, 3
    ode_max_steps, vel_save_steps = 16, 4
    batch = _prepare_batch(rng, shape=(batch_size, dim))
    x = batch["x0"]
    model = mlp.MLP(dim, hidden_dims=[dim] * 2, rngs=nnx.Rngs(0))

    save_trajectory_kwargs = {"steps": True}
    save_velocity_kwargs = {"ts": jnp.linspace(0.0, 1.0, vel_save_steps)}
    eval_vf = functools.partial(
        fm.evaluate_velocity_field,
        num_steps=num_steps,
        save_trajectory_kwargs=save_trajectory_kwargs,
        save_velocity_kwargs=save_velocity_kwargs,
        max_steps=ode_max_steps,
    )
    eval_vf = nnx.jit(jax.vmap(eval_vf, in_axes=[None, 0]))

    out = eval_vf(model, x)

    ode_steps = num_steps or ode_max_steps
    assert out.ys["x_t"].shape == (batch_size, ode_steps, dim)
    assert out.ys["v_t"].shape == (batch_size, vel_save_steps, dim)

  def test_curvature(self):
    pass

  def test_gaussian_nll(self):
    pass
