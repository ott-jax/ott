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
from typing import Dict, Literal, Optional, Tuple, Union

import pytest

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import optax
from flax import nnx

from ott.neural.methods import flow_matching as fm
from ott.neural.networks.velocity_field import ema, mlp, unet


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
    x = batch["x1"] if reverse else batch["x1"]

    model = mlp.MLP(dim, hidden_dims=[dim] * 3, rngs=nnx.Rngs(0))
    model.eval()

    eval_vf = functools.partial(
        fm.evaluate_velocity_field,
        num_steps=num_steps,
        reverse=reverse,
        max_steps=8,
    )
    eval_vf = nnx.jit(jax.vmap(eval_vf, in_axes=[None, 0]))

    sol = eval_vf(model, x)

    assert sol.ys.shape == (batch_size, 1, dim)

  @pytest.mark.parametrize("num_steps", [None, 3])
  def test_evaluate_vf_save_extra(
      self, rng: jax.Array, num_steps: Optional[int]
  ):
    batch_size, dim = 5, 3
    ode_max_steps, vel_save_steps = 16, 4
    batch = _prepare_batch(rng, shape=(batch_size, dim))
    x = batch["x0"]

    model = mlp.MLP(dim, hidden_dims=[dim] * 2, rngs=nnx.Rngs(0))
    model.eval()

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

    sol = eval_vf(model, x)

    ode_steps = num_steps or ode_max_steps
    assert sol.ys["x_t"].shape == (batch_size, ode_steps, dim)
    assert sol.ys["v_t"].shape == (batch_size, vel_save_steps, dim)

  @pytest.mark.parametrize("drop_last_velocity", [None, False, True])
  @pytest.mark.parametrize("ts", [3, tuple(jnp.linspace(0.0, 1.0, 5).tolist())])
  def test_curvature(
      self, rng: jax.Array, ts: Union[int, jax.Array],
      drop_last_velocity: Optional[bool]
  ):
    dim = 4
    batch = _prepare_batch(rng, shape=(1, dim))
    x = batch["x0"].squeeze(0)
    num_vt = ts if isinstance(ts, int) else len(ts)

    model = mlp.MLP(
        dim, hidden_dims=[dim] * 3, rngs=nnx.Rngs(0), dropout_rate=0.1
    )
    model.eval()

    curv, sol = nnx.jit(fm.curvature, static_argnames=["ts"])(model, x, ts=ts)

    assert jnp.isscalar(curv), curv.shape
    assert jnp.isfinite(curv)

    assert sol.ys["x_t"].shape == (1, dim)
    assert sol.ys["v_t"].shape == (num_vt, dim)

  def test_gaussian_nll(self, rng: jax.Array):
    shape = (8, 8, 3)
    batch = _prepare_batch(rng, shape=(1, *shape))

    model = unet.UNet(
        shape=shape,
        model_channels=32,
        num_res_blocks=1,
        attention_resolutions=(1,),
        channel_mult=(1,),
        rngs=nnx.Rngs(0)
    )
    model.eval()

    x1 = batch["x1"].squeeze(0)

    gaussian_nll_fn = nnx.jit(fm.gaussian_nll, static_argnames=["num_steps"])
    nll, out = gaussian_nll_fn(model, x1, num_steps=16)

    assert jnp.isscalar(nll), nll.shape
    assert jnp.isfinite(nll)

    x0, neg_int01_div_v = out.ys
    assert x0.shape == (1, *shape)
    assert neg_int01_div_v.shape == (1,)
