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
import jax.random as jr

from flax import nnx

from ott.neural.networks.velocity_field import unet


def _prepare_inputs(
    rng: jax.Array,
    *,
    shape: Tuple[int, ...],
    num_classes: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
  rng_t, rng_x, rng_cond = jr.split(rng, 3)
  batch_size, *_ = shape
  t = jr.uniform(rng_t, (batch_size,))
  x = jr.uniform(rng_x, shape, minval=-1.0, maxval=1.0)
  cond = jr.choice(
      rng_cond, num_classes, (batch_size,)
  ) if num_classes else None
  return t, x, cond


class TestUNet:

  def test_condition(self, rng: jax.Array):
    num_classes = 10
    shape = (2, 8, 8, 3)
    t, x, cond = _prepare_inputs(rng, shape=shape, num_classes=num_classes)
    model = unet.UNet(
        shape=shape[1:],
        model_channels=32,
        num_res_blocks=1,
        attention_resolutions=(),
        channel_mult=(1,),
        dropout=0.1,
        rngs=nnx.Rngs(0),
    )

    v_t = nnx.jit(model)(t, x, cond, rngs=nnx.Rngs(1))

    assert v_t.shape == shape

  @pytest.mark.parametrize("param_dtype", [jnp.bfloat16, jnp.float32])
  def test_param_dtype(self, rng: jax.Array, param_dtype: jnp.dtype):
    shape = (1, 32, 32, 4)
    t, x, _ = _prepare_inputs(rng, shape=shape)
    model = unet.UNet(
        shape=shape[1:],
        model_channels=32,
        num_res_blocks=1,
        attention_resolutions=(1,),
        channel_mult=(1.0,),
        dropout=0.1,
        param_dtype=param_dtype,
        rngs=nnx.Rngs(0),
    )

    state = nnx.to_flat_state(nnx.state(model))
    for k, v in state:
      assert v[...].dtype == param_dtype, k

    v_t = nnx.jit(model)(t, x, rngs=nnx.Rngs(1))
    assert v_t.dtype == t.dtype
    assert v_t.shape == shape

  @pytest.mark.parametrize("conv_resample", [False, True])
  @pytest.mark.parametrize("resblock_updown", [False, True])
  def test_flags(
      self, rng: jax.Array, conv_resample: bool, resblock_updown: bool
  ):

    @nnx.jit
    def run_model(model: nnx.Module, t: jax.Array, x: jax.Array) -> jax.Array:
      return model(t, x)

    shape = (1, 16, 16, 3)
    t, x, _ = _prepare_inputs(rng, shape=shape)
    model = unet.UNet(
        shape=shape[1:],
        model_channels=32,
        num_res_blocks=2,
        attention_resolutions=(1,),
        channel_mult=(1, 2, 3, 4),
        conv_resample=conv_resample,
        resblock_updown=resblock_updown,
        rngs=nnx.Rngs(0),
    )

    v_t = run_model(model, t, x)

    assert v_t.shape == shape
