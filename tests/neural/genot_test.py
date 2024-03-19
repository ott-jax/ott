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
from typing import Literal

import pytest

import jax
import jax.numpy as jnp

import optax

from ott.neural.flow_models import flows, genot, models, utils


def data_match_fn(
    src_lin: jnp.ndarray, tgt_lin: jnp.ndarray, src_quad: jnp.ndarray,
    tgt_quad: jnp.ndarray, *, type: Literal["linear", "quadratic", "fused"]
):
  if type == "linear":
    return utils.match_linear(x=src_lin, y=tgt_lin)
  if type == "quadratic":
    return utils.match_quadratic(xx=src_quad, yy=tgt_quad)
  if type == "fused":
    return utils.match_quadratic(xx=src_quad, yy=tgt_quad, x=src_lin, y=tgt_lin)
  raise NotImplementedError(f"Unknown type: {type}")


class TestGENOT:

  # TODO(michalk8): test gw/fgw, k, etc.
  @pytest.mark.parametrize("dl", ["lin_dl", "conditional_lin_dl"])
  def test_genot_linear(self, rng: jax.Array, dl: str, request):
    rng_init, rng_call = jax.random.split(rng)
    hidden_dim = 7
    dl = request.getfixturevalue(dl)

    batch = next(iter(dl))
    src = jnp.asarray(batch["src_lin"])
    tgt = jnp.asarray(batch["tgt_lin"])
    src_cond = batch.get("src_condition")
    if src_cond is not None:
      src_cond = jnp.asarray(src_cond)
    src_dim = src.shape[-1]
    tgt_dim = tgt.shape[-1]
    cond_dim = src_cond.shape[-1] if src_cond is not None else 0

    vf = models.VelocityField(
        hidden_dim=hidden_dim,
        output_dim=tgt_dim,
        condition_dim=src_dim + cond_dim,
    )

    data_mfn = functools.partial(data_match_fn, type="linear")

    model = genot.GENOT(
        vf,
        flow=flows.ConstantNoiseFlow(0.0),
        data_match_fn=data_mfn,
        source_dim=src_dim,
        target_dim=tgt_dim,
        condition_dim=cond_dim,
        rng=rng_init,
        optimizer=optax.adam(learning_rate=1e-4),
    )

    _logs = model(dl, n_iters=3, rng=rng_call)
    res = model.transport(src, condition=src_cond)

    assert jnp.sum(jnp.isnan(res)) == 0
    assert res.shape[-1] == tgt_dim

  @pytest.mark.parametrize("dl", ["quad_dl", "conditional_quad_dl"])
  def test_genot_quad(self, rng: jax.Array, dl: str, request):
    rng_init, rng_call = jax.random.split(rng)
    hidden_dim = 7
    dl = request.getfixturevalue(dl)

    batch = next(iter(dl))
    src = jnp.asarray(batch["src_quad"])
    tgt = jnp.asarray(batch["tgt_quad"])
    src_cond = batch.get("src_condition")
    if src_cond is not None:
      src_cond = jnp.asarray(src_cond)
    src_dim = src.shape[-1]
    tgt_dim = tgt.shape[-1]
    cond_dim = src_cond.shape[-1] if src_cond is not None else 0

    vf = models.VelocityField(
        hidden_dim=hidden_dim,
        output_dim=tgt_dim,
        condition_dim=src_dim + cond_dim,
    )

    data_mfn = functools.partial(data_match_fn, type="quadratic")

    model = genot.GENOT(
        vf,
        flow=flows.ConstantNoiseFlow(0.0),
        data_match_fn=data_mfn,
        source_dim=src_dim,
        target_dim=tgt_dim,
        condition_dim=cond_dim,
        rng=rng_init,
        optimizer=optax.adam(learning_rate=1e-4),
    )

    _logs = model(dl, n_iters=3, rng=rng_call)
    res = model.transport(src, condition=src_cond)

    assert jnp.sum(jnp.isnan(res)) == 0
    assert res.shape[-1] == tgt_dim

  @pytest.mark.parametrize("dl", ["fused_dl", "conditional_fused_dl"])
  def test_genot_fused(self, rng: jax.Array, dl: str, request):
    rng_init, rng_call = jax.random.split(rng)
    hidden_dim = 7
    dl = request.getfixturevalue(dl)

    batch = next(iter(dl))
    src_lin = jnp.asarray(batch["src_lin"])
    tgt_lin = jnp.asarray(batch["tgt_lin"])
    src_quad = jnp.asarray(batch["src_quad"])
    tgt_quad = jnp.asarray(batch["tgt_quad"])
    src_cond = batch.get("src_condition")
    if src_cond is not None:
      src_cond = jnp.asarray(src_cond)
    src_dim = src_lin.shape[-1] + src_quad.shape[-1]
    tgt_dim = tgt_lin.shape[-1] + tgt_quad.shape[-1]
    cond_dim = src_cond.shape[-1] if src_cond is not None else 0

    vf = models.VelocityField(
        hidden_dim=hidden_dim,
        output_dim=tgt_dim,
        condition_dim=src_dim + cond_dim,
    )

    data_mfn = functools.partial(data_match_fn, type="fused")

    model = genot.GENOT(
        vf,
        flow=flows.ConstantNoiseFlow(0.0),
        data_match_fn=data_mfn,
        source_dim=src_dim,
        target_dim=tgt_dim,
        condition_dim=cond_dim,
        rng=rng_init,
        optimizer=optax.adam(learning_rate=1e-4),
    )

    _logs = model(dl, n_iters=3, rng=rng_call)
    src = jnp.concatenate([src_lin, src_quad], axis=-1)
    res = model.transport(src, condition=src_cond)

    assert jnp.sum(jnp.isnan(res)) == 0
    assert res.shape[-1] == tgt_dim
