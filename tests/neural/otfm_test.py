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

import pytest

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader

import optax

from ott.neural.data import datasets
from ott.neural.flow_models import flows, models, otfm, samplers, utils


class TestOTFlowMatching:

  def test_fm(self, lin_dl: DataLoader):
    input_dim = 2
    neural_vf = models.VelocityField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    fm = otfm.OTFlowMatching(
        input_dim,
        neural_vf,
        flow=flows.ConstantNoiseFlow(0.0),
        time_sampler=samplers.uniform_sampler,
        match_fn=jax.jit(utils.match_linear),
        optimizer=optax.adam(learning_rate=1e-3),
    )

    _logs = fm(lin_dl, n_iters=2)

    for batch in lin_dl:
      src = jnp.asarray(batch["src_lin"])
      tgt = jnp.asarray(batch["tgt_lin"])
      break

    res_fwd = fm.transport(src)
    res_bwd = fm.transport(tgt, t0=1.0, t1=0.0)

    # TODO(michalk8): better assertions
    assert jnp.sum(jnp.isnan(res_fwd)) == 0
    assert jnp.sum(jnp.isnan(res_bwd)) == 0

  def test_fm_with_conds(self, lin_dl_with_conds: DataLoader):
    input_dim, cond_dim = 2, 1
    neural_vf = models.VelocityField(
        output_dim=input_dim,
        condition_dim=cond_dim,
        latent_embed_dim=5,
    )
    fm = otfm.OTFlowMatching(
        2,
        neural_vf,
        flow=flows.BrownianNoiseFlow(0.12),
        time_sampler=functools.partial(samplers.uniform_sampler, offset=1e-5),
        match_fn=jax.jit(utils.match_linear),
        optimizer=optax.adam(learning_rate=1e-3),
    )

    _logs = fm(lin_dl_with_conds, n_iters=2)

    for batch in lin_dl_with_conds:
      src = jnp.asarray(batch["src_lin"])
      tgt = jnp.asarray(batch["tgt_lin"])
      src_cond = jnp.asarray(batch["src_condition"])
      break

    res_fwd = fm.transport(src, condition=src_cond)
    res_bwd = fm.transport(tgt, condition=src_cond, t0=1.0, t1=0.0)

    # TODO(michalk8): better assertions
    assert jnp.sum(jnp.isnan(res_fwd)) == 0
    assert jnp.sum(jnp.isnan(res_bwd)) == 0

  @pytest.mark.parametrize("rank", [-1, 10])
  def test_fm_conditional_loader(
      self, rank: int, conditional_lin_dl: datasets.ConditionalLoader
  ):
    input_dim, cond_dim = 2, 0
    neural_vf = models.VelocityField(
        output_dim=input_dim,
        condition_dim=cond_dim,
        latent_embed_dim=5,
    )
    fm = otfm.OTFlowMatching(
        input_dim,
        neural_vf,
        flow=flows.ConstantNoiseFlow(13.0),
        time_sampler=samplers.uniform_sampler,
        match_fn=jax.jit(functools.partial(utils.match_linear, rank=rank)),
        optimizer=optax.adam(learning_rate=1e-3),
    )

    _logs = fm(conditional_lin_dl, n_iters=2)

    for batch in conditional_lin_dl:
      src = jnp.asarray(batch["src_lin"])
      tgt = jnp.asarray(batch["tgt_lin"])
      src_cond = jnp.asarray(batch["src_condition"])
      break

    res_fwd = fm.transport(src, condition=src_cond)
    res_bwd = fm.transport(tgt, condition=src_cond, t0=1.0, t1=0.0)

    # TODO(michalk8): better assertions
    assert jnp.sum(jnp.isnan(res_fwd)) == 0
    assert jnp.sum(jnp.isnan(res_bwd)) == 0
