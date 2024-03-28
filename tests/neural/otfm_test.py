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
import jax.tree_util as jtu

import optax

from ott.neural.flow_models import flows, models, otfm, utils


class TestOTFlowMatching:

  @pytest.mark.parametrize("dl", ["lin_dl", "lin_cond_dl"])
  def test_otfm(self, rng: jax.Array, dl: str, request):
    dl = request.getfixturevalue(dl)
    dim, cond_dim = dl.lin_dim, dl.cond_dim

    neural_vf = models.VelocityField(
        hidden_dims=[5, 5, 5],
        output_dims=[7, dim],
        condition_dims=None if cond_dim is None else [4, 3, 2],
    )
    fm = otfm.OTFlowMatching(
        neural_vf,
        flows.ConstantNoiseFlow(0.0),
        match_fn=jax.jit(utils.match_linear),
        rng=rng,
        optimizer=optax.adam(learning_rate=1e-3),
        condition_dim=cond_dim,
    )

    _logs = fm(dl.loader, n_iters=3)

    batch = next(iter(dl.loader))
    batch = jtu.tree_map(jnp.asarray, batch)
    src_cond = batch.get("src_condition")

    res_fwd = fm.transport(batch["src_lin"], condition=src_cond)
    res_bwd = fm.transport(batch["tgt_lin"], t0=1.0, t1=0.0, condition=src_cond)

    # TODO(michalk8): better assertions
    assert jnp.sum(jnp.isnan(res_fwd)) == 0
    assert jnp.sum(jnp.isnan(res_bwd)) == 0
