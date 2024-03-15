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

import optax

from ott.neural.flow_models import flows, genot, models, utils


def data_match_fn(
    src_lin: jnp.ndarray, tgt_lin: jnp.ndarray, src_quad: jnp.ndarray,
    tgt_quad: jnp.ndarray
):
  # TODO(michalk8): extend for GW/FGW
  return utils.match_linear(src_lin, tgt_lin)


class TestGENOT:

  # TODO(michalk8): test gw/fgw, k, etc.
  @pytest.mark.parametrize(("cond_dim", "dl"), [(2, "lin_dl")])
  def test_genot2(self, rng: jax.Array, cond_dim: int, dl: str, request):
    rng_init, rng_call = jax.random.split(rng)
    input_dim, hidden_dim = 2, 7
    dl = request.getfixturevalue(dl)

    vf = models.VelocityField(
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        # TODO(michalk8): the source is the condition
        condition_dim=cond_dim,
    )

    model = genot.GENOT(
        vf,
        flow=flows.ConstantNoiseFlow(0.0),
        data_match_fn=data_match_fn,
        rng=rng_init,
        optimizer=optax.adam(learning_rate=1e-3),
        input_dim=input_dim,
        condition_dim=cond_dim,
    )

    _logs = model(dl, n_iters=3, rng=rng_call)

    # TODO(michalk8): generalize for gw/fgw
    batch = next(iter(dl))
    src = jnp.asarray(batch["src_lin"])
    src_cond = batch.get("src_condition")
    if src_cond is not None:
      src_cond = jnp.asarray(src_cond)

    res = model.transport(src, condition=src_cond)

    assert jnp.sum(jnp.isnan(res)) == 0
