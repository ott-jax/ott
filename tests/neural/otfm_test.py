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

from ott.neural.flow_models import flows, models, otfm, utils


class TestOTFlowMatching:

  @pytest.mark.parametrize(("cond_dim", "dl"), [(0, "lin_dl"),
                                                (3, "lin_dl_with_conds"),
                                                (4, "conditional_lin_dl")])
  def test_fm(self, rng: jax.Array, cond_dim: int, dl: str, request):
    dim = 2  # all dataloaders have this dim
    dl = request.getfixturevalue(dl)

    neural_vf = models.VelocityField(
        dim,
        hidden_dims=[5, 5, 5],
        condition_dims=[5, 5, 5] if cond_dim > 0 else None,
    )
    fm = otfm.OTFlowMatching(
        neural_vf,
        flows.ConstantNoiseFlow(0.0),
        match_fn=jax.jit(utils.match_linear),
        rng=rng,
        optimizer=optax.adam(learning_rate=1e-3),
        input_dim=dim,
        condition_dim=cond_dim,
    )

    _logs = fm(dl, n_iters=3)

    batch = next(iter(dl))
    src = jnp.asarray(batch["src_lin"])
    tgt = jnp.asarray(batch["tgt_lin"])
    src_cond = batch.get("src_condition")
    if src_cond is not None:
      src_cond = jnp.asarray(src_cond)

    res_fwd = fm.transport(src, condition=src_cond)
    res_bwd = fm.transport(tgt, t0=1.0, t1=0.0, condition=src_cond)

    # TODO(michalk8): better assertions
    assert jnp.sum(jnp.isnan(res_fwd)) == 0
    assert jnp.sum(jnp.isnan(res_bwd)) == 0
