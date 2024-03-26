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
from typing import Literal, Optional

import pytest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import optax

from ott.neural.flow_models import flows, genot, models, utils


def data_match_fn(
    src_lin: Optional[jnp.ndarray], tgt_lin: Optional[jnp.ndarray],
    src_quad: Optional[jnp.ndarray], tgt_quad: Optional[jnp.ndarray], *,
    typ: Literal["lin", "quad", "fused"]
) -> jnp.ndarray:
  if typ == "lin":
    return utils.match_linear(x=src_lin, y=tgt_lin)
  if typ == "quad":
    return utils.match_quadratic(xx=src_quad, yy=tgt_quad)
  if typ == "fused":
    return utils.match_quadratic(xx=src_quad, yy=tgt_quad, x=src_lin, y=tgt_lin)
  raise NotImplementedError(f"Unknown type: {typ}.")


class TestGENOT:

  # TODO(michalk8): add conds
  @pytest.mark.parametrize("dl", ["lin_dl", "quad_dl", "fused_dl"])
  def test_genot(self, rng: jax.Array, dl: str, request):
    rng_init, rng_call, rng_data = jax.random.split(rng, 3)
    problem_type = dl.split("_")[0]
    dl = request.getfixturevalue(dl)

    batch = next(iter(dl))
    batch = jtu.tree_map(jnp.asarray, batch)
    src_cond = batch.get("src_condition")

    dims = jtu.tree_map(lambda x: x.shape[-1], batch)
    src_dim = dims.get("src_lin", 0) + dims.get("src_quad", 0)
    tgt_dim = dims.get("tgt_lin", 0) + dims.get("tgt_quad", 0)

    vf = models.VelocityField(
        tgt_dim,
        hidden_dims=[7, 7, 7],
        condition_dims=[7, 7, 7],
    )

    model = genot.GENOT(
        vf,
        flow=flows.ConstantNoiseFlow(0.0),
        data_match_fn=functools.partial(data_match_fn, typ=problem_type),
        source_dim=src_dim,
        target_dim=tgt_dim,
        condition_dim=None if src_cond is None else src_cond.shape[-1],
        rng=rng_init,
        optimizer=optax.adam(learning_rate=1e-4),
    )

    _logs = model(dl, n_iters=3, rng=rng_call)

    src = jax.random.normal(rng_data, (3, src_dim))
    res = model.transport(src, condition=src_cond)

    assert jnp.sum(jnp.isnan(res)) == 0
    assert res.shape[-1] == tgt_dim
