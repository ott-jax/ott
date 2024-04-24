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
from typing import Literal

import pytest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import optax

from ott.neural.methods.flows import dynamics, genot
from ott.neural.networks import velocity_field
from ott.solvers import utils as solver_utils


def get_match_fn(typ: Literal["lin", "quad", "fused"]):
  if typ == "lin":
    return solver_utils.match_linear
  if typ == "quad":
    return solver_utils.match_quadratic
  if typ == "fused":
    return solver_utils.match_quadratic
  raise NotImplementedError(typ)


class TestGENOT:

  @pytest.mark.parametrize(
      "dl", [
          "lin_dl", "quad_dl", "fused_dl", "lin_cond_dl", "quad_cond_dl",
          "fused_cond_dl"
      ]
  )
  def test_genot(self, rng: jax.Array, dl: str, request):
    rng_init, rng_call, rng_data = jax.random.split(rng, 3)
    problem_type = dl.split("_")[0]
    dl = request.getfixturevalue(dl)

    src_dim = dl.lin_dim + dl.quad_src_dim
    tgt_dim = dl.lin_dim + dl.quad_tgt_dim
    cond_dim = dl.cond_dim

    vf = velocity_field.VelocityField(
        hidden_dims=[7, 7, 7],
        output_dims=[15, tgt_dim],
        condition_dims=None if cond_dim is None else [1, 3, 2],
        dropout_rate=0.5,
    )
    model = genot.GENOT(
        vf,
        flow=dynamics.ConstantNoiseFlow(0.0),
        data_match_fn=get_match_fn(problem_type),
        source_dim=src_dim,
        target_dim=tgt_dim,
        condition_dim=cond_dim,
        rng=rng_init,
        optimizer=optax.adam(learning_rate=1e-4),
    )

    _logs = model(dl.loader, n_iters=2, rng=rng_call)

    batch = next(iter(dl.loader))
    batch = jtu.tree_map(jnp.asarray, batch)
    src_cond = batch.get("src_condition")
    batch_size = 4 if src_cond is None else src_cond.shape[0]
    src = jax.random.normal(rng_data, (batch_size, src_dim))

    res = model.transport(src, condition=src_cond)

    assert len(_logs["loss"]) == 2
    np.testing.assert_array_equal(jnp.isfinite(res), True)
    assert res.shape == (batch_size, tgt_dim)
