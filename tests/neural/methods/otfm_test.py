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
import numpy as np

import optax

from ott.neural.methods.flows import dynamics, otfm
from ott.neural.networks import velocity_field
from ott.solvers import utils as solver_utils


class TestOTFlowMatching:

  @pytest.mark.parametrize("dl", ["lin_dl", "lin_cond_dl"])
  def skip_test_otfm(self, rng: jax.Array, dl: str, request):
    dl = request.getfixturevalue(dl)
    dim, cond_dim = dl.lin_dim, dl.cond_dim

    vf = velocity_field.VelocityField(
        hidden_dims=[5, 5, 5],
        output_dims=[7, dim],
        condition_dims=None if cond_dim is None else [4, 3, 2],
    )
    fm = otfm.OTFlowMatching(
        vf,
        dynamics.ConstantNoiseFlow(0.0),
        match_fn=jax.jit(solver_utils.match_linear),
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

    assert len(_logs["loss"]) == 3

    assert res_fwd.shape == batch["src_lin"].shape
    assert res_bwd.shape == batch["tgt_lin"].shape
    np.testing.assert_array_equal(jnp.isfinite(res_fwd), True)
    np.testing.assert_array_equal(jnp.isfinite(res_bwd), True)
