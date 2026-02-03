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
import numpy as np

from flax import nnx

from ott.neural.networks.velocity_field import ema, mlp


class TestEMA:

  @pytest.mark.parametrize("dropout_rate", [0.0, 0.5])
  def test_init(self, dropout_rate: float):
    model = mlp.MLP(
        2, rngs=nnx.Rngs(0), hidden_dims=[5], dropout_rate=dropout_rate
    )
    model_ema = ema.init_ema(model)

    ema_state = nnx.to_flat_state(nnx.state(model_ema))
    for k, v in ema_state:
      np.testing.assert_array_equal(v[...], 0.0, err_msg=str(k))

  @pytest.mark.parametrize(("dropout_rate", "decay"), [(0.0, 0.3), (0.2, 0.9)])
  def test_update(self, dropout_rate: float, decay: float):

    @nnx.jit
    def update_ema(model: nnx.Module, model_ema: nnx.Module) -> None:
      model_ema(model)

    model = mlp.MLP(
        5, hidden_dims=[7, 3], rngs=nnx.Rngs(0), dropout_rate=dropout_rate
    )

    # first step EMA update
    expected_ema_state = jax.tree.map(
        lambda p_model: (1.0 - decay) * p_model, nnx.state(model)
    )
    expected_ema_state = nnx.to_flat_state(expected_ema_state)

    model_ema = ema.EMA(model, decay=decay)
    update_ema(model, model_ema)
    # nnx.jit(model_ema)(model)  # throws TraceError below

    ema_state = nnx.to_flat_state(nnx.state(model_ema))
    for (k_act, act), (k_exp, exp) in zip(ema_state, expected_ema_state):
      k_act = k_act[1:]  # drop the `ema` prefix
      assert k_act == k_exp, (k_act, k_exp)
      np.testing.assert_array_equal(act[...], exp[...], err_msg=str(k_act))
