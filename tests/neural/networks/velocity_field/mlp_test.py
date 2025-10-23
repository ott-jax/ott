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
import jax.random as jr

from flax import nnx

from ott.neural.networks.velocity_field import mlp


class TestMLP:

  @pytest.mark.parametrize("cond_dim", [0, 3])
  def test_condition(self, rng: jax.Array, cond_dim: int):
    batch_size, dim, cond_dim = 6, 2, 5
    rng_t, rng_x, rng_cond = jr.split(rng, 3)

    t = jr.uniform(rng_t, (batch_size,))
    x = jr.normal(rng_x, (batch_size, dim))
    cond = jr.normal(rng_cond, (batch_size, cond_dim)) if cond_dim else None
    model = mlp.MLP(dim, cond_dim=cond_dim, dropout_rate=0.1, rngs=nnx.Rngs(0))

    v_t = model(t, x, cond=cond, rngs=nnx.Rngs(1))

    assert v_t.shape == (batch_size, dim)

  @pytest.mark.parametrize("num_freqs", [1, 2, 3])
  def test_time_encoder(self, rng: jax.Array, num_freqs: int):
    batch_size = 6
    t = jr.normal(rng, (batch_size,))
    t_emb = mlp._encode_time(t, num_freqs=num_freqs)

    assert t_emb.shape == (batch_size, 2 * num_freqs)
