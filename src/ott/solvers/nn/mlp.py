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

from typing import Sequence, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
import optax
from flax.training import train_state

__all__ = ["PotentialGradientMLP"]


class PotentialGradientMLP(nn.Module):
  dim_hidden: Sequence[int]

  @property
  def provides_gradient(self):
    return True

  @nn.compact
  def __call__(self, x):
    single = x.ndim == 1
    if single:
      x = jnp.expand_dims(x, 0)
    assert x.ndim == 2
    n_input = x.shape[-1]

    z = x
    for n_hidden in self.dim_hidden:
      Wx = nn.Dense(n_hidden, use_bias=True)
      z = nn.elu(Wx(z))

    Wx = nn.Dense(n_input, use_bias=True)

    z = x + Wx(z)  # Encourage identity initialization.

    if single:
      z = jnp.squeeze(z, 0)
    return z

  def create_train_state(
      self,
      rng: jnp.ndarray,
      optimizer: optax.OptState,
      input: Union[int, Tuple[int, ...]],
  ) -> train_state.TrainState:
    """Create initial `TrainState`."""
    params = self.init(rng, jnp.ones(input))["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
