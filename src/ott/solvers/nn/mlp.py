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

from typing import Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

__all__ = ["MLP"]


class MLP(nn.Module):
  dim_hidden: Sequence[int]
  returns_potential: bool = True
  act_fn: Callable = nn.leaky_relu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    squeeze = x.ndim == 1
    if squeeze:
      x = jnp.expand_dims(x, 0)
    assert x.ndim == 2, x.ndim
    n_input = x.shape[-1]

    z = x
    for n_hidden in self.dim_hidden:
      Wx = nn.Dense(n_hidden, use_bias=True)
      z = self.act_fn(Wx(z))

    if self.returns_potential:
      Wx = nn.Dense(1, use_bias=True)
      z = Wx(z).squeeze(-1)

      quad_term = 0.5 * jax.vmap(jnp.dot)(x, x)
      z += quad_term
    else:
      Wx = nn.Dense(n_input, use_bias=True)
      z = x + Wx(z)

    return z.squeeze(0) if squeeze else z

  def create_train_state(
      self,
      rng: jnp.ndarray,
      optimizer: optax.OptState,
      input: Union[int, Tuple[int, ...]],
      params: Optional[FrozenDict] = None,
  ) -> train_state.TrainState:
    """Create initial `TrainState`."""
    if params is None:
      params = self.init(rng, jnp.ones(input))["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
