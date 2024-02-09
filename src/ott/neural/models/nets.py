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
from typing import Callable, Optional

import jax
import jax.numpy as jnp

import flax.linen as nn
import optax
from flax.training import train_state

from ott.neural.models import layers

__all__ = ["RescalingMLP"]


class RescalingMLP(nn.Module):
  """Network to learn distributional rescaling factors based on a MLP.

  The input is passed through a block consisting of ``num_layers_per_block``
  with size ``hidden_dim``.
  If ``condition_dim`` is greater than 0, the conditioning vector is passed
  through a block of the same size.
  Both outputs are concatenated and passed through another block of the same
  size.

  To ensure non-negativity of the output, the output is exponentiated.

  Args:
    hidden_dim: Dimensionality of the hidden layers.
    condition_dim: Dimensionality of the conditioning vector.
    num_layers_per_block: Number of layers per block.
    act_fn: Activation function.

  Returns:
    Non-negative escaling factors.
  """
  hidden_dim: int
  condition_dim: int = 0
  num_layers_per_block: int = 3
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.selu

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None
  ) -> jnp.ndarray:
    """Forward pass through the rescaling network.

    Args:
      x: Data of shape ``[n, ...]``.
      condition: Condition of shape ``[n, condition_dim]``.

    Returns:
      Estimated rescaling factors.
    """
    x_layer = layers.MLPBlock(
        dim=self.hidden_dim,
        out_dim=self.hidden_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn
    )
    x = x_layer(x)

    if self.condition_dim > 0:
      condition_layer = layers.MLPBlock(
          dim=self.hidden_dim,
          out_dim=self.hidden_dim,
          num_layers=self.num_layers_per_block,
          act_fn=self.act_fn
      )

      condition = condition_layer(condition)
      concatenated = jnp.concatenate((x, condition), axis=-1)
    else:
      concatenated = x

    out_layer = layers.MLPBlock(
        dim=self.hidden_dim,
        out_dim=self.hidden_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn
    )

    out = out_layer(concatenated)
    return jnp.exp(out)

  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.OptState,
      input_dim: int,
  ) -> train_state.TrainState:
    """Create the training state.

    Args:
      rng: Random number generator.
      optimizer: Optimizer.
      input_dim: Dimensionality of the input.

    Returns:
      Training state.
    """
    params = self.init(
        rng, jnp.ones((1, input_dim)), jnp.ones((1, self.condition_dim))
    )["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
