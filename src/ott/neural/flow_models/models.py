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

from ott.neural.flow_models import utils

__all__ = ["VelocityField"]


class VelocityField(nn.Module):
  r"""Parameterized neural vector field.

  This class learns a map :math:`v: \mathbb{R}\times \mathbb{R}^d
  \rightarrow \mathbb{R}^d` solving the ODE :math:`\frac{dx}{dt} = v(t, x)`.
  Given a source distribution at time :math:`t_0`, the velocity field can be
  used to transport the source distribution given at :math:`t_0` to
  a target distribution given at :math:`t_1` by integrating :math:`v(t, x)`
  from :math:`t=t_0` to :math:`t=t_1`.

  Each of the input, condition, and time embeddings are passed through a block
  consisting of ``num_layers`` layers of dimension
  ``hidden_dim``, ``condition_dim``, and ``time_embed_dim``,
  respectively. The output of each block is concatenated and passed through
  a final block of dimension ``joint_hidden_dim``.

  Args:
    hidden_dim: Dimensionality of the embedding of the data.
    output_dim: Dimensionality of the neural vector field.
    num_layers: Number of layers.
    condition_dim: Dimensionality of the embedding of the condition.
      If :obj:`None`, TODO.
    time_dim: Dimensionality of the time embedding.
      If :obj:`None`, set to ``hidden_dim``.
    time_encoder: TODO.
    act_fn: Activation function.
  """
  hidden_dim: int
  output_dim: int
  num_layers: int = 3
  condition_dim: Optional[int] = None
  time_dim: Optional[int] = None
  time_encoder: Callable[[jnp.ndarray],
                         jnp.ndarray] = utils.cyclical_time_encoder
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

  @nn.compact
  def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Forward pass through the neural vector field.

    Args:
      t: Time of shape ``[batch, 1]``.
      x: Data of shape ``[batch, ...]``.
      condition: Conditioning vector of shape ``[batch, ...]``.

    Returns:
      Output of the neural vector field of shape ``[batch, output_dim]``.
    """
    time_dim = self.hidden_dim if self.time_dim is None else self.time_dim

    t = self.time_encoder(t)
    for _ in range(self.num_layers):
      t = self.act_fn(nn.Dense(time_dim)(t))
      x = self.act_fn(nn.Dense(self.hidden_dim)(x))
      if self.condition_dim is not None:
        assert condition is not None, "TODO."
        condition = self.act_fn(nn.Dense(self.condition_dim)(condition))

    feats = [t, x] + ([] if condition is None else [condition])
    feats = jnp.concatenate(feats, axis=-1)
    joint_dim = feats.shape[-1]

    for _ in range(self.num_layers):
      feats = self.act_fn(nn.Dense(joint_dim)(feats))

    return nn.Dense(self.output_dim, use_bias=True)(feats)

  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.OptState,
      input_dim: int,
      condition_dim: Optional[int] = None,
  ) -> train_state.TrainState:
    """Create the training state.

    Args:
      rng: Random number generator.
      optimizer: Optimizer.
      input_dim: Dimensionality of the velocity field.
      condition_dim: Dimensionsanilty of the condition
        to the velocity field.

    Returns:
      The training state.
    """
    t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
    cond = None if self.condition_dim is None else jnp.ones((1, condition_dim))

    params = self.init(rng, t, x, cond)["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
