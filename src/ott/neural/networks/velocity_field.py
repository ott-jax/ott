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
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax.training import train_state

from ott.neural.networks.layers import time_encoder

__all__ = ["VelocityField"]


class VelocityField(nn.Module):
  r"""Neural vector field.

  This class learns a map :math:`v: \mathbb{R}\times \mathbb{R}^d
  \rightarrow \mathbb{R}^d` solving the ODE :math:`\frac{dx}{dt} = v(t, x)`.
  Given a source distribution at time :math:`t_0`, the velocity field can be
  used to transport the source distribution given at :math:`t_0` to
  a target distribution given at :math:`t_1` by integrating :math:`v(t, x)`
  from :math:`t=t_0` to :math:`t=t_1`.

  Args:
    hidden_dims: Dimensionality of the embedding of the data.
    output_dims: Dimensionality of the embedding of the output.
    condition_dims: Dimensionality of the embedding of the condition.
      If :obj:`None`, the velocity field has no conditions.
    time_dims: Dimensionality of the time embedding.
      If :obj:`None`, ``hidden_dims`` is used.
    time_encoder: Time encoder for the velocity field.
    act_fn: Activation function.
  """
  hidden_dims: Sequence[int]
  output_dims: Sequence[int]
  condition_dims: Optional[Sequence[int]] = None
  time_dims: Optional[Sequence[int]] = None
  time_encoder: Callable[[jnp.ndarray],
                         jnp.ndarray] = time_encoder.cyclical_time_encoder
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      train: bool = True,
  ) -> jnp.ndarray:
    """Forward pass through the neural vector field.

    Args:
      t: Time of shape ``[batch, 1]``.
      x: Data of shape ``[batch, ...]``.
      condition: Conditioning vector of shape ``[batch, ...]``.
      train: If `True`, enables dropout for training.

    Returns:
      Output of the neural vector field of shape ``[batch, output_dim]``.
    """
    time_dims = self.hidden_dims if self.time_dims is None else self.time_dims

    t = self.time_encoder(t)
    for time_dim in time_dims:
      t = self.act_fn(nn.Dense(time_dim)(t))
      t = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(t)

    for hidden_dim in self.hidden_dims:
      x = self.act_fn(nn.Dense(hidden_dim)(x))
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

    if self.condition_dims is not None:
      assert condition is not None, "No condition was passed."
      for cond_dim in self.condition_dims:
        condition = self.act_fn(nn.Dense(cond_dim)(condition))
        condition = nn.Dropout(
            rate=self.dropout_rate, deterministic=not train
        )(
            condition
        )
      feats = jnp.concatenate([t, x, condition], axis=-1)
    else:
      feats = jnp.concatenate([t, x], axis=-1)

    for output_dim in self.output_dims[:-1]:
      feats = self.act_fn(nn.Dense(output_dim)(feats))
      feats = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(feats)

    # No activation function for the final layer
    return nn.Dense(self.output_dims[-1])(feats)

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
      condition_dim: Dimensionality of the condition of the velocity field.

    Returns:
      The training state.
    """
    t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
    if self.condition_dims is None:
      cond = None
    else:
      assert condition_dim > 0, "Condition dimension must be positive."
      cond = jnp.ones((1, condition_dim))

    params = self.init(rng, t, x, cond, train=False)["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
