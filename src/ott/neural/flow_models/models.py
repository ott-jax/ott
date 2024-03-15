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

import ott.neural.flow_models.layers as flow_layers

__all__ = ["VelocityField"]


class VelocityField(nn.Module):
  r"""Parameterized neural vector field.

  The `VelocityField` learns a map :math:`v: \mathbb{R}\times \mathbb{R}^d
  \rightarrow \mathbb{R}^d` solving the ODE :math:`\frac{dx}{dt} = v(t, x)`.
  Given a source distribution at time :math:`t_0`, the velocity field can be
  used to transport the source distribution given at :math:`t_0` to
  a target distribution given at :math:`t_1` by integrating :math:`v(t, x)`
  from :math:`t=t_0` to :math:`t=t_1`.

  Each of the input, condition, and time embeddings are passed through a block
  consisting of ``num_layers`` layers of dimension
  ``latent_embed_dim``, ``condition_embed_dim``, and ``time_embed_dim``,
  respectively. The output of each block is concatenated and passed through
  a final block of dimension ``joint_hidden_dim``.

  Args:
    output_dim: Dimensionality of the neural vector field.
    latent_embed_dim: Dimensionality of the embedding of the data.
    condition_dim: Dimensionality of the conditioning vector.
    condition_embed_dim: Dimensionality of the embedding of the condition.
      If :obj:`None`, set to ``latent_embed_dim``.
    t_embed_dim: Dimensionality of the time embedding.
      If :obj:`None`, set to ``latent_embed_dim``.
    num_layers: Number of layers.
    act_fn: Activation function.
    n_freqs: Number of frequencies to use for the time embedding.
  """
  output_dim: int
  latent_embed_dim: int
  condition_dim: int = 0
  condition_embed_dim: Optional[int] = None
  t_embed_dim: Optional[int] = None
  num_layers: int = 3
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
  n_freqs: int = 128

  def __post_init__(self) -> None:
    if self.condition_embed_dim is None:
      self.condition_embed_dim = self.latent_embed_dim
    if self.t_embed_dim is None:
      self.t_embed_dim = self.latent_embed_dim
    self.joint_hidden_dim = (
        self.latent_embed_dim + self.condition_embed_dim + self.t_embed_dim
    )
    super().__post_init__()

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
      condition: Conditioning vector of shape ``[batch, cond_dim]``.

    Returns:
      Output of the neural vector field of shape ``[batch, output_dim]``.
    """
    t = flow_layers.CyclicalTimeEncoder(self.n_freqs)(t)

    for _ in range(self.num_layers):
      t = self.act_fn(nn.Dense(self.t_embed_dim)(t))
      x = self.act_fn(nn.Dense(self.latent_embed_dim)(x))
      if self.condition_dim > 0:
        assert condition is not None, "TODO."
        condition = self.act_fn(nn.Dense(self.condition_embed_dim)(condition))

    arrs = [t, x] + ([] if condition is None else [condition])
    out = jnp.concatenate(arrs, axis=-1)

    for _ in range(self.num_layers):
      out = self.act_fn(nn.Dense(self.joint_hidden_dim)(out))
    return nn.Dense(self.output_dim, use_bias=True)(out)

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
      The training state.
    """
    params = self.init(
        rng, jnp.ones((1, 1)), jnp.ones((1, input_dim)),
        jnp.ones((1, self.condition_dim))
    )["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
