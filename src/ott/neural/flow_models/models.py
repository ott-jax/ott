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

  Args:
    hidden_dims: Dimensionality of the embedding of the data.
    condition_dims: Dimensionality of the embedding of the condition.
      If :obj:`None`, the velocity field has no conditions.
    time_dims: Dimensionality of the time embedding.
      If :obj:`None`, ``hidden_dims`` will be used.
    time_encoder: Function to encode the time input to the time-dependent
      velocity field.
    act_fn: Activation function.
  """
  output_dim: int
  hidden_dims: Sequence[int] = (128, 128, 128)
  condition_dims: Optional[Sequence[int]] = None
  time_dims: Optional[Sequence[int]] = None
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
    if self.condition_dims is None:
      cond_dims = [None] * len(self.hidden_dims)
    else:
      cond_dims = self.condition_dims
    time_dims = self.hidden_dims if self.time_dims is None else self.time_dims

    assert len(self.hidden_dims) == len(cond_dims), "TODO"
    assert len(self.hidden_dims) == len(time_dims), "TODO"

    t = self.time_encoder(t)
    for time_dim, cond_dim, hidden_dim in zip(
        time_dims, cond_dims, self.hidden_dims
    ):
      t = self.act_fn(nn.Dense(time_dim)(t))
      x = self.act_fn(nn.Dense(hidden_dim)(x))
      if self.condition_dims is not None:
        assert condition is not None, "No condition was specified."
        condition = self.act_fn(nn.Dense(cond_dim)(condition))

    feats = [t, x] + ([] if self.condition_dims is None else [condition])
    feats = jnp.concatenate(feats, axis=-1)
    joint_dim = feats.shape[-1]

    for _ in range(len(self.hidden_dims)):
      feats = self.act_fn(nn.Dense(joint_dim)(feats))

    return nn.Dense(self.output_dim)(feats)

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
    if self.condition_dims is not None:
      assert condition_dim is not None, "TODO"
      cond = jnp.ones((1, condition_dim))
    else:
      cond = None

    params = self.init(rng, t, x, cond)["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
