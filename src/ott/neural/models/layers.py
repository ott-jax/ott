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
from typing import Any

import jax.numpy as jnp

import flax.linen as nn

__all__ = ["MLPBlock"]


class MLPBlock(nn.Module):
  """An MLP block.

  Args:
    dim: Dimensionality of the input data.
    num_layers: Number of layers in the MLP block.
    act_fn: Activation function.
    out_dim: Dimensionality of the output data.
  """
  dim: int = 128
  num_layers: int = 3
  act_fn: Any = nn.silu
  out_dim: int = 128

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply the MLP block.

    Args:
      x: Input data of shape (batch_size, dim).

    Returns:
      Output data of shape (batch_size, out_dim).
    """
    for _ in range(self.num_layers):
      x = nn.Dense(self.dim)(x)
      x = self.act_fn(x)
    return nn.Dense(self.out_dim)(x)
