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
import abc

import jax.numpy as jnp

import flax.linen as nn

__all__ = ["TimeEncoder", "CyclicalTimeEncoder"]


class TimeEncoder(nn.Module, abc.ABC):
  """A time encoder."""

  @abc.abstractmethod
  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    """Encode the time.

    Args:
      t: Input time of shape (batch_size, 1).

    Returns:
      The encoded time.
    """
    pass


class CyclicalTimeEncoder(nn.Module):
  """A cyclical time encoder."""
  n_frequencies: int = 128

  @nn.compact
  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
    freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
    t = freq * t
    return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)
