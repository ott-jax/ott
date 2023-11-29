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
from typing import Optional

import jax.numpy as jnp

import flax.linen as nn

__all__ = ["BaseNeuralVectorField", "BaseRescalingNet"]


class BaseNeuralVectorField(nn.Module, abc.ABC):
  """Base class for neural vector field models."""

  @abc.abstractmethod
  def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      keys_model: Optional[jnp.ndarray] = None
  ) -> jnp.ndarray:
    """"Evaluate the vector field.

    Args:
      t: Time.
      x: Input data.
      condition: Condition.
      keys_model: Random keys for the model.
    """
    pass


class BaseRescalingNet(nn.Module, abc.ABC):
  """Base class for models to learn distributional rescaling factors."""

  @abc.abstractmethod
  def __call__(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None
  ) -> jnp.ndarray:
    """Evaluate the model.

    Args:
      x: Input data.
      condition: Condition.
    """
    pass
