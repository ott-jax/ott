# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax CNN model."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class ConvBlock(nn.Module):
  """A simple CNN blockl."""

  features: int = 32
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    return x


class CNN(nn.Module):
  """A simple CNN model."""

  num_classes: int = 10
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = ConvBlock(features=32)(x)
    x = ConvBlock(features=64)(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=512)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)
    return x
