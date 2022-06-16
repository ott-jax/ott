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
"""Defines classification losses."""

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp

from ott.tools import soft_sort


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray):
  logits = nn.log_softmax(logits)
  return -jnp.sum(labels * logits) / labels.shape[0]


def soft_error_loss(
    logits: jnp.ndarray, labels: jnp.ndarray, epsilon: float = 1e-2
):
  """Average distance between the top rank and the rank of the true class."""
  ranks_fn = functools.partial(soft_sort.ranks, axis=-1, epsilon=epsilon)
  ranks_fn = jax.jit(ranks_fn)
  soft_ranks = ranks_fn(logits)
  return jnp.mean(
      nn.relu(labels.shape[-1] - 1 - jnp.sum(labels * soft_ranks, axis=1))
  )


def get(name: str = 'cross_entropy'):
  """Return the loss function corresponding to the input name."""
  losses = {'soft_error': soft_error_loss, 'cross_entropy': cross_entropy_loss}
  result = losses.get(name, None)
  if result is None:
    raise ValueError(
        f'Unknown loss {name}. Possible values: {",".join(losses)}'
    )
  return result
