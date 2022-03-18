# coding=utf-8
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

"""Pytree for a vector of probabilities."""

from typing import Optional

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class Probabilities:
  """Parameterized array of probabilities of length n.

  The internal representation is a length n-1 unconstrained array. We convert
  to a length n simplex by appending a 0 and taking a softmax.
  """
  _params: jnp.ndarray

  def __init__(self, params):
    self._params = params

  @classmethod
  def from_random(
      cls,
      key: jnp.ndarray,
      n_dimensions: int,
      stdev: Optional[float] = 0.1,
      dtype: Optional[jnp.dtype] = None) -> 'Probabilities':
    """Construct a random Probabilities."""
    return cls(params=jax.random.normal(
        key=key, shape=(n_dimensions - 1,), dtype=dtype) * stdev)

  @classmethod
  def from_probs(
      cls,
      probs: jnp.ndarray) -> 'Probabilities':
    """Construct Probabilities from a vector of probabilities."""
    log_probs = jnp.log(probs)
    log_probs_normalized, norm = log_probs[:-1], log_probs[-1]
    log_probs_normalized -= norm
    return cls(params=log_probs_normalized)

  @property
  def params(self):
    return self._params

  @property
  def dtype(self):
    return self._params.dtype

  def unnormalized_log_probs(self) -> jnp.ndarray:
    return jnp.concatenate(
        [self._params, jnp.zeros((1,), dtype=self.dtype)], axis=-1)

  def log_probs(self) -> jnp.ndarray:
    return jax.nn.log_softmax(self.unnormalized_log_probs())

  def probs(self) -> jnp.ndarray:
    return jax.nn.softmax(self.unnormalized_log_probs())

  def sample(self, key: jnp.ndarray, size: int) -> jnp.ndarray:
    return jax.random.categorical(
        key=key, logits=self.unnormalized_log_probs(), shape=(size,))

  def tree_flatten(self):
    children = (self.params,)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  def __repr__(self):
    class_name = type(self).__name__
    children, aux = self.tree_flatten()
    return '{}({})'.format(
        class_name, ', '.join([repr(c) for c in children] +
                              [f'{k}: {repr(v)}' for k, v in aux.items()]))

  def __hash__(self):
    return jax.tree_util.tree_flatten(self).__hash__()

  def __eq__(self, other):
    return jax.tree_util.tree_flatten(self) == jax.tree_util.tree_flatten(other)
