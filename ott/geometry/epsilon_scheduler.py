# coding=utf-8
# Copyright 2021 Google LLC.
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

# Lint as: python3
"""A class to define a scheduler for the entropic regularization epsilon."""
from typing import Optional

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class Epsilon:
  """Scheduler class for the regularization parameter epsilon."""

  def __init__(self,
               target: float = 1e-2,
               init: float = 1.0,
               decay: float = 1.0):
    self.target = target
    self._init = init
    self._decay = decay

  def at(self, iteration: Optional[int] = 1) -> float:
    if iteration is None:
      return self.target
    init = jnp.where(self._decay < 1.0, self._init, self.target)
    decay = jnp.where(self._decay < 1.0, self._decay, 1.0)
    return jnp.maximum(init * decay**iteration, self.target)

  def done(self, eps):
    return eps == self.target

  def done_at(self, iteration):
    return self.done(self.at(iteration))

  def tree_flatten(self):
    return (self.target, self._init, self._decay), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)

  @classmethod
  def make(cls, *args, **kwargs):
    """Create or return an Epsilon instance."""
    if isinstance(args[0], cls):
      return args[0]
    else:
      return cls(*args, **kwargs)

