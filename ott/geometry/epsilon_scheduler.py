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
               target: Optional[float] = None,
               scale: Optional[float] = None,
               init: Optional[float] = None,
               decay: Optional[float] = None):
    """Initializes a scheduler using possibly geometric decay.

    The entropic regularization value is given either directly or relative to
    a scale. In that case, the initial ``target`` value is understood to be a
    proportion of the ``scale``. Both are recorded and merged in the ``target``
    field which is built from those two parameters.

    Args:
      target: the epsilon regularizing value that is targeted, understood
        as a multiple of scale.
      scale: scale to be used with target_init to define a target epsilon.
      init: initial value when using epsilon scheduling, understood as a
        a fraction of scale as well.
      decay: geometric decay factor, smaller than 1.
    """
    self._target_init = .01 if target is None else target
    self._scale = 1.0 if scale is None else scale
    self._init = 1.0 if init is None else init
    self._decay = 1.0 if decay is None else decay

  @property
  def target(self):
    return self._target_init * self._scale

  def at(self, iteration: Optional[int] = 1) -> float:
    if iteration is None:
      return self.target
    init = jnp.where(self._decay < 1.0, self._init, self._target_init)
    decay = jnp.where(self._decay < 1.0, self._decay, 1.0)
    return jnp.maximum(init * decay**iteration, self._target_init) * self._scale

  def done(self, eps):
    return eps == self.target

  def done_at(self, iteration):
    return self.done(self.at(iteration))

  def tree_flatten(self):
    return (self._target_init, self._scale, self._init, self._decay), None

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

