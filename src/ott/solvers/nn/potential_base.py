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
"""Base module for potentials."""

import abc
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state

__all__ = ["PotentialBase", "PotentialTrainState"]

PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]


class PotentialTrainState(train_state.TrainState):
  """Adds information about the potential value and gradient to the state."""
  potential_value_fn: Callable[
      [frozen_dict.FrozenDict[str, jnp.ndarray], Optional[PotentialValueFn_t]],
      PotentialValueFn_t] = struct.field(pytree_node=False)
  potential_gradient_fn: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                                  PotentialGradientFn_t] = struct.field(
                                      pytree_node=False
                                  )


class PotentialBase(abc.ABC, nn.Module):
  """Base class for potentials."""

  @property
  @abc.abstractmethod
  def is_potential(self) -> bool:
    """Indicates if the module defines the potential's value or the gradient.

    Returns:
      ``True`` if the module defines the potential's value, ``False``
      if it defines the gradient.
    """

  def potential_value_fn(
      self,
      params: frozen_dict.FrozenDict[str, jnp.ndarray],
      other_potential_value_fn: Optional[PotentialValueFn_t] = None,
  ) -> PotentialValueFn_t:
    r"""Return a function giving the value of the potential.

    Applies the module if ``self.is_potential`` is ``True``, otherwise
    constructs the value of the potential from the gradient with

    .. math::
      g(y) = -f(\nabla_y g(y)) + y^T \nabla_y g(y)

    where :math:`\nabla_y g(y)` is detached for the envelope theorem
    :cite:`danskin:67,bertsekas:71`
    to give the appropriate first derivatives of this construction.

    Args:
      params: parameters of the module
      x: point to evaluate the value at
      other_potential_value: function giving the value of the other potential.
        Only needed when ``self.is_potential`` is ``False``.

    Returns:
      A function that can be evaluated to obtain the potential's value
    """
    if self.is_potential:
      return lambda x: self.apply({"params": params}, x)
    else:
      assert other_potential_value_fn is not None

      def value_fn(x):
        squeeze = x.ndim == 1
        if squeeze:
          x = jnp.expand_dims(x, 0)
        grad_g_x = jax.lax.stop_gradient(self.apply({"params": params}, x))
        value = -other_potential_value_fn(grad_g_x) + \
            jax.vmap(jnp.dot)(grad_g_x, x)
        return value.squeeze(0) if squeeze else value

      return value_fn

  def potential_gradient_fn(
      self,
      params: frozen_dict.FrozenDict[str, jnp.ndarray],
  ) -> PotentialGradientFn_t:
    """Return a function giving the gradient of the potential.

    Args:
      params: parameters of the module

    Returns:
      A function that can be evaluated to obtain the potential's gradient
    """
    if self.is_potential:
      return jax.vmap(jax.grad(self.potential_value_fn(params)))
    else:
      return lambda x: self.apply({'params': params}, x)
