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
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state

__all__ = ["PotentialTrainState", "BasePotential", "PotentialMLP"]

PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]


class PotentialTrainState(train_state.TrainState):
  """Adds information about the model's value and gradient to the state.

  This extends :class:`~flax.training.train_state.TrainState` to include
  the potential methods from the
  :class:`~ott.neural.networks.potentials.BasePotential` used during training.

  Args:
    potential_value_fn: the potential's value function
    potential_gradient_fn: the potential's gradient function
  """
  potential_value_fn: Callable[
      [frozen_dict.FrozenDict[str, jnp.ndarray], Optional[PotentialValueFn_t]],
      PotentialValueFn_t] = struct.field(pytree_node=False)
  potential_gradient_fn: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                                  PotentialGradientFn_t] = struct.field(
                                      pytree_node=False
                                  )


class BasePotential(abc.ABC, nn.Module):
  """Base class for the neural solver models."""

  @property
  @abc.abstractmethod
  def is_potential(self) -> bool:
    """Indicates if the module implements a potential value or a vector field.

    Returns:
      ``True`` if the module defines a potential, ``False`` if it defines a
       vector field.
    """

  def potential_value_fn(
      self,
      params: frozen_dict.FrozenDict[str, jnp.ndarray],
      other_potential_value_fn: Optional[PotentialValueFn_t] = None,
  ) -> PotentialValueFn_t:
    r"""Return a function giving the value of the potential.

    Applies the module if :attr:`is_potential` is ``True``, otherwise
    constructs the value of the potential from the gradient with

    .. math::

      g(y) = -f(\nabla_y g(y)) + y^T \nabla_y g(y)

    where :math:`\nabla_y g(y)` is detached for the envelope theorem
    :cite:`danskin:67,bertsekas:71`
    to give the appropriate first derivatives of this construction.

    Args:
      params: parameters of the module
      other_potential_value_fn: function giving the value of the other
        potential. Only needed when :attr:`is_potential` is ``False``.

    Returns:
      A function that can be evaluated to obtain a potential value, or a linear
      interpolation of a potential.
    """
    if self.is_potential:
      return lambda x: self.apply({"params": params}, x)

    assert other_potential_value_fn is not None, \
      "The value of the gradient-based potential depends " \
      "on the value of the other potential."

    def value_fn(x: jnp.ndarray) -> jnp.ndarray:
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
    """Return a function returning a vector or the gradient of the potential.

    Args:
      params: parameters of the module

    Returns:
      A function that can be evaluated to obtain the potential's gradient
    """
    if self.is_potential:
      return jax.vmap(jax.grad(self.potential_value_fn(params)))
    return lambda x: self.apply({"params": params}, x)

  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.OptState,
      input: Union[int, Tuple[int, ...]],
      **kwargs: Any,
  ) -> PotentialTrainState:
    """Create initial training state."""
    params = self.init(rng, jnp.ones(input))["params"]

    return PotentialTrainState.create(
        apply_fn=self.apply,
        params=params,
        tx=optimizer,
        potential_value_fn=self.potential_value_fn,
        potential_gradient_fn=self.potential_gradient_fn,
        **kwargs,
    )


class PotentialMLP(BasePotential):
  """Potential MLP.

  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The output
      dimension of the last layer is automatically set to 1 if
      :attr:`is_potential` is ``True``, or the dimension of the input otherwise.
    is_potential: Model the potential if ``True``, otherwise
      model the gradient of the potential.
    act_fn: Activation function.
  """

  dim_hidden: Sequence[int]
  is_potential: bool = True
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
    squeeze = x.ndim == 1
    if squeeze:
      x = jnp.expand_dims(x, 0)
    assert x.ndim == 2, x.ndim
    n_input = x.shape[-1]

    z = x
    for n_hidden in self.dim_hidden:
      Wx = nn.Dense(n_hidden, use_bias=True)
      z = self.act_fn(Wx(z))

    if self.is_potential:
      Wx = nn.Dense(1, use_bias=True)
      z = Wx(z).squeeze(-1)

      quad_term = 0.5 * jax.vmap(jnp.dot)(x, x)
      z += quad_term
    else:
      Wx = nn.Dense(n_input, use_bias=True)
      z = x + Wx(z)

    return z.squeeze(0) if squeeze else z
