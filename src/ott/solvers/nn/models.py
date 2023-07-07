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

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state
from jax.nn import initializers

from ott.math import matrix_square_root
from ott.solvers.nn import layers

__all__ = ["NeuralTrainState", "ModelBase", "ICNN", "MLP"]

PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]


class NeuralTrainState(train_state.TrainState):
  """Adds information about the model's value and gradient to the state.

  This extends :class:`~flax.training.train_state.TrainState` to include
  the potential methods from :class:`~ott.solvers.nn.models.ModelBase`
  used during training.

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


class ModelBase(abc.ABC, nn.Module):
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
      rng: jax.random.PRNGKeyArray,
      optimizer: optax.OptState,
      input: Union[int, Tuple[int, ...]],
      **kwargs: Any,
  ) -> NeuralTrainState:
    """Create initial training state."""
    params = self.init(rng, jnp.ones(input))["params"]

    return NeuralTrainState.create(
        apply_fn=self.apply,
        params=params,
        tx=optimizer,
        potential_value_fn=self.potential_value_fn,
        potential_gradient_fn=self.potential_gradient_fn,
        **kwargs,
    )


class ICNN(ModelBase):
  """Input convex neural network (ICNN) architecture with initialization.

  Implementation of input convex neural networks as introduced in
  :cite:`amos:17` with initialization schemes proposed by :cite:`bunne:22`.

  Args:
    dim_data: data dimensionality.
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is 1 by default.
    init_std: value of standard deviation of weight initialization method.
    init_fn: choice of initialization method for weight matrices (default:
      :func:`jax.nn.initializers.normal`).
    act_fn: choice of activation function used in network architecture
      (needs to be convex, default: :obj:`jax.nn.relu`).
    pos_weights: Enforce positive weights with a projection.
      If ``False``, the positive weights should be enforced with clipping
      or regularization in the loss.
    gaussian_map_samples: data inputs of source and target measures for
      initialization scheme based on Gaussian approximation of input and
      target measure (if ``None``, identity initialization is used).
  """
  dim_data: int
  dim_hidden: Sequence[int]
  init_std: float = 1e-2
  init_fn: Callable = jax.nn.initializers.normal
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  pos_weights: bool = True
  gaussian_map_samples: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None

  @property
  def is_potential(self) -> bool:  # noqa: D102
    return True

  def setup(self) -> None:  # noqa: D102
    self.num_hidden = len(self.dim_hidden)

    if self.pos_weights:
      hid_dense = layers.PositiveDense
      # this function needs to be the inverse map of function
      # used in PositiveDense layers
      rescale = hid_dense.inv_rectifier_fn
    else:
      hid_dense = nn.Dense
      rescale = lambda x: x
    self.use_init = False
    # check if Gaussian map was provided
    if self.gaussian_map_samples is not None:
      factor, mean = self._compute_gaussian_map_params(
          self.gaussian_map_samples
      )
    else:
      factor, mean = self._compute_identity_map_params(self.dim_data)

    w_zs = []
    # keep track of previous size to normalize accordingly
    normalization = 1

    for i in range(1, self.num_hidden):
      w_zs.append(
          hid_dense(
              self.dim_hidden[i],
              kernel_init=initializers.constant(rescale(1.0 / normalization)),
              use_bias=False,
          )
      )
      normalization = self.dim_hidden[i]
    # final layer computes average, still with normalized rescaling
    w_zs.append(
        hid_dense(
            1,
            kernel_init=initializers.constant(rescale(1.0 / normalization)),
            use_bias=False,
        )
    )
    self.w_zs = w_zs

    # positive definite potential (the identity mapping or linear OT)
    self.pos_def_potential = layers.PosDefPotentials(
        self.dim_data,
        num_potentials=1,
        kernel_init=lambda *_: factor,
        bias_init=lambda *_: mean,
        use_bias=True,
    )

    # subsequent layers re-injected into convex functions
    w_xs = []
    for i in range(self.num_hidden):
      w_xs.append(
          nn.Dense(
              self.dim_hidden[i],
              kernel_init=self.init_fn(self.init_std),
              bias_init=initializers.constant(0.),
              use_bias=True,
          )
      )
    # final layer, to output number
    w_xs.append(
        nn.Dense(
            1,
            kernel_init=self.init_fn(self.init_std),
            bias_init=initializers.constant(0.),
            use_bias=True,
        )
    )
    self.w_xs = w_xs

  @staticmethod
  def _compute_gaussian_map_params(
      samples: Tuple[jnp.ndarray, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    from ott.tools.gaussian_mixture import gaussian
    source, target = samples
    # print(source)
    # print(type(source))
    g_s = gaussian.Gaussian.from_samples(source)
    g_t = gaussian.Gaussian.from_samples(target)
    lin_operator = g_s.scale.gaussian_map(g_t.scale)
    b = jnp.squeeze(g_t.loc
                   ) - jnp.linalg.solve(lin_operator, jnp.squeeze(g_t.loc))
    lin_operator = matrix_square_root.sqrtm_only(lin_operator)
    return jnp.expand_dims(lin_operator, 0), jnp.expand_dims(b, 0)

  @staticmethod
  def _compute_identity_map_params(
      input_dim: int
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A = jnp.eye(input_dim).reshape((1, input_dim, input_dim))
    b = jnp.zeros((1, input_dim))
    return A, b

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    z = self.act_fn(self.w_xs[0](x))
    for i in range(self.num_hidden):
      z = jnp.add(self.w_zs[i](z), self.w_xs[i + 1](x))
      z = self.act_fn(z)
    z += self.pos_def_potential(x)
    return z.squeeze()


class MLP(ModelBase):
  """A generic, typically not-convex (w.r.t input) MLP.

  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The output
      dimension of the last layer is automatically set to 1 if
      :attr:`is_potential` is ``True``, or the dimension of the input otherwise
    is_potential: Model the potential if ``True``, otherwise
      model the gradient of the potential
    act_fn: Activation function
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
