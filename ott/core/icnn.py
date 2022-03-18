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

# Lint as: python3
"""Implementation of Amos+(2017) input convex neural networks (ICNN)."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Sequence, Tuple

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


class PositiveDense(nn.Module):
  """A linear transformation using a weight matrix with all entries positive.

  Args:
    dim_hidden: the number of output dim_hidden.
    beta: inverse temperature parameter of the softplus function (default: 1).
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  dim_hidden: int
  beta: float = 1.0
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[
    [PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    """Applies a linear transformation to inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param(
      'kernel', self.kernel_init, (inputs.shape[-1], self.dim_hidden))
    scaled_kernel = self.beta * kernel
    kernel = jnp.asarray(
      1 / self.beta * nn.softplus(scaled_kernel), self.dtype)
    y = jax.lax.dot_general(
      inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.dim_hidden,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


class ICNN(nn.Module):
  """Input convex neural network (ICNN) architeture.

  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is 1 by default.
    init_std: value of standard deviation of weight initialization method.
    init_fn: choice of initialization method for weight matrices (default:
      `jax.nn.initializers.normal`).
    act_fn: choice of activation function used in network architecture
      (needs to be convex, default: `nn.leaky_relu`).
  """

  dim_hidden: Sequence[int]
  init_std: float = 0.1
  init_fn: Callable = jax.nn.initializers.normal
  act_fn: Callable = nn.leaky_relu
  pos_weights: bool = True

  def setup(self):
    num_hidden = len(self.dim_hidden)

    w_zs = list()

    if self.pos_weights:
      Dense = PositiveDense
    else:
      Dense = nn.Dense

    for i in range(1, num_hidden):
      w_zs.append(Dense(
        self.dim_hidden[i], kernel_init=self.init_fn(self.init_std),
        use_bias=False))
    w_zs.append(Dense(
      1, kernel_init=self.init_fn(self.init_std), use_bias=False))
    self.w_zs = w_zs

    w_xs = list()
    for i in range(num_hidden):
      w_xs.append(nn.Dense(
        self.dim_hidden[i], kernel_init=self.init_fn(self.init_std),
        use_bias=True))
    w_xs.append(nn.Dense(
      1, kernel_init=self.init_fn(self.init_std), use_bias=True))
    self.w_xs = w_xs

  @nn.compact
  def __call__(self, x):
    """Applies ICNN module.

    Args:
      x: jnp.ndarray<float>[batch_size, n_features]: input to the ICNN.

    Returns:
      jnp.ndarray<float>[1]: output of ICNN.
    """
    z = self.act_fn(self.w_xs[0](x))
    z = jnp.multiply(z, z)

    for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
      z = self.act_fn(jnp.add(Wz(z), Wx(x)))
    y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))

    return jnp.squeeze(y)
