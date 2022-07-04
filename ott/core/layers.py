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
"""Layers used in input convex neural networks (Amos+(2017), Bunne+(2022))."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class PositiveDense(nn.Module):
  """A linear transformation using a weight matrix with all entries positive.

  Args:
    dim_hidden: the number of output dim_hidden.
    rectifier_fn: choice of rectiver function (default: softplus function).
    inv_rectifier_fn: choice of inverse rectiver function
      (default: inverse softplus function).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  dim_hidden: int
  rectifier_fn: Callable = nn.softplus
  inv_rectifier_fn: Callable = lambda x: jnp.log(jnp.exp(x) - 1)
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

  def setup(self):
    try:
      if round(self.inv_rectifier_fn(self.rectifier_fn(0.1)), 3) != 0.1:
        raise RuntimeError(
            "Make sure both rectifier and inverse are defined properly."
        )
    except TypeError as e:
      if "doesn't define __round__ method" not in str(e):
        raise  # not comparing tracer values, raise

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
        'kernel', self.kernel_init, (inputs.shape[-1], self.dim_hidden)
    )
    kernel = self.rectifier_fn(kernel)
    y = jax.lax.dot_general(
        inputs,
        kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision
    )
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.dim_hidden,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


class PosDefPotentials(nn.Module):
  """A layer to output  (0.5 [A_i A_i^T] (x - b_i)_i potentials.

  Args:
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  dim_data: int
  num_potentials: int
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    """Applies a few quadratic forms.

    Args:
      inputs: The nd-array to be transformed (possibly batched)
    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param(
        "kernel", self.kernel_init,
        (self.num_potentials, inputs.shape[-1], inputs.shape[-1])
    )

    if self.use_bias:
      bias = self.param(
          "bias", self.bias_init, (self.num_potentials, self.dim_data)
      )
      bias = jnp.asarray(bias, self.dtype)

      y = inputs.reshape((-1, inputs.shape[-1])) if inputs.ndim == 1 else inputs
      y = y[..., None] - bias.T[None, ...]
      y = jax.lax.dot_general(
          y, kernel, (((1,), (1,)), ((2,), (0,))), precision=self.precision
      )
    else:
      y = jax.lax.dot_general(
          inputs,
          kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
          precision=self.precision
      )

    y = 0.5 * y * y
    out = jnp.sum(y.reshape((-1, self.num_potentials, self.dim_data)), axis=2)
    return out
