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
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

__all__ = ["PositiveDense", "PosDefPotentials"]

PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

class PositiveDense(nn.Module):
    """A linear transformation using a weight matrix with all entries positive.

    Args:
      dim_hidden: the number of output dim_hidden.
      rectifier_fn: choice of rectifier function (default: softplus function).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    dim_hidden: int
    rectifier_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to inputs along the last dimension.

        Args:
          inputs: Array to be transformed.

        Returns:
          The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", self.kernel_init, (inputs.shape[-1], self.dim_hidden))
        kernel = self.rectifier_fn(kernel)
        kernel = jnp.asarray(kernel, self.dtype)
        y = jax.lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.dim_hidden,))
            bias = jnp.asarray(bias, self.dtype)
            return y + bias
        return y


class PosDefPotentials(nn.Module):
    """A layer to output  0.5 x^T(A_i A_i^T + Diag(d_i^2))x + b_i^T x + c_i potentials.

    Args:
      num_potentials: the dimension of the output
      rank: the rank of the matrix used for the quadratic layer
      use_linear: whether to add a linear layer to the output
      use_bias: whether to add a bias to the output.
      dtype: the dtype of the computation.
      precision: numerical precision of computation see `jax.lax.Precision` for details.
      kernel_quadratic_init: initializer function for the weight matrix of the quadratic layer.
      kernel_linear_init: initializer function for the weight matrix of the linea layer.
      bias_init: initializer function for the bias.
    """

    num_potentials: int
    rank: int = 0
    use_linear: bool = True
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_quadratic_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    kernel_diagonal_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones
    kernel_linear_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply a few quadratic forms.

        Args:
          inputs: Array to be transformed (possibly batched).

        Returns:
          The transformed input.
        """
        dim_data = inputs.shape[-1]
        inputs = jnp.asarray(inputs, self.dtype)
        inputs = inputs.reshape((-1, dim_data))

        diag_kernel = self.param("diag_kernel", self.kernel_diagonal_init, (1, dim_data, self.num_potentials))

        # ensures the diag_kernel parameter stays non negative
        diag_kernel = nn.activation.relu(diag_kernel)
        y = 0.5 * jnp.sum(jnp.multiply(inputs[..., None], diag_kernel) ** 2, axis=1)

        if self.rank > 0:
            quadratic_kernel = self.param(
                "quad_kernel", self.kernel_quadratic_init, (self.num_potentials, dim_data, self.rank)
            )
            y += jnp.sum(
                0.5
                * jnp.tensordot(inputs, quadratic_kernel, axes=((inputs.ndim - 1,), (1,)), precision=self.precision)
                ** 2,
                axis=2,
            )

        if self.use_linear:
            linear_kernel = self.param("lin_kernel", self.kernel_linear_init, (dim_data, self.num_potentials))
            y = y + jnp.dot(inputs, linear_kernel, precision=self.precision)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (1, self.num_potentials))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias

        return y
