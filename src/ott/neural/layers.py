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
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

__all__ = ["PositiveDense", "PosDefPotentials"]

PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = jnp.ndarray

DEFAULT_KERNEL_INIT = nn.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nn.initializers.zeros
DEFAULT_RECTIFIER = nn.activation.relu


class PositiveDense(nn.Module):
  """A linear transformation using a weight matrix with all entries positive.

  Args:
    dim_hidden: Number of output dimensions.
    rectifier_fn: The rectifier function.
    kernel_init: Initializer for the weight matrix.
    bias_init: Initializer for the bias.
    precision: Numerical precision of computation,
      see :class:`~jax.lax.Precision` for details.
  """

  dim_hidden: int
  rectifier_fn: Callable[[Array], Array] = DEFAULT_RECTIFIER
  use_bias: bool = True
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  precision: Optional[jax.lax.Precision] = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies a linear transformation to inputs along the last dimension.

    Args:
      x: Array of shape ``[batch, ..., features]``.

    Returns:
      The output, array of shape ``[batch, ..., dim_hidden]``.
    """
    kernel = self.param(
        "kernel", self.kernel_init, (x.shape[-1], self.dim_hidden)
    )
    kernel = self.rectifier_fn(kernel)

    x = jnp.tensordot(x, kernel, axes=(-1, 0), precision=self.precision)
    if self.use_bias:
      x = x + self.param("bias", self.bias_init, (self.dim_hidden,))

    return x


# TODO(michalk8): update the docstring
class PosDefPotentials(nn.Module):
  """A layer to output  0.5 x^T(A_i A_i^T + Diag(d_i^2))x + b_i^T x + c_i potentials.

  Args:
    num_potentials: the dimension of the output
    rank: The rank of the matrix used for the quadratic layer.
    use_linear: Whether to add a linear layer to the output.
    use_bias: Whether to add a bias to the output.
    kernel_quad_init: Initializer for the weight matrix of the quadratic layer.
    kernel_diag_init: Initializer for the weight matrix of the diagonal layer.
    kernel_linear_init: Initializer for the weight matrix of the linear layer.
    bias_init: Initializer for the bias.
    precision: Numerical precision of computation,
      see :class:`~jax.lax.Precision` for details.
  """

  num_potentials: int
  rank: int = 0
  use_linear: bool = True
  use_bias: bool = True
  kernel_quad_init: Callable[[PRNGKey, Shape, Dtype],
                             Array] = DEFAULT_KERNEL_INIT
  kernel_diag_init: Callable[[PRNGKey, Shape, Dtype],
                             Array] = nn.initializers.ones
  kernel_linear_init: Callable[[PRNGKey, Shape, Dtype],
                               Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  precision: Optional[jax.lax.Precision] = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Apply a few quadratic forms.

    Args:
      inputs: Array to be transformed (possibly batched).

    Returns:
      The transformed input.
    """
    dim_data = inputs.shape[-1]
    inputs = inputs.reshape((-1, dim_data))

    diag_kernel = self.param(
        "diag_kernel", self.kernel_diag_init,
        (1, dim_data, self.num_potentials)
    )
    # ensures the diag_kernel parameter stays non negative
    diag_kernel = DEFAULT_RECTIFIER(diag_kernel)
    y = 0.5 * jnp.sum(jnp.multiply(inputs[..., None], diag_kernel) ** 2, axis=1)

    if self.rank > 0:
      quad_kernel = self.param(
          "quad_kernel", self.kernel_quad_init,
          (self.num_potentials, dim_data, self.rank)
      )
      # TODO(michalk8): nicer formatting
      y += jnp.sum(
          0.5 * jnp.tensordot(
              inputs,
              quad_kernel,
              axes=((inputs.ndim - 1,), (1,)),
              precision=self.precision
          ) ** 2,
          axis=2,
      )

    if self.use_linear:
      linear_kernel = self.param(
          "lin_kernel", self.kernel_linear_init,
          (dim_data, self.num_potentials)
      )
      y = y + jnp.dot(inputs, linear_kernel, precision=self.precision)

    if self.use_bias:
      y = y + self.param("bias", self.bias_init, (1, self.num_potentials))

    return y

  @classmethod
  def init_from_samples(
      cls, source: jnp.ndarray, target: jnp.ndarray, **kwargs: Any
  ) -> "PosDefPotentials":
    """Initialize the layer using Gaussian approximation :cite:`bunne:22`.

    Args:
      source: Samples from the source distribution, array of shape ``[n, d]``.
      target: Samples from the target distribution, array of shape ``[m, d]``.
      kwargs: Keyword arguments for initialization. Note that ``use_linear``
        will be always set to :obj:`True`.

    Returns:
      The positive-definite potentials.
    """
    factor, mean = _compute_gaussian_map_params(source, target)

    kwargs["use_linear"] = True
    return cls(
        kernel_quad_init=lambda *_, **__: factor,
        kernel_linear_init=lambda *_, **__: mean.T,
        **kwargs,
    )


def _compute_gaussian_map_params(
    source: jnp.ndarray, target: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  from ott.math import matrix_square_root
  from ott.tools.gaussian_mixture import gaussian

  g_s = gaussian.Gaussian.from_samples(source)
  g_t = gaussian.Gaussian.from_samples(target)
  lin_op = g_s.scale.gaussian_map(g_t.scale)
  b = jnp.squeeze(g_t.loc) - lin_op @ jnp.squeeze(g_s.loc)
  lin_op = matrix_square_root.sqrtm_only(lin_op)

  return jnp.expand_dims(lin_op, 0), jnp.expand_dims(b, 0)
