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

DEFAULT_KERNEL_INIT = lambda *a, **k: nn.initializers.lecun_normal()(*a, **k)
DEFAULT_BIAS_INIT = nn.initializers.zeros
DEFAULT_RECTIFIER = nn.activation.relu


class PositiveDense(nn.Module):
  """A linear transformation using a matrix with all entries non-negative.

  Args:
    dim_hidden: Number of output dimensions.
    rectifier_fn: Rectifier function. The default is
      :func:`~flax.linen.activation.relu`.
    use_bias: Whether to add bias to the output.
    kernel_init: Initializer for the matrix. The default is
      :func:`~flax.linen.initializers.lecun_normal`.
    bias_init: Initializer for the bias. The default is
      :func:`~flax.linen.initializers.zeros`.
    precision: Numerical precision of the computation.
  """

  dim_hidden: int
  rectifier_fn: Callable[[Array], Array] = DEFAULT_RECTIFIER
  use_bias: bool = True
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  precision: Optional[jax.lax.Precision] = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies a linear transformation to x along the last dimension.

    Args:
      x: Array of shape ``[batch, ..., features]``.

    Returns:
      Array of shape ``[batch, ..., dim_hidden]``.
    """
    # TODO(michalk8): update when refactoring neuraldual
    # assert x.ndim > 1, x.ndim

    kernel = self.param(
        "kernel", self.kernel_init, (x.shape[-1], self.dim_hidden)
    )
    kernel = self.rectifier_fn(kernel)

    x = jnp.tensordot(x, kernel, axes=(-1, 0), precision=self.precision)
    if self.use_bias:
      x = x + self.param("bias", self.bias_init, (self.dim_hidden,))

    return x


class PosDefPotentials(nn.Module):
  r""":math:`\frac{1}{2} x^T (A_i A_i^T + \text{Diag}(d_i)) x + b_i^T x^2 + c_i`
    potentials.

  This class implements a layer that takes (batched) ``d``-dimensional vectors
  ``x`` in, to output a ``num_potentials``-dimensional vector. Each of the
  entries in that output is a positive definite quadratic form evaluated at
  ``x``; each of these quadratic terms is parameterized as a low-rank plus
  diagonal matrix. The low-rank term is parameterized as :math:`A_i A_i^T`,
  where each of these matrices is of size ``(rank, d)``. Taken together,
  these matrices form a tensor ``(num_potentials, rank, d)``.
  The diagonal terms :math:`d_i` form a ``(num_potentials, d)`` matrix of
  positive values; the linear terms :math:`b_i` form a ``(num_potentials, d)``
  matrix. Finally, the :math:`c_i` are contained in a vector of size
  ``(num_potentials,)``.

  Args:
    num_potentials: Dimension of the output.
    rank: Rank of the matrices :math:`A_i` used as low-rank factors
      for the quadratic potentials.
    rectifier_fn: Rectifier function to ensure non-negativity of the diagonals
      :math:`d_i`. The default is :func:`~flax.linen.activation.relu`.
    use_linear: Whether to add a linear layers :math:`b_i` to the outputs.
    use_bias: Whether to add biases :math:`c_i` to the outputs.
    kernel_lr_init: Initializer for the matrices :math:`A_i`
      of the quadratic potentials when ``rank > 0``.
      The default is :func:`~flax.linen.initializers.lecun_normal`.
    kernel_diag_init: Initializer for the diagonals :math:`d_i`.
      The default is :func:`~flax.linen.initializers.ones`.
    kernel_linear_init: Initializer for the linear layers :math:`b_i`.
      The default is :func:`~flax.linen.initializers.lecun_normal`.
    bias_init: Initializer for the bias. The default is
      :func:`~flax.linen.initializers.zeros`.
    precision: Numerical precision of the computation.
  """  # noqa: D205,E501

  num_potentials: int
  rank: int = 0
  rectifier_fn: Callable[[Array], Array] = DEFAULT_RECTIFIER
  use_linear: bool = True
  use_bias: bool = True
  kernel_lr_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  kernel_diag_init: Callable[[PRNGKey, Shape, Dtype],
                             Array] = nn.initializers.ones
  kernel_linear_init: Callable[[PRNGKey, Shape, Dtype],
                               Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  precision: Optional[jax.lax.Precision] = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute quadratic forms of the input.

    Args:
      x: Array of shape ``[batch, ..., features]``.

    Returns:
      Array of shape ``[batch, ..., num_potentials]``.
    """
    # TODO(michalk8): update when refactoring neuraldual
    # assert x.ndim > 1, x.ndim

    dim_data = x.shape[-1]
    x = x.reshape((-1, dim_data))

    diag_kernel = self.param(
        "diag_kernel", self.kernel_diag_init, (dim_data, self.num_potentials)
    )
    # ensures the diag_kernel parameter stays non negative
    diag_kernel = self.rectifier_fn(diag_kernel)

    # (batch, dim_data, 1), (1, dim_data, num_potentials)
    y = 0.5 * jnp.sum(((x ** 2)[..., None] * diag_kernel[None]), axis=1)

    if self.rank > 0:
      quad_kernel = self.param(
          "quad_kernel", self.kernel_lr_init,
          (self.num_potentials, dim_data, self.rank)
      )
      # (batch, num_potentials, rank)
      quad = 0.5 * jnp.tensordot(
          x, quad_kernel, axes=(-1, 1), precision=self.precision
      ) ** 2
      y = y + jnp.sum(quad, axis=-1)

    if self.use_linear:
      linear_kernel = self.param(
          "lin_kernel", self.kernel_linear_init,
          (dim_data, self.num_potentials)
      )
      y = y + jnp.dot(x, linear_kernel, precision=self.precision)

    if self.use_bias:
      y = y + self.param("bias", self.bias_init, (self.num_potentials,))

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
      The layer with fixed linear and quadratic initialization.
    """
    factor, mean = _compute_gaussian_map_params(source, target)

    kwargs["use_linear"] = True
    return cls(
        kernel_lr_init=lambda *_, **__: factor,
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
