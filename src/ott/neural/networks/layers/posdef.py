# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Positive-weight dense layer for input convex neural networks."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from flax import nnx

__all__ = ["PositiveDense", "PosDefPotentials"]

DEFAULT_KERNEL_INIT = nnx.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nnx.initializers.zeros_init()
DEFAULT_DIAG_INIT = nnx.initializers.constant(-2.0)


def _sinkhorn_normalize(
    log_kernel: jax.Array,
    num_iter: int = 10,
    epsilon: float = 0.1,
) -> jax.Array:
  """Sinkhorn normalization in log-space for positive weight matrices."""
  log_k = log_kernel / epsilon

  def body_fn(carry, _):
    log_u, log_v = carry
    log_u = -jax.nn.logsumexp(log_k + log_v[None, :], axis=1)
    log_v = -jax.nn.logsumexp(log_k + log_u[:, None], axis=0)
    return (log_u, log_v), None

  d_in, d_out = log_kernel.shape
  log_u = jnp.zeros(d_in)
  log_v = jnp.zeros(d_out)
  (log_u, log_v), _ = jax.lax.scan(
      body_fn, (log_u, log_v), None, length=num_iter
  )
  return jnp.exp(log_k + log_u[:, None] + log_v[None, :])


class PositiveDense(nnx.Module):
  """A linear transformation with non-negative weights.

  Three modes for enforcing positivity:

  - **Element-wise rectifier** (default): applies ``rectifier_fn``
    (e.g., softplus, relu) to each weight independently.
  - **Softmax** (``use_softmax=True``): column-wise softmax so each
    column sums to 1, producing stochastic weight matrices.
  - **Sinkhorn** (``use_sinkhorn=True``): Sinkhorn normalization in
    log-space produces approximately doubly-stochastic matrices.

  Args:
    in_features: Input dimension.
    out_features: Output dimension.
    rectifier_fn: Function to enforce non-negativity. Ignored when
      ``use_softmax`` or ``use_sinkhorn`` is True.
    use_softmax: If True, use column-wise softmax normalization.
    use_sinkhorn: If True, use Sinkhorn normalization.
    use_bias: Whether to add a bias term.
    kernel_init: Initializer for the kernel.
    bias_init: Initializer for the bias.
    rngs: Random number generators.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      rectifier_fn: Optional[Callable[[jax.Array],
                                      jax.Array]] = jax.nn.softplus,
      use_softmax: bool = False,
      use_sinkhorn: bool = False,
      use_bias: bool = True,
      kernel_init: nnx.initializers.Initializer = DEFAULT_KERNEL_INIT,
      bias_init: nnx.initializers.Initializer = DEFAULT_BIAS_INIT,
      rngs: nnx.Rngs,
  ):
    self.rectifier_fn = rectifier_fn
    self.use_softmax = use_softmax
    self.use_sinkhorn = use_sinkhorn
    if out_features == 1 and use_sinkhorn:
      self.use_sinkhorn = False
      self.use_softmax = True
    self.kernel = nnx.Param(
        kernel_init(rngs.params(), (in_features, out_features))
    )
    self.bias = (
        nnx.Param(bias_init(rngs.params(),
                            (out_features,))) if use_bias else None
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Apply positive-weight linear transformation."""
    kernel = self._get_positive_kernel()
    out = x @ kernel
    if self.bias is not None:
      out = out + self.bias[...]
    return out

  def _get_positive_kernel(self) -> jax.Array:
    """Get the positive kernel via the configured normalization."""
    raw = self.kernel[...]
    if self.use_sinkhorn:
      return _sinkhorn_normalize(jnp.clip(raw, -5.0, 5.0))
    if self.use_softmax:
      return jax.nn.softmax(raw, axis=0)
    return self.rectifier_fn(raw)


class PosDefPotentials(nnx.Module):
  """Low-rank plus diagonal positive definite quadratic potentials.

  Computes: sum_i 0.5 * x^T (A_i A_i^T + diag(d_i)) x + b_i^T x + c_i

  This is used as an optional additive term in the ICNN to ensure
  strong convexity.

  Args:
    in_features: Input dimension.
    num_potentials: Number of output potentials.
    rank: Rank of the low-rank factors A_i.
    use_linear: Whether to include the linear term b^T x.
    use_bias: Whether to include the scalar bias c.
    rngs: Random number generators.
  """

  def __init__(
      self,
      in_features: int,
      num_potentials: int,
      *,
      rank: int = 1,
      use_linear: bool = True,
      use_bias: bool = True,
      kernel_diag_init: nnx.initializers.Initializer = DEFAULT_DIAG_INIT,
      kernel_lr_init: nnx.initializers.Initializer = DEFAULT_KERNEL_INIT,
      kernel_linear_init: nnx.initializers.Initializer = DEFAULT_KERNEL_INIT,
      bias_init: nnx.initializers.Initializer = DEFAULT_BIAS_INIT,
      rectifier_fn: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
      rngs: nnx.Rngs,
  ):
    self.rectifier_fn = rectifier_fn
    self.num_potentials = num_potentials

    # Diagonal: [num_potentials, in_features]
    self.kernel_diag = nnx.Param(
        kernel_diag_init(rngs.params(), (num_potentials, in_features))
    )
    # Low-rank factors: [num_potentials, in_features, rank]
    self.kernel_lr = nnx.Param(
        kernel_lr_init(rngs.params(), (num_potentials, in_features, rank))
    )
    # Linear term: [num_potentials, in_features]
    self.kernel_linear = (
        nnx.Param(
            kernel_linear_init(rngs.params(), (num_potentials, in_features))
        ) if use_linear else None
    )
    # Bias: [num_potentials]
    self.bias = (
        nnx.Param(bias_init(rngs.params(),
                            (num_potentials,))) if use_bias else None
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Evaluate positive definite quadratic potentials.

    Args:
      x: Input array of shape ``[..., in_features]``.

    Returns:
      Output array of shape ``[..., num_potentials]``.
    """
    # Quadratic term: 0.5 * x^T (A A^T + diag(d)) x
    diag = self.rectifier_fn(self.kernel_diag[...])  # [n_pot, d]
    lr = self.kernel_lr[...]  # [n_pot, d, rank]

    # x: [..., d] -> [..., 1, d]
    x_expanded = x[..., None, :]

    # Diagonal part: sum_d x_d^2 * diag_d -> [..., n_pot]
    quad_diag = jnp.sum(x_expanded ** 2 * diag, axis=-1)

    # Low-rank part: ||A^T x||^2 -> [..., n_pot]
    # x_expanded: [..., 1, d], lr: [n_pot, d, rank]
    atx = jnp.einsum("...d,ndr->...nr", x, lr)  # [..., n_pot, rank]
    quad_lr = jnp.sum(atx ** 2, axis=-1)  # [..., n_pot]

    out = 0.5 * (quad_diag + quad_lr)

    # Linear term
    if self.kernel_linear is not None:
      linear = jnp.einsum("...d,nd->...n", x, self.kernel_linear[...])
      out = out + linear

    # Bias
    if self.bias is not None:
      out = out + self.bias[...]

    return out
