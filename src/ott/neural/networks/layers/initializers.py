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
"""Principled initialization for input convex neural networks.

Implements the initialization scheme from :cite:`hoedt:2023`
for ICNNs with non-negative weights, ensuring controlled correlation
and variance propagation through layers.
"""
import math
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp

from flax import nnx

__all__ = ["get_rectifier_inverse", "principled_icnn_init"]

Initializer = Callable[[jax.Array, Tuple[int, ...], jnp.dtype], jax.Array]
RectifierName = Literal["exp", "softplus", "relu", "identity"]


def _softplus_inv(x: jax.Array) -> jax.Array:
  return x + jnp.log(-jnp.expm1(-x))


def get_rectifier_inverse(
    rectifier_fn: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
  """Return the inverse of a rectifier function.

  Args:
    rectifier_fn: A rectifier function (softplus, relu, exp, or identity).

  Returns:
    The inverse function.
  """
  if rectifier_fn is jax.nn.softplus:
    return _softplus_inv
  if rectifier_fn is jnp.exp:
    return jnp.log
  # relu, identity — no invertible mapping; use identity as fallback
  return lambda x: x


def _principled_icnn_weights(
    fan_in: int,
    *,
    alpha: float,
    rho: float = 0.5,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Compute log-normal weight parameters for principled ICNN init."""

  def _corr_func(fan_in: int, *, rho: float) -> float:
    mix_mom = (1 - rho ** 2) ** 0.5 + rho * math.acos(-rho)
    return fan_in * (math.pi - fan_in + (fan_in - 1) * mix_mom) / (2 * math.pi)

  mean_sq = rho / _corr_func(fan_in, rho=rho)
  var = (2.0 / (1.0 + (alpha ** 2))) * (1.0 / fan_in) * (1.0 - rho)

  log_mom2 = jnp.log(mean_sq + var)
  log_mean = jnp.log(mean_sq) - log_mom2 / 2.0
  log_var = log_mom2 - math.log(mean_sq)

  return log_mean, log_var, mean_sq


def _principled_icnn_biases(
    fan_in: int,
    *,
    target_mean_sq: float,
    target_var: float = 1.0,
) -> float:
  """Compute bias initialization for principled ICNN init."""
  return -fan_in * (target_mean_sq * target_var / (2 * math.pi)) ** 0.5


def principled_icnn_init(
    fan_in: int,
    *,
    alpha: float,
    rectifier_fn: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
    target_rho: float = 0.5,
    target_var: float = 1.0,
) -> Tuple[Initializer, Initializer]:
  """Compute principled weight and bias initializers for ICNN layers.

  Implements the initialization from :cite:`hoedt:2023` that
  ensures controlled correlation (``target_rho``) and variance
  (``target_var``) propagation through ICNN layers with positive
  weights.

  Args:
    fan_in: Input dimension of the layer.
    alpha: Negative slope of the activation function (0 for ReLU,
      small positive for LeakyReLU, 1 for identity).
    rectifier_fn: The rectifier used by ``PositiveDense``.
    target_rho: Target correlation between consecutive layers.
    target_var: Target output variance.

  Returns:
    A ``(weights_init, biases_init)`` tuple of initializer functions.
  """
  inv_fn = get_rectifier_inverse(rectifier_fn)

  if target_rho == 0.5 and target_var == 1.0:
    return _principled_init_fixed(fan_in, alpha=alpha, inv_fn=inv_fn)

  w_log_mean, w_log_var, w_mean_sq = _principled_icnn_weights(
      fan_in, alpha=alpha, rho=target_rho
  )
  b_mean = _principled_icnn_biases(
      fan_in, target_mean_sq=w_mean_sq, target_var=target_var
  )

  def weights_init(
      rng: jax.Array,
      shape: Tuple[int, ...],
      dtype: jnp.dtype = None
  ) -> jax.Array:
    w = nnx.initializers.normal(stddev=w_log_var ** 0.5)(rng, shape, dtype)
    w = jnp.exp(w_log_mean + w)
    return inv_fn(w)

  def biases_init(
      rng: jax.Array,
      shape: Tuple[int, ...],
      dtype: jnp.dtype = None
  ) -> jax.Array:
    return nnx.initializers.constant(b_mean)(rng, shape, dtype)

  return weights_init, biases_init


def _principled_init_fixed(
    fan_in: int,
    *,
    alpha: float,
    inv_fn: Callable[[jax.Array], jax.Array],
) -> Tuple[Initializer, Initializer]:
  """Optimized principled init for target_rho=0.5, target_var=1.0."""

  def get_factors():
    a = ((1 - alpha) ** 2) * (
        6 * math.pi - 6 * fan_in + (fan_in - 1) *
        (3 * (3 ** 0.5) + 2 * math.pi)
    )
    b = 6 * (fan_in + 1) * math.pi * alpha
    return a, b

  def sample_weights(rng, shape, dtype, *, log_mean, log_var):
    w = nnx.initializers.normal(stddev=log_var ** 0.5)(rng, shape, dtype)
    w = jnp.exp(log_mean + w)
    return inv_fn(w)

  def weights_init(rng, shape, dtype=None):
    a, b = get_factors()
    mean_sq = 6 * math.pi / (fan_in * (a + b))
    var = (1.0 / (1 + (alpha ** 2))) * (1.0 / fan_in)

    log_mom2 = math.log(mean_sq + var)
    log_mean = math.log(mean_sq) - log_mom2 / 2.0
    log_var = log_mom2 - math.log(mean_sq)

    return sample_weights(rng, shape, dtype, log_mean=log_mean, log_var=log_var)

  def biases_init(rng, shape, dtype=None):
    a, b = get_factors()
    mean = (3 * fan_in * ((1 - alpha) ** 2) / (a + b)) ** 0.5
    return nnx.initializers.constant(mean)(rng, shape, dtype)

  return weights_init, biases_init
