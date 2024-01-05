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

import jax.numpy as jnp

__all__ = [
    "BaseFlow",
    "StraightFlow",
    "ConstantNoiseFlow",
    "BrownianNoiseFlow",
]


class BaseFlow(abc.ABC):
  """Base class for all flows.

  Args:
    sigma: Constant noise used for computing time-dependent noise schedule.
  """

  def __init__(self, sigma: float):
    self.sigma = sigma

  @abc.abstractmethod
  def compute_mu_t(
      self, t: jnp.ndarray, src: jnp.ndarray, tgt: jnp.ndarray
  ) -> jnp.ndarray:
    """Compute the mean of the probablitiy path.

    Compute the mean of the probablitiy path between :math:`x` and :math:`y`
    at time :math:`t`.

    Args:
      t: Time :math:`t`.
      src: Sample from the source distribution.
      tgt: Sample from the target distribution.
    """

  @abc.abstractmethod
  def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute the standard deviation of the probablity path at time :math:`t`.

    Args:
      t: Time :math:`t`.
    """

  @abc.abstractmethod
  def compute_ut(
      self, t: jnp.ndarray, src: jnp.ndarray, tgt: jnp.ndarray
  ) -> jnp.ndarray:
    """Evaluate the conditional vector field.

    Evaluate the conditional vector field defined between :math:`x_0` and
    :math:`x_1` at time :math:`t`.

    Args:
      t: Time :math:`t`.
      src: Sample from the source distribution.
      tgt: Sample from the target distribution.
    """

  def compute_xt(
      self, noise: jnp.ndarray, t: jnp.ndarray, src: jnp.ndarray,
      tgt: jnp.ndarray
  ) -> jnp.ndarray:
    """Sample from the probability path.

    Sample from the probability path between :math:`x_0` and :math:`x_1` at
    time :math:`t`.

    Args:
      noise: Noise sampled from a standard normal distribution.
      t: Time :math:`t`.
      src: Sample from the source distribution.
      tgt: Sample from the target distribution.

    Returns:
      Samples from the probability path between :math:`x_0` and :math:`x_1`
      at time :math:`t`.
    """
    mu_t = self.compute_mu_t(t, src, tgt)
    sigma_t = self.compute_sigma_t(t)
    return mu_t + sigma_t * noise


class StraightFlow(BaseFlow, abc.ABC):
  """Base class for flows with straight paths."""

  def compute_mu_t(  # noqa: D102
      self, t: jnp.ndarray, src: jnp.ndarray, tgt: jnp.ndarray
  ) -> jnp.ndarray:
    return (1 - t) * src + t * tgt

  def compute_ut(
      self, t: jnp.ndarray, src: jnp.ndarray, tgt: jnp.ndarray
  ) -> jnp.ndarray:
    """Evaluate the conditional vector field.

    Evaluate the conditional vector field defined between :math:`x_0` and
    :math:`x_1` at time :math:`t`.

    Args:
      t: Time :math:`t`.
      src: Sample from the source distribution.
      tgt: Sample from the target distribution.

    Returns:
      Conditional vector field evaluated at time :math:`t`.
    """
    return tgt - src


class ConstantNoiseFlow(StraightFlow):
  r"""Flow with straight paths and constant flow noise :math:`\sigma`."""

  def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    r"""Compute noise of the flow at time :math:`t`.

    Args:
      t: Time :math:`t`.

    Returns:
      Constant, time-independent standard deviation :math:`\sigma`.
    """
    return self.sigma


class BrownianNoiseFlow(StraightFlow):
  r"""Brownian Bridge Flow.

  Sampler for sampling noise implicitly defined by a Schroedinger Bridge
  problem with parameter `\sigma` such that
  :math:`\sigma_t = \sigma * \sqrt(t * (1-t))`.
  """

  def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute the standard deviation of the probablity path at time :math:`t`.

    Args:
      t: Time :math:`t`.

    Returns:
      Standard deviation of the probablity path at time :math:`t`.
    """
    return jnp.sqrt(self.sigma * t * (1 - t))
