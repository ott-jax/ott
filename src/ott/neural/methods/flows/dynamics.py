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

import jax
import jax.numpy as jnp

__all__ = [
    "BaseFlow",
    "StraightFlow",
    "ConstantNoiseFlow",
    "BrownianBridge",
]


class BaseFlow(abc.ABC):
  """Base class for all flows.

  Args:
    sigma: Noise used for computing time-dependent noise schedule.
  """

  def __init__(self, sigma: float):
    self.sigma = sigma

  @abc.abstractmethod
  def compute_mu_t(
      self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
  ) -> jnp.ndarray:
    """Compute the mean of the probability path.

    Compute the mean of the probability path between :math:`x_0` and :math:`x_1`
    at time :math:`t`.

    Args:
      t: Time :math:`t` of shape ``[batch, 1]``.
      x0: Sample from the source distribution of shape ``[batch, ...]``.
      x1: Sample from the target distribution of shape ``[batch, ...]``.
    """

  @abc.abstractmethod
  def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute the standard deviation of the probability path at time :math:`t`.

    Args:
      t: Time :math:`t` of shape ``[batch, 1]``.

    Returns:
      Standard deviation of the probability path at time :math:`t`.
    """

  @abc.abstractmethod
  def compute_ut(
      self, t: jnp.ndarray, x: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
  ) -> jnp.ndarray:
    """Evaluate the conditional vector field.

    Evaluate the conditional vector field defined between :math:`x_0` and
    :math:`x_1` at time :math:`t`.

    Args:
      t: Time :math:`t` of shape ``[batch, 1]``.
      x: Current position :math:`x` of shape ``[batch, ...]``.
      x0: Source position :math:`x_0` of shape ``[batch, ...]``.
      x1: Target position :math:`x_1` of shape ``[batch, ...]``.

    Returns:
      Conditional vector field evaluated at time :math:`t`.
    """

  def compute_xt(
      self, rng: jax.Array, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
  ) -> jnp.ndarray:
    """Sample from the probability path.

    Sample from the probability path between :math:`x_0` and :math:`x_1` at
    time :math:`t`.

    Args:
      rng: Random number generator.
      t: Time :math:`t` of shape ``[batch, 1]``.
      x0: Sample from the source distribution of shape ``[batch, ...]``.
      x1: Sample from the target distribution of shape ``[batch, ...]``.

    Returns:
      Samples from the probability path between :math:`x_0` and :math:`x_1`
      at time :math:`t`.
    """
    noise = jax.random.normal(rng, shape=x0.shape)
    mu_t = self.compute_mu_t(t, x0, x1)
    sigma_t = self.compute_sigma_t(t)
    return mu_t + sigma_t * noise


class StraightFlow(BaseFlow, abc.ABC):
  """Base class for flows with straight paths.

  Args:
    sigma: Noise used for computing time-dependent noise schedule.
  """

  def compute_mu_t(  # noqa: D102
      self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
  ) -> jnp.ndarray:
    return (1.0 - t) * x0 + t * x1

  def compute_ut(  # noqa: D102
      self, t: jnp.ndarray, x: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
  ) -> jnp.ndarray:
    del t, x
    return x1 - x0


class ConstantNoiseFlow(StraightFlow):
  r"""Flow with straight paths and constant flow noise :math:`\sigma`.

  Args:
    sigma: Constant noise used for computing time-independent noise schedule.
  """

  def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    r"""Compute noise of the flow at time :math:`t`.

    Args:
      t: Time :math:`t` of shape ``[batch, 1]``.

    Returns:
      Constant, time-independent standard deviation :math:`\sigma`.
    """
    return jnp.full_like(t, fill_value=self.sigma)


class BrownianBridge(StraightFlow):
  r"""Brownian Bridge.

  Sampler for sampling noise implicitly defined by a Schrödinger Bridge
  problem with parameter :math:`\sigma` such that
  :math:`\sigma_t = \sigma \cdot \sqrt{t \cdot (1 - t)}` :cite:`tong:23`.

  Args:
    sigma: Noise used for computing time-dependent noise schedule.
  """

  def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    r"""Compute noise of the flow at time :math:`t`.

    Args:
      t: Time :math:`t` of shape ``[batch, 1]``.

    Returns:
      Samples from the probability path between :math:`x_0` and :math:`x_1`
      at time :math:`t`.
    """
    return self.sigma * jnp.sqrt(t * (1.0 - t))

  def compute_ut(
      self, t: jnp.ndarray, x: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
  ) -> jnp.ndarray:
    r"""Compute the conditional vector field :math:`u_t(x|z)`.

    Args:
        t: Time :math:`t` of shape ``[batch, 1]``.
        x: Current position :math:`x` of shape ``[batch, ...]``.
        x0: Source position :math:`x_0` of shape ``[batch, ...]``.
        x1: Target position :math:`x_1` of shape ``[batch, ...]``.

    Returns:
        The vector field :math:`u_t(x|z)` at time :math:`t`.
    """
    drift_term = (1 - 2 * t) / (2 * t * (1 - t)) * (x - (t * x1 + (1 - t) * x0))
    control_term = x1 - x0
    return drift_term + control_term
