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


class BaseFlow(abc.ABC):

  def __init__(self, sigma: float) -> None:
    self.sigma = sigma

  @abc.abstractmethod
  def compute_mu_t(self, t: jax.Array, x_0: jax.Array, x_1: jax.Array):
    pass

  @abc.abstractmethod
  def compute_sigma_t(self, t: jax.Array):
    pass

  @abc.abstractmethod
  def compute_ut(
      self, t: jax.Array, x_0: jax.Array, x_1: jax.Array
  ) -> jax.Array:
    pass

  def compute_xt(
      self, noise: jax.Array, t: jax.Array, x_0: jax.Array, x_1: jax.Array
  ) -> jax.Array:
    mu_t = self.compute_mu_t(t, x_0, x_1)
    sigma_t = self.compute_sigma_t(t)
    return mu_t + sigma_t * noise


class StraightFlow(BaseFlow):

  def compute_mu_t(
      self, t: jax.Array, x_0: jax.Array, x_1: jax.Array
  ) -> jax.Array:
    return t * x_0 + (1 - t) * x_1

  def compute_ut(
      self, t: jax.Array, x_0: jax.Array, x_1: jax.Array
  ) -> jax.Array:
    return x_1 - x_0


class ConstantNoiseFlow(StraightFlow):

  def compute_sigma_t(self, t: jax.Array):
    return self.sigma


class BrownianNoiseFlow(StraightFlow):

  def compute_sigma_t(self, t: jax.Array):
    return jnp.sqrt(self.sigma * t * (1 - t))


class BaseTimeSampler(abc.ABC):

  @abc.abstractmethod
  def __call__(self, rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    pass


class UniformSampler(BaseTimeSampler):

  def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
    self.low = low
    self.high = high

  def __call__(self, rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    return jax.random.uniform(
        rng, (num_samples, 1), minval=self.low, maxval=self.high
    )


class OffsetUniformSampler(BaseTimeSampler):

  def __init__(
      self, offset: float, low: float = 0.0, high: float = 1.0
  ) -> None:
    self.offset = offset
    self.low = low
    self.high = high

  def __call__(self, rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    return (
        jax.random.uniform(rng, (1, 1), minval=self.low, maxval=self.high) +
        jnp.arange(num_samples)[:, None] / num_samples
    ) % ((self.high - self.low) - self.offset)
