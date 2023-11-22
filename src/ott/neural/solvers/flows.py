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
    sigma_t = self.compute_sigma_t(t, x_0, x_1)
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
