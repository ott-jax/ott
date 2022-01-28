# coding=utf-8
# Copyright 2022 Google LLC.
#
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

"""Pytree for a normal distribution."""

import math
from typing import Optional

import jax
import jax.numpy as jnp
from ott.tools.gaussian_mixture import scale_tril

LOG2PI = math.log(2. * math.pi)


@jax.tree_util.register_pytree_node_class
class Gaussian:
  """PyTree for a normal distribution."""

  def __init__(self, loc: jnp.ndarray, scale: scale_tril.ScaleTriL):
    self._loc = loc
    self._scale = scale

  @classmethod
  def from_random(
      cls,
      key: jnp.ndarray,
      n_dimensions: int,
      stdev: float = 0.1,
      dtype: Optional[jnp.dtype] = None
  ) -> 'Gaussian':
    """Construct a random Gaussian.

    Args:
      key: jax.random seed
      n_dimensions: desired covariance dimensions
      stdev: standard deviation of loc and log eigenvalues
        (means for both are 0)
      dtype: data type

    Returns:
      A random Gaussian.
    """
    key, subkey0, subkey1 = jax.random.split(key, num=3)
    loc = jax.random.normal(
        key=subkey0, shape=(n_dimensions,), dtype=dtype) * stdev
    scale = scale_tril.ScaleTriL.from_random(
        key=subkey1, n_dimensions=n_dimensions, stdev=stdev, dtype=dtype)
    return cls(loc=loc, scale=scale)

  @classmethod
  def from_mean_and_cov(
      cls,
      mean: jnp.ndarray,
      cov: jnp.ndarray):
    """Construct a Gaussian from a mean and covariance."""
    scale = scale_tril.ScaleTriL.from_covariance(cov)
    return cls(loc=mean, scale=scale)

  @property
  def loc(self) -> jnp.ndarray:
    return self._loc

  @property
  def scale(self) -> scale_tril.ScaleTriL:
    return self._scale

  @property
  def n_dimensions(self) -> int:
    return self.loc.shape[-1]

  def covariance(self) -> jnp.ndarray:
    return self.scale.covariance()

  def to_z(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.scale.centered_to_z(x_centered=x - self.loc)

  def from_z(self, z: jnp.ndarray) -> jnp.ndarray:
    return self.scale.z_to_centered(z=z) + self.loc

  def log_prob(
      self,
      x: jnp.ndarray,  # (?, d)
  ) -> jnp.ndarray:  # (?, d)
    """Log probability for a gaussian with a diagonal covariance."""
    d = x.shape[-1]
    z = self.to_z(x)
    log_det = self.scale.log_det_covariance()
    return (-0.5 * (d * LOG2PI +
                    log_det[None] +
                    jnp.sum(z ** 2., axis=-1)))  # (?, k)

  def sample(self, key: jnp.ndarray, size: int) -> jnp.ndarray:
    """Generate samples from the distribution."""
    std_samples_t = jax.random.normal(
        key=key, shape=(self.n_dimensions, size))
    return self.loc[None] + (
        jnp.swapaxes(jnp.matmul(self.scale.cholesky(), std_samples_t),
                     axis1=-2, axis2=-1))

  def w2_dist(self, other: 'Gaussian') -> jnp.ndarray:
    r"""Wasserstein distance W_2^2 to another Gaussian.

    W_2^2 = ||\mu_0-\mu_1||^2 +
       \text{trace} ( (\Lambda_0^\frac{1}{2} - \Lambda_1^\frac{1}{2})^2 )

    Args:
      other: other Gaussian

    Returns:
      The W_2^2 distance between self and other
    """
    delta_mean = jnp.sum((self.loc - other.loc) ** 2., axis=-1)
    delta_sigma = self.scale.w2_dist(other.scale)
    return delta_mean + delta_sigma

  def transport(
      self,
      dest: 'Gaussian',
      points: jnp.ndarray) -> jnp.ndarray:
    return self.scale.transport(
        dest_scale=dest.scale,
        points=points - self.loc[None]) + dest.loc[None]

  def tree_flatten(self):
    children = (self.loc, self.scale)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  def __hash__(self):
    return jax.tree_util.tree_flatten(self).__hash__()

  def __eq__(self, other):
    return jax.tree_util.tree_flatten(self) == jax.tree_util.tree_flatten(other)
