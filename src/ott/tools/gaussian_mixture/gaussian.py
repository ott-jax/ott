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
import math
from typing import Optional, Union

import jax
import jax.numpy as jnp

from ott.tools.gaussian_mixture import scale_tril

__all__ = ["Gaussian"]

LOG2PI = math.log(2.0 * math.pi)


@jax.tree_util.register_pytree_node_class
class Gaussian:
  """Normal distribution."""

  def __init__(self, loc: jnp.ndarray, scale: scale_tril.ScaleTriL):
    self._loc = loc
    self._scale = scale

  @classmethod
  def from_samples(
      cls,
      points: jnp.ndarray,
      weights: Optional[jnp.ndarray] = None
  ) -> "Gaussian":
    """Construct a Gaussian from weighted samples.

    Unbiased, weighted covariance formula from `GSL
    <https://www.gnu.org/software/gsl/doc/html/statistics.html#weighted-samples>`_.

    Args:
      points: [n x d] array of samples
      weights: [n] array of weights

    Returns:
      Gaussian.
    """
    n = points.shape[0]
    if weights is None:
      weights = jnp.ones(n) / n

    mean = weights.dot(points)
    centered_x = (points - mean)
    scaled_centered_x = centered_x * weights.reshape(-1, 1)
    cov = scaled_centered_x.T.dot(centered_x) / (1 - weights.dot(weights))
    return cls.from_mean_and_cov(mean=mean, cov=cov)

  @classmethod
  def from_random(
      cls,
      rng: jax.Array,
      n_dimensions: int,
      stdev_mean: float = 0.1,
      stdev_cov: float = 0.1,
      ridge: Union[float, jnp.ndarray] = 0,
  ) -> "Gaussian":
    """Construct a random Gaussian.

    Args:
      rng: jax.random key
      n_dimensions: desired covariance dimensions
      stdev_mean: standard deviation of location and log eigenvalues
        (means for both are 0)
      stdev_cov: standard deviated of the covariance
      ridge: Offset for means.

    Returns:
      A random Gaussian.
    """
    rng, subrng0, subrng1 = jax.random.split(rng, num=3)
    loc = jax.random.normal(subrng0, shape=(n_dimensions,)) * stdev_mean + ridge
    scale = scale_tril.ScaleTriL.from_random(
        subrng1, n_dimensions=n_dimensions, stdev=stdev_cov
    )
    return cls(loc=loc, scale=scale)

  @classmethod
  def from_mean_and_cov(cls, mean: jnp.ndarray, cov: jnp.ndarray) -> "Gaussian":
    """Construct a Gaussian from a mean and covariance."""
    scale = scale_tril.ScaleTriL.from_covariance(cov)
    return cls(loc=mean, scale=scale)

  @property
  def loc(self) -> jnp.ndarray:
    """Mean of the Gaussian."""
    return self._loc

  @property
  def scale(self) -> scale_tril.ScaleTriL:
    """Scale of the Gaussian."""
    return self._scale

  @property
  def n_dimensions(self) -> int:
    """Dimensionality of the Gaussian."""
    return self.loc.shape[-1]

  def covariance(self) -> jnp.ndarray:
    """Covariance of the Gaussian."""
    return self.scale.covariance()

  def to_z(self, x: jnp.ndarray) -> jnp.ndarray:
    r"""Transform :math:`x` to :math:`z = \frac{x - loc}{scale}`."""
    return self.scale.centered_to_z(x_centered=x - self.loc)

  def from_z(self, z: jnp.ndarray) -> jnp.ndarray:
    r"""Transform :math:`z` to :math:`x = loc + scale \cdot z`."""
    return self.scale.z_to_centered(z=z) + self.loc

  def log_prob(
      self,
      x: jnp.ndarray,  # (?, d)
  ) -> jnp.ndarray:  # (?, d)
    """Log probability for a Gaussian with a diagonal covariance."""
    d = x.shape[-1]
    z = self.to_z(x)
    log_det = self.scale.log_det_covariance()
    return (
        -0.5 * (d * LOG2PI + log_det[None] + jnp.sum(z ** 2, axis=-1))
    )  # (?, k)

  def sample(self, rng: jax.Array, size: int) -> jnp.ndarray:
    """Generate samples from the distribution."""
    std_samples_t = jax.random.normal(rng, shape=(self.n_dimensions, size))
    return self.loc[None] + (
        jnp.swapaxes(
            jnp.matmul(self.scale.cholesky(), std_samples_t),
            axis1=-2,
            axis2=-1
        )
    )

  def w2_dist(self, other: "Gaussian") -> jnp.ndarray:
    r"""Wasserstein distance :math:`W_2^2` to another Gaussian.

    .. math::

      W_2^2 = ||\mu_0-\mu_1||^2 +
         \text{trace} ( (\Lambda_0^\frac{1}{2} - \Lambda_1^\frac{1}{2})^2 )

    Args:
      other: other Gaussian

    Returns:
      The :math:`W_2^2` distance between self and other
    """
    delta_mean = jnp.sum((self.loc - other.loc) ** 2, axis=-1)
    delta_sigma = self.scale.w2_dist(other.scale)
    return delta_mean + delta_sigma

  def f_potential(self, dest: "Gaussian", points: jnp.ndarray) -> jnp.ndarray:
    """Optimal potential for W2 distance between Gaussians. Evaluated on points.

    Args:
      dest: Gaussian object
      points: samples

    Returns:
      Dual potential, f
    """
    scale_matrix = self.scale.gaussian_map(dest_scale=dest.scale)
    centered_x = points - self.loc
    scaled_x = (scale_matrix @ centered_x.T)

    @jax.vmap
    def batch_inner_product(x, y):
      return x.dot(y)

    return (
        0.5 * batch_inner_product(points, points) -
        0.5 * batch_inner_product(centered_x, scaled_x.T) -
        points.dot(dest.loc)
    )

  def transport(self, dest: "Gaussian", points: jnp.ndarray) -> jnp.ndarray:
    """Transport points according to map between two Gaussian measures.

    Args:
      dest: Gaussian object
      points: samples

    Returns:
      Transported samples
    """
    return self.scale.transport(
        dest_scale=dest.scale, points=points - self.loc[None]
    ) + dest.loc[None]

  def tree_flatten(self):  # noqa: D102
    children = (self.loc, self.scale)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)

  def __hash__(self):
    return jax.tree_util.tree_flatten(self).__hash__()

  def __eq__(self, other):
    return jax.tree_util.tree_flatten(self) == jax.tree_util.tree_flatten(other)
