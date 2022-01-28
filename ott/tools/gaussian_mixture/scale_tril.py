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

"""Pytree for a lower triangular Cholesky factored covariance matrix."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from ott.geometry import costs
from ott.geometry import matrix_square_root
from ott.tools.gaussian_mixture import linalg


@jax.tree_util.register_pytree_node_class
class ScaleTriL:
  """Pytree for a lower triangular Cholesky-factored covariance matrix."""

  def __init__(self, params: jnp.ndarray, size: int):
    self._params = params
    self._size = size

  @classmethod
  def from_points_and_weights(
      cls,
      points: jnp.ndarray,
      weights: jnp.ndarray,
  )  -> Tuple[jnp.ndarray, 'ScaleTriL']:
    """Get a mean and a ScaleTriL from a set of points and weights."""
    mean, cov = linalg.get_mean_and_cov(points=points, weights=weights)
    return mean, cls.from_covariance(cov)

  @classmethod
  def from_random(
      cls,
      key: jnp.ndarray,
      n_dimensions: int,
      stdev: Optional[float] = 0.1,
      dtype: jnp.dtype = jnp.float32,
  ) -> 'ScaleTriL':
    """Construct a random ScaleTriL.

    Args:
      key: pseudo-random number generator key
      n_dimensions: number of dimensions
      stdev: desired standard deviation (around 0) for the log eigenvalues
      dtype: data type for the covariance matrix

    Returns:
      A ScaleTriL.
    """
    # generate a random orthogonal matrix
    key, subkey = jax.random.split(key)
    q = linalg.get_random_orthogonal(key=subkey, dim=n_dimensions, dtype=dtype)

    # generate random eigenvalues
    eigs = stdev * jnp.exp(
        jax.random.normal(key=key, shape=(n_dimensions,), dtype=dtype))

    # random positive definite matrix
    sigma = jnp.matmul(
        jnp.expand_dims(eigs, -2) * q, jnp.transpose(q))

    # cholesky factorization
    chol = jnp.linalg.cholesky(sigma)
    # flatten
    m = linalg.apply_to_diag(chol, jnp.log)
    flat = linalg.tril_to_flat(m)
    return cls(params=flat, size=n_dimensions)

  @classmethod
  def from_cholesky(
      cls,
      cholesky: jnp.ndarray
  ) -> 'ScaleTriL':
    """Construct ScaleTriL from a Cholesky factor of a covariance matrix."""
    m = linalg.apply_to_diag(cholesky, jnp.log)
    flat = linalg.tril_to_flat(m)
    return cls(params=flat, size=cholesky.shape[-1])

  @classmethod
  def from_covariance(
      cls,
      covariance: jnp.ndarray,
  ) -> 'ScaleTriL':
    """Construct ScaleTriL from a covariance matrix."""
    cholesky = jnp.linalg.cholesky(covariance)
    return cls.from_cholesky(cholesky)

  @property
  def params(self) -> jnp.ndarray:
    """Internal representation."""
    return self._params

  @property
  def size(self) -> int:
    """Size of the covariance matrix."""
    return self._size

  @property
  def dtype(self):
    """Data type of the covariance matrix."""
    return self._params.dtype

  def cholesky(self) -> jnp.ndarray:
    """Get a lower triangular Cholesky factor for the covariance matrix."""
    m = linalg.flat_to_tril(self._params, size=self._size)
    return linalg.apply_to_diag(m, jnp.exp)

  def covariance(self) -> jnp.ndarray:
    """Get the covariance matrix."""
    cholesky = self.cholesky()
    return jnp.matmul(cholesky, jnp.transpose(cholesky))

  def covariance_sqrt(self) -> jnp.ndarray:
    """Get the square root of the covariance matrix."""
    return linalg.matrix_powers(self.covariance(), (0.5,))[0]

  def log_det_covariance(self) -> jnp.ndarray:
    """Get the log of the determinant of the covariance matrix."""
    diag = jnp.diagonal(self.cholesky(), axis1=-2, axis2=-1)
    return 2. * jnp.sum(jnp.log(diag), axis=-1)

  def centered_to_z(self, x_centered: jnp.ndarray) -> jnp.ndarray:
    """Map centered points to standardized centered points (i.e. cov(z) = I)."""
    return linalg.invmatvectril(
        m=self.cholesky(), x=x_centered, lower=True)

  def z_to_centered(self, z: jnp.ndarray) -> jnp.ndarray:
    """Scale standardized points to points with the specified covariance."""
    return jnp.transpose(jnp.matmul(self.cholesky(), jnp.transpose(z)))

  def w2_dist(
      self,
      other: 'ScaleTriL') -> jnp.ndarray:
    r"""Wasserstein distance W_2^2 to another Gaussian with same mean.

    Args:
      other: Scale for the other Gaussian

    Returns:
      The W_2^2 distance
    """
    dimension = self.size

    def _flatten_cov(cov: jnp.ndarray) -> jnp.ndarray:
      cov = cov.reshape(cov.shape[:-2] + (dimension * dimension,))
      return jnp.concatenate([jnp.zeros(dimension), cov], axis=-1)

    x0 = _flatten_cov(self.covariance())
    x1 = _flatten_cov(other.covariance())
    cost_fn = costs.Bures(dimension=dimension)
    return (cost_fn.norm(x0) + cost_fn.norm(x1) +
            cost_fn.pairwise(x0, x1))[..., 0]

  def transport(
      self,
      dest_scale: 'ScaleTriL',
      points: jnp.ndarray
  ) -> jnp.ndarray:
    """Transport between 0-mean normal w/ current scale to one w/ dest_scale.

    Args:
      dest_scale: destination Scale
      points: points to transport

    Returns:
      Points transported to a Gaussian with the new scale.
    """
    sqrt0, sqrt0_inv = linalg.matrix_powers(self.covariance(), (0.5, -0.5))
    sigma1 = dest_scale.covariance()
    m = matrix_square_root.sqrtm_only(
        jnp.matmul(sqrt0, jnp.matmul(sigma1, sqrt0)))
    m = jnp.matmul(sqrt0_inv, jnp.matmul(m, sqrt0_inv))
    return jnp.transpose(jnp.matmul(m, jnp.transpose(points)))

  def tree_flatten(self):
    children = (self.params,)
    aux_data = {'size': self.size}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  def __repr__(self):
    class_name = type(self).__name__
    children, aux = self.tree_flatten()
    return '{}({})'.format(
        class_name, ', '.join([repr(c) for c in children] +
                              [f'{k}: {repr(v)}' for k, v in aux.items()]))

  def __hash__(self):
    return jax.tree_util.tree_flatten(self).__hash__()

  def __eq__(self, other):
    return jax.tree_util.tree_flatten(self) == jax.tree_util.tree_flatten(other)
