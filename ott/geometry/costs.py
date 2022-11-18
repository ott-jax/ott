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

# Lint as: python3
"""Several cost/norm functions for relevant vector types."""
import abc
import functools
import math
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott.math import fixed_point_loop, matrix_square_root

__all__ = [
    "PNorm", "SqPNorm", "Euclidean", "SqEuclidean", "Cosine", "Bures",
    "UnbalancedBures"
]


@jax.tree_util.register_pytree_node_class
class CostFn(abc.ABC):
  """A generic cost function, taking two vectors as input.

  Cost functions evaluate a function on a pair of inputs. For convenience,
  that function is split into two norms -- evaluated on each input separately --
  followed by a pairwise cost that involves both inputs, as in:

  .. math::

    c(x,y) = norm(x) + norm(y) + pairwise(x,y)

  If the norm function is not implemented, that value is handled as a 0.
  """

  # no norm function created by default.
  norm: Optional[Callable[[jnp.ndarray], Union[float, jnp.ndarray]]] = None

  @abc.abstractmethod
  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    pass

  # TODO(michalk8): make weights optional?
  def barycenter(self, weights: jnp.ndarray, xs: jnp.ndarray) -> float:
    """Barycentric projection.

    Args:
      weights: TODO.
      xs: TODO.

    Returns:
      TODO.
    """
    raise NotImplementedError("Barycenter is not yet implemented.")

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    """TODO.

    Args:
      dim: TODO.

    Returns:
      TODO.
    """
    return jnp.zeros((1, dim))

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    cost = self.pairwise(x, y)
    if self.norm is None:
      return cost
    return cost + self.norm(x) + self.norm(y)

  def all_pairs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all costs (including norms) for vectors in x / y.

    Args:
      x: [num_a, d] jnp.ndarray
      y: [num_b, d] jnp.ndarray
    Returns:
      [num_a, num_b] matrix of cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self(x_, y_))(y))(x)

  def all_pairs_pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all pairwise-costs (no norms) for vectors in x / y.

    Args:
      x: [num_a, d] jnp.ndarray
      y: [num_b, d] jnp.ndarray
    Returns:
      [num_a, num_b] matrix of pairwise cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self.pairwise(x_, y_))(y))(x)

  def tree_flatten(self):
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class TICost(CostFn):
  """A class for translation invariant (TI) costs.

  Such costs are defined using a function :math:`h`, mapping vectors to
  real-values, to be used as:

  .. math::

    c(x,y) = h(z), z := x-y.

  If that cost function is used to form an Entropic map using the
  :cite:`brenier:91` theorem, then the user should ensure :math:`h` is
  strictly convex, as well as provide the Legendre transform of :math:`h`,
  whose gradient is necessarily the inverse of the gradient of :math:`h`.
  """

  @abc.abstractmethod
  def h(self, z: jnp.ndarray) -> float:
    """TI function acting on difference of :math:`x-y` to output cost."""

  def h_legendre(self, z: jnp.ndarray) -> float:
    """Legendre transform of :func:`h` when it is convex."""
    raise NotImplementedError("`h_legendre` not implemented.")

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost as evaluation of :func:`h` on :math:`x-y`."""
    return self.h(x - y)


@jax.tree_util.register_pytree_node_class
class SqPNorm(TICost):
  """Squared p-norm of the difference of two vectors.

  For details on the derivation of the Legendre transform of the norm, see e.g.
  the reference :cite:`boyd:04`, p.93/94.

  Args:
    p: TODO.
  """

  def __init__(self, p: float):
    super().__init__()
    assert p >= 1.0, "p parameter in sq. p-norm should be >= 1.0"
    self.p = p
    self.q = 1. / (1. - 1. / self.p) if p > 1.0 else "inf"

  def h(self, z: jnp.ndarray) -> float:
    return 0.5 * jnp.linalg.norm(z, self.p) ** 2

  def h_legendre(self, z: jnp.ndarray) -> float:
    return 0.5 * jnp.linalg.norm(z, self.q) ** 2

  def tree_flatten(self):
    return (), (self.p,)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del children
    return cls(aux_data[0])


@jax.tree_util.register_pytree_node_class
class PNorm(TICost):
  """p-norm (to the power p) of the difference of two vectors.

  Args:
    p: TODO.
  """

  def __init__(self, p: float):
    super().__init__()
    assert p >= 1.0, "p parameter in p-norm should be >= 1.0"
    self.p = p
    self.q = 1. / (1. - 1. / self.p) if p > 1. else "inf"

  def h(self, z: jnp.ndarray) -> float:
    return jnp.linalg.norm(z, self.p) ** self.p / self.p

  def h_legendre(self, z: jnp.ndarray) -> float:
    return jnp.linalg.norm(z, self.q) ** self.q / self.q

  def tree_flatten(self):
    return (), (self.p,)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del children
    return cls(aux_data[0])


@jax.tree_util.register_pytree_node_class
class Euclidean(CostFn):
  """Euclidean distance.

  Note that the Euclidean distance is not cast as a
  :class:`~ott.geometry.costs.TICost`, since this would correspond to :math:`h`
  being :func:`jax.numpy.linalg.norm`, whose gradient is not invertible,
  because the function is not strictly convex (it is linear on rays).
  """

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute Euclidean norm."""
    return jnp.linalg.norm(x - y)


@jax.tree_util.register_pytree_node_class
class SqEuclidean(TICost):
  """Squared Euclidean distance."""

  def norm(self, x: jnp.ndarray) -> Union[float, jnp.ndarray]:
    """Compute squared Euclidean norm for vector."""
    return jnp.sum(x ** 2, axis=-1)

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute minus twice the dot-product between vectors."""
    return -2. * jnp.vdot(x, y)

  def h(self, z: jnp.ndarray) -> float:
    return jnp.sum(z ** 2)

  def h_legendre(self, z: jnp.ndarray) -> float:
    return 0.25 * jnp.sum(z ** 2)

  def barycenter(self, weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    """Output barycenter of vectors when using squared-Euclidean distance."""
    return jnp.average(xs, weights=weights, axis=0)


@jax.tree_util.register_pytree_node_class
class Cosine(CostFn):
  """Cosine distance cost function.

  Args:
    ridge: TODO.
  """

  def __init__(self, ridge: float = 1e-8):
    super().__init__()
    self._ridge = ridge

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Cosine distance between vectors, denominator regularized with ridge."""
    ridge = self._ridge
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    cosine_similarity = jnp.vdot(x, y) / (x_norm * y_norm + ridge)
    cosine_distance = 1.0 - cosine_similarity
    # similarity is in [-1, 1], clip because of numerical imprecisions
    return jnp.clip(cosine_distance, 0., 2.)

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    return jnp.ones((1, dim))


@jax.tree_util.register_pytree_node_class
class Bures(CostFn):
  """Bures distance between a pair of (mean, cov matrix) raveled as vectors.

  Args:
    dimension: Dimensionality of the data.
    kwargs: Keyword arguments for :func:`ott.math.matrix_square_root.sqrtm`.
  """

  def __init__(self, dimension: int, **kwargs: Any):
    super().__init__()
    self._dimension = dimension
    self._sqrtm_kw = kwargs

  def norm(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute norm of Gaussian, sq. 2-norm of mean + trace of covariance."""
    mean, cov = x_to_means_and_covs(x, self._dimension)
    norm = jnp.sum(mean ** 2, axis=-1)
    norm += jnp.trace(cov, axis1=-2, axis2=-1)
    return norm

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute - 2 x Bures dot-product."""
    mean_x, cov_x = x_to_means_and_covs(x, self._dimension)
    mean_y, cov_y = x_to_means_and_covs(y, self._dimension)
    mean_dot_prod = jnp.vdot(mean_x, mean_y)
    sq_x = matrix_square_root.sqrtm(cov_x, self._dimension, **self._sqrtm_kw)[0]
    sq_x_y_sq_x = jnp.matmul(sq_x, jnp.matmul(cov_y, sq_x))
    sq__sq_x_y_sq_x = matrix_square_root.sqrtm(
        sq_x_y_sq_x, self._dimension, **self._sqrtm_kw
    )[0]
    return -2 * (mean_dot_prod + jnp.trace(sq__sq_x_y_sq_x, axis1=-2, axis2=-1))

  def covariance_fixpoint_iter(
      self,
      covs: jnp.ndarray,
      lambdas: jnp.ndarray,
      rtol: float = 1e-2
  ) -> jnp.ndarray:
    """Iterate fix-point updates to compute barycenter of Gaussians."""

    @functools.partial(jax.vmap, in_axes=[None, 0, 0])
    def scale_covariances(
        cov_sqrt: jnp.ndarray, cov_i: jnp.ndarray, lambda_i: jnp.ndarray
    ) -> jnp.ndarray:
      """Iterate update needed to compute barycenter of covariances."""
      return lambda_i * matrix_square_root.sqrtm_only(
          (cov_sqrt @ cov_i) @ cov_sqrt
      )

    def cond_fn(iteration: int, constants: Tuple[...], state) -> bool:
      del iteration, constants
      _, diff = state
      return diff > rtol

    def body_fn(
        iteration: int, constants: Tuple[...], state: Tuple[jnp.ndarray, float],
        compute_error: bool
    ) -> Tuple[jnp.ndarray, float]:
      del iteration, constants, compute_error
      cov, _ = state
      cov_sqrt, cov_inv_sqrt, _ = matrix_square_root.sqrtm(cov)
      scaled_cov = jnp.linalg.matrix_power(
          jnp.sum(self.scale_covariances(cov_sqrt, covs, lambdas), axis=0), 2
      )
      next_cov = (cov_inv_sqrt @ scaled_cov) @ cov_inv_sqrt
      diff = jnp.sum((next_cov - cov) ** 2) / jnp.prod(jnp.array(cov.shape))
      return next_cov, diff

    def init_state() -> Tuple[jnp.ndarray, float]:
      cov_init = jnp.eye(self._dimension)
      diff = jnp.inf
      return cov_init, diff

    cov, _ = fixed_point_loop.fixpoint_iter(
        cond_fn=cond_fn,
        body_fn=body_fn,
        min_iterations=10,
        max_iterations=500,
        inner_iterations=1,
        constants=(),
        state=init_state()
    )
    return cov

  def barycenter(self, weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    """Compute the Bures barycenter of weighted Gaussian distributions.

    Implements the fixed point approach proposed in :cite:`alvarez-esteban:16`
    for the computation of the mean and the covariance of the barycenter of
    weighted Gaussian distributions.

    Args:
      weights: The barycentric weights.
      xs: The points to be used in the computation of the barycenter, where
        each point is described by a concatenation of the mean and the
        covariance (raveled).

    Returns:
      A concatenation of the mean and the raveled covariance of the barycenter.
    """
    # Ensure that barycentric weights sum to 1.
    weights = weights / jnp.sum(weights)
    mus, covs = x_to_means_and_covs(xs, self._dimension)
    mu_bary = jnp.sum(weights[:, None] * mus, axis=0)
    cov_bary = self.covariance_fixpoint_iter(covs=covs, lambdas=weights)
    barycenter = mean_and_cov_to_x(mu_bary, cov_bary, self._dimension)
    return barycenter

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    """Pad with concatenated zero means and \
      raveled identity covariance matrix."""
    dimension = int((-1 + math.sqrt(1 + 4 * dim)) / 2)
    padding = mean_and_cov_to_x(
        jnp.zeros((dimension,)), jnp.eye(dimension), dimension
    )
    return padding[jnp.newaxis, :]

  def tree_flatten(self):
    return (), (self._dimension, self._sqrtm_kw)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del children
    return cls(aux_data[0], **aux_data[1])


@jax.tree_util.register_pytree_node_class
class UnbalancedBures(CostFn):
  """Regularized/unbalanced Bures dist between two triplets of (mass,mean,cov).

  This cost implements the value defined in :cite:`janati:20`, eq. 37, 39, 40.
  We follow their notations. It is assumed inputs are given as
  triplets (mass, mean, covariance) raveled as vectors, in that order.

  Args:
    dimension: TODO.
    gamma: TODO.
    sigma: TODO.
    kwargs: TODO.
  """

  def __init__(
      self,
      dimension: int,
      gamma: float = 1.0,
      sigma: float = 1.0,
      **kwargs: Any,
  ):
    super().__init__()
    self._dimension = dimension
    self._gamma = gamma
    self._sigma = sigma
    self._sqrtm_kw = kwargs

  def norm(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute norm of Gaussian for unbalanced Bures."""
    return self._gamma * x[0]

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute dot-product for unbalanced Bures."""
    # Sets a few constants
    gam = self._gamma
    sig2 = self._sigma ** 2
    lam = sig2 + gam / 2
    tau = gam / (2 * lam)

    # Extracts mass, mean vector, covariance matrices
    mass_x, mass_y = x[0], y[0]
    mean_x, cov_x = x_to_means_and_covs(x[1:], self._dimension)
    mean_y, cov_y = x_to_means_and_covs(y[1:], self._dimension)

    diff_means = mean_x - mean_y

    # Identity matrix of suitable size
    iden = jnp.eye(self._dimension, dtype=x.dtype)

    # Creates matrices needed in the computation
    tilde_a = 0.5 * gam * (iden - lam * jnp.linalg.inv(cov_x + lam * iden))
    tilde_b = 0.5 * gam * (iden - lam * jnp.linalg.inv(cov_y + lam * iden))

    tilde_a_b = jnp.matmul(tilde_a, tilde_b)
    c_mat = matrix_square_root.sqrtm(
        1 / tau * tilde_a_b + 0.25 * (sig2 ** 2) * iden, **self._sqrtm_kw
    )[0]
    c_mat -= 0.5 * sig2 * iden

    # Computes log determinants (their sign should be >0).
    sldet_c, ldet_c = jnp.linalg.slogdet(c_mat)
    sldet_t_ab, ldet_t_ab = jnp.linalg.slogdet(tilde_a_b)
    sldet_ab, ldet_ab = jnp.linalg.slogdet(jnp.matmul(cov_x, cov_y))
    sldet_c_ab, ldet_c_ab = jnp.linalg.slogdet(c_mat - 2 * tilde_a_b / gam)

    # Gathers all these results to compute log total mass of transport
    log_m_pi = (0.5 * self._dimension * sig2 / (gam + sig2)) * jnp.log(sig2)

    log_m_pi += (1 / (tau + 1)) * (
        jnp.log(mass_x) + jnp.log(mass_y) + ldet_c + 0.5 *
        (tau * ldet_t_ab - ldet_ab)
    )

    log_m_pi += -jnp.sum(
        diff_means * jnp.linalg.solve(cov_x + cov_y + lam * iden, diff_means)
    ) / (2 * (tau + 1))

    log_m_pi += -0.5 * ldet_c_ab

    # If all logdet signs are 1, output value, nan otherwise.
    # TODO(michalk8): use lax.cond
    return jnp.where(
        sldet_c == 1 and sldet_c_ab == 1 and sldet_ab == 1 and sldet_t_ab == 1,
        2 * sig2 * mass_x * mass_y - 2 * (sig2 + gam) * jnp.exp(log_m_pi),
        jnp.nan
    )

  def tree_flatten(self):
    return (), (self._dimension, self._gamma, self._sigma, self._sqrtm_kw)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del children
    dim, gamma, sigma, kwargs = aux_data
    return cls(dim, gamma=gamma, sigma=sigma, **kwargs)


def x_to_means_and_covs(x: jnp.ndarray,
                        dimension: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Extract means and covariance matrices of Gaussians from raveled vector.

  Args:
    x: [num_gaussians, dimension, (1 + dimension)] array of concatenated means
    and covariances (raveled) dimension: the dimension of the Gaussians.

  Returns:
    means: [num_gaussians, dimension] array that holds the means.
    covariances: [num_gaussians, dimension] array that holds the covariances.
  """
  x = jnp.atleast_2d(x)
  means = x[:, :dimension]
  covariances = jnp.reshape(
      x[:, dimension:dimension + dimension ** 2], (-1, dimension, dimension)
  )
  return jnp.squeeze(means), jnp.squeeze(covariances)


def mean_and_cov_to_x(
    mean: jnp.ndarray, covariance: jnp.ndarray, dimension: int
) -> jnp.ndarray:
  """Ravel a Gaussian's mean and covariance matrix to d(1 + d) vector."""
  x = jnp.concatenate((mean, jnp.reshape(covariance, (dimension * dimension))))
  return x
