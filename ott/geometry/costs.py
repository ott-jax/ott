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

# Lint as: python3
"""Several cost/norm functions for relevant vector types."""
import abc

import jax
import jax.numpy as jnp
from ott.geometry import matrix_square_root


@jax.tree_util.register_pytree_node_class
class CostFn(abc.ABC):
  """A generic cost function, taking two vectors as input.

  Cost functions evaluate a function on a pair of inputs. For convenience,
  that function is split into two norms -- evaluated on each input separately --
  followed by a pairwise cost that involves both inputs, as in

  c(x,y) = norm(x) + norm(y) + pairwise(x,y)

  If the norm function is not implemented, that value is handled as a 0.
  """

  norm = None  #  no norm function created by default.

  @abc.abstractmethod
  def pairwise(self, x, y):
    pass

  def barycenter(self, weights, xs):
    pass

  def __call__(self, x, y):
    return self.pairwise(x, y) + (
        0 if self.norm is None else self.norm(x) + self.norm(y))  # pylint: disable=not-callable

  def all_pairs(self, x: jnp.ndarray, y: jnp.ndarray):
    """Computes matrix of all costs (including norms) for vectors in x / y.

    Args:
      x: [num_a, d] jnp.ndarray
      y: [num_b, d] jnp.ndarray
    Returns:
      [num_a, num_b] matrix of cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self(x_, y_))(y))(x)

  def all_pairs_pairwise(self, x: jnp.ndarray, y: jnp.ndarray):
    """Computes matrix of all pairwise-costs (no norms) for vectors in x / y.

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
class Euclidean(CostFn):
  """Squared Euclidean distance CostFn."""

  def norm(self, x):
    return jnp.sum(x ** 2, axis=-1)

  def pairwise(self, x, y):
    return -2 * jnp.vdot(x, y)
  
  def barycenter(self, weights, xs):
    return jnp.average(xs, weights=weights, axis=0)



@jax.tree_util.register_pytree_node_class
class Cosine(CostFn):
  """Cosine distance CostFn."""

  def __init__(self, ridge=1e-8):
    super().__init__()
    self._ridge = ridge

  def pairwise(self, x, y):
    ridge = self._ridge
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    cosine_similarity = jnp.vdot(x, y) / (x_norm * y_norm + ridge)
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance


@jax.tree_util.register_pytree_node_class
class Bures(CostFn):
  """Bures distance between a pair of (mean, cov matrix) raveled as vectors."""

  def __init__(self, dimension, **kwargs):
    super().__init__()
    self._dimension = dimension
    self._sqrtm_kw = kwargs

  def norm(self, x):
    norm = jnp.sum(x[..., 0:self._dimension]**2, axis=-1)
    x_mat = jnp.reshape(x[..., self._dimension:],
                        (-1, self._dimension, self._dimension))

    norm += jnp.trace(x_mat, axis1=-2, axis2=-1)
    return norm

  def pairwise(self, x, y):
    mean_dot_prod = jnp.vdot(x[0:self._dimension], y[0:self._dimension])
    x_mat = jnp.reshape(x[self._dimension:],
                        (self._dimension, self._dimension))
    y_mat = jnp.reshape(y[self._dimension:],
                        (self._dimension, self._dimension))

    sq_x = matrix_square_root.sqrtm(x_mat, self._dimension,
                                    **self._sqrtm_kw)[0]
    sq_x_y_sq_x = jnp.matmul(sq_x, jnp.matmul(y_mat, sq_x))
    sq__sq_x_y_sq_x = matrix_square_root.sqrtm(sq_x_y_sq_x, self._dimension,
                                               **self._sqrtm_kw)[0]
    return -2 * (
        mean_dot_prod + jnp.trace(sq__sq_x_y_sq_x, axis1=-2, axis2=-1))

  def tree_flatten(self):
    return (), (self._dimension, self._sqrtm_kw)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del children
    return cls(aux_data[0], **aux_data[1])


@jax.tree_util.register_pytree_node_class
class UnbalancedBures(CostFn):
  """Regularized, unbalanced Bures distance between triplets.

  This cost implements the value defined in https://arxiv.org/pdf/2006.02572.pdf
  Equation 37, 39, 40. We follow their notations. It is assumed inputs are given
  as triplets (mass, mean, covariance) raveled as vectors, in that order.
  """

  def __init__(self,
               dimension: int,
               gamma: float = 1,
               sigma: float = 1,
               **kwargs):
    super().__init__()
    self._dimension = dimension
    self._gamma = gamma
    self._sigma2 = sigma ** 2
    self._sqrtm_kw = kwargs

  def norm(self, x):
    return self._gamma * x[0]

  def pairwise(self, x, y):
    # Sets a few constants
    gam = self._gamma
    sig2 = self._sigma2
    lam = sig2 + gam / 2
    tau = gam / (2 * lam)

    # Extracts mass, mean vector, covariance matrices
    mass_x, mass_y = x[0], y[0]
    diff_means = x[1:1+self._dimension] - y[1:1+self._dimension]
    x_mat = jnp.reshape(x[1+self._dimension:],
                        (self._dimension, self._dimension))
    y_mat = jnp.reshape(y[1+self._dimension:],
                        (self._dimension, self._dimension))

    # Identity matrix of suitable size
    iden = jnp.eye(self._dimension, dtype=x.dtype)

    # Creates matrices needed in the computation
    tilde_a = 0.5 * gam * (iden - lam * jnp.linalg.inv(x_mat + lam * iden))
    tilde_b = 0.5 * gam * (iden - lam * jnp.linalg.inv(y_mat + lam * iden))

    tilde_a_b = jnp.matmul(tilde_a, tilde_b)
    c_mat = matrix_square_root.sqrtm(
        1 / tau * tilde_a_b + 0.25 * (sig2 ** 2) * iden, **self._sqrtm_kw)[0]
    c_mat -= 0.5 * sig2 * iden

    # Computes log determinants (their sign should be >0).
    sldet_c, ldet_c = jnp.linalg.slogdet(c_mat)
    sldet_t_ab, ldet_t_ab = jnp.linalg.slogdet(tilde_a_b)
    sldet_ab, ldet_ab = jnp.linalg.slogdet(jnp.matmul(x_mat, y_mat))
    sldet_c_ab, ldet_c_ab = jnp.linalg.slogdet(c_mat - 2 * tilde_a_b / gam)

    # Gathers all these results to compute log total mass of transport
    log_m_pi = (0.5 * self._dimension * sig2 / (gam + sig2)) * jnp.log(sig2)

    log_m_pi += (1 / (tau + 1)) * (jnp.log(mass_x) + jnp.log(mass_y) + ldet_c
                                   + 0.5 * (tau * ldet_t_ab - ldet_ab))

    log_m_pi += -jnp.sum(diff_means * jnp.linalg.solve(
        x_mat + y_mat + lam * iden, diff_means)) / (2 * (tau + 1))

    log_m_pi += - 0.5 * ldet_c_ab

    # If all logdet signs are 1, output value, nan otherwise.
    return jnp.where(
        sldet_c == 1 and sldet_c_ab == 1 and sldet_ab == 1 and sldet_t_ab == 1,
        2 * sig2 * mass_x * mass_y - 2 * (sig2 + gam) * jnp.exp(log_m_pi),
        jnp.nan)

  def tree_flatten(self):
    return (), (self._dimension, self._gamma, self._sigma2, self._sqrtm_kw)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del children
    return cls(aux_data[0], aux_data[1], aux_data[2], **aux_data[3])
