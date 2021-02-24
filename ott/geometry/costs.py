# coding=utf-8
# Copyright 2021 Google LLC.
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
from jax.lib import xla_bridge
import jax.numpy as jnp
from ott.geometry import matrix_square_root


def dot(x: jnp.ndarray, y: jnp.ndarray):
  """Accelerator dependent dot. Implemented to avoid OOMs with online mode."""
  platform = xla_bridge.get_backend().platform
  return jnp.sum(x * y) if platform == 'gpu' else jnp.vdot(x, y)


@jax.tree_util.register_pytree_node_class
class CostFn(abc.ABC):
  """A generic cost function, taking two vectors as input.

  Cost functions evaluate a function on a pair of inputs. For convenience,
  that function is split into two norms -- evaluated on each input separately --
  followed by a "dot product" like quantity, as in

  c(x,y) = norm(x) + norm(y) - dotprod(x,y)

  If the norm function is not implemented, that value is handled as a 0.
  """

  norm = None  #  no norm function created by default.

  @abc.abstractmethod
  def dotprod(self, x, y):
    pass

  def __call__(self, x, y):
    return - self.dotprod(x, y) + (
        0 if self.norm is None else self.norm(x) + self.norm(y))  # pylint: disable=not-callable

  def all_pairs(self, x: jnp.ndarray, y: jnp.ndarray):
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self(x_, y_))(y))(x)

  def all_pairs_dotprod(self, x: jnp.ndarray, y: jnp.ndarray):
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self.dotprod(x_, y_))(y))(x)

  def tree_flatten(self):
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class Euclidean(CostFn):
  """Squared Euclidean distance cost function."""

  def norm(self, x):
    return jnp.sum(x ** 2, axis=-1)

  def dotprod(self, x, y):
    return 2 * dot(x, y)


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

  def dotprod(self, x, y):
    mean_dot_prod = dot(x[0:self._dimension], y[0:self._dimension])
    x_mat = jnp.reshape(x[self._dimension:],
                        (self._dimension, self._dimension))
    y_mat = jnp.reshape(y[self._dimension:],
                        (self._dimension, self._dimension))

    sq_x = matrix_square_root.sqrtm(x_mat, self._dimension,
                                    **self._sqrtm_kw)[0]
    sq_x_y_sq_x = jnp.matmul(sq_x, jnp.matmul(y_mat, sq_x))
    sq__sq_x_y_sq_x = matrix_square_root.sqrtm(sq_x_y_sq_x, self._dimension,
                                               **self._sqrtm_kw)[0]
    return 2 * (
        mean_dot_prod + jnp.trace(sq__sq_x_y_sq_x, axis1=-2, axis2=-1))

  def tree_flatten(self):
    return (), (self._dimension, self._sqrtm_kw)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del children
    return cls(aux_data[0], **aux_data[1])


