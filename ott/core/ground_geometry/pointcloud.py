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
"""A class describing common operations for the Euclidean geometry."""
import abc
from typing import Union

import jax
import jax.numpy as np
from ott.core.ground_geometry import epsilon_scheduler
from ott.core.ground_geometry import geometry


@jax.tree_util.register_pytree_node_class
class CostFn(abc.ABC):
  """A generic cost function."""

  @abc.abstractmethod
  def __call__(self, x, y):
    pass

  def all_pairs(self, x: np.ndarray, y: np.ndarray):
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self(x_, y_))(y))(x)

  def tree_flatten(self):
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class EuclideanCostFn(CostFn):
  """A cost function based on the euclidean distance."""

  def __init__(self, power: float = 2.0):
    super().__init__()
    self._power = power

  def __call__(self, x, y):
    nx = np.sum(x ** 2, axis=-1)
    ny = np.sum(y ** 2, axis=-1)
    return np.abs(nx + ny - 2 * np.dot(x, y)) ** (self._power / 2)

  def tree_flatten(self):
    return (self._power,), None


@jax.tree_util.register_pytree_node_class
class PointCloudGeometry(geometry.Geometry):
  """Implements power of euclidean distance as a ground metric."""

  def __init__(self,
               x: np.ndarray,
               y: np.ndarray = None,
               cost_fn=None,
               epsilon: Union[epsilon_scheduler.Epsilon, float] = 1e-2,
               online: bool = False,
               **kwargs):
    """The geometry between two point clouds.

    Args:
      x : n x d array of n d-dimensional vectors
      y : m x d array of m d-dimensional vectors
      cost_fn: a cost function between two points in dimension d.
      epsilon: the value of the regularization parameter.
      online: whether to run the online version of the computation or not. The
        online computation is particularly useful for big point clouds which
        cost matrix does not fit in memory.
      **kwargs: potential additional params to epsilon.
    """
    self.x = x
    self.y = x if y is None else y
    self._cost_fn = EuclideanCostFn() if cost_fn is None else cost_fn
    self._online = online

    super().__init__(epsilon=epsilon, **kwargs)

    # Functions for online computations.
    lse = jax.scipy.special.logsumexp

    def center(x, y, f, g, eps):
      return (f + g - self._cost_fn(x, y)) / eps

    self._apply_lse_kernel_xy = jax.vmap(
        lambda y, g, f, eps, vec: lse(
            center(self.x, y, f, g, eps), b=vec, return_sign=True),
        in_axes=[0, 0, None, None, None])

    self._apply_lse_kernel_yx = jax.vmap(
        lambda x, f, g, eps, vec: lse(
            center(self.y, x, f, g, eps), b=vec, return_sign=True),
        in_axes=[0, 0, None, None, None])

    self._transport_from_potentials_xy = jax.jit(jax.vmap(
        lambda x, f, g, eps: np.exp(center(self.y, x, f, g, eps)),
        in_axes=[0, 0, None, None]))

    self._apply_kernel_xy = jax.vmap(
        lambda y, vec, eps: np.dot(
            np.exp(-self._cost_fn(self.x, y) / eps), vec),
        in_axes=[0, None, None])

    self._apply_kernel_yx = jax.vmap(
        lambda x, vec, eps: np.dot(
            np.exp(-self._cost_fn(self.y, x) / eps), vec),
        in_axes=[0, None, None])

    self._transport_from_scalings_xy = jax.vmap(
        lambda x, u, v, eps: np.exp(-self._cost_fn(x, self.y) / eps) * u * v,
        in_axes=[0, 0, 0, None, None])

  @property
  def cost_matrix(self):
    return self._cost_fn.all_pairs(self.x, self.y)

  @property
  def kernel_matrix(self):
    return np.exp(-self.cost_matrix/self.epsilon)

  @property
  def shape(self):
    return self.x.shape[0], self.y.shape[0]

  @property
  def is_symmetric(self):
    return np.all(self.x.shape == self.y.shape) and np.all(self.x == self.y)

  def apply_lse_kernel(self,
                       f: np.ndarray,
                       g: np.ndarray,
                       eps: float,
                       vec: np.ndarray = None,
                       axis: int = 0) -> np.ndarray:
    if not self._online:
      return super().apply_lse_kernel(f, g, eps, vec, axis)

    if axis == 0:
      h_res, h_sgn = self._apply_lse_kernel_xy(self.y, g, f, eps, vec)
      h_res = eps * h_res - g
    if axis == 1:
      h_res, h_sgn = self._apply_lse_kernel_yx(self.x, f, g, eps, vec)
      h_res = eps * h_res - f
    return h_res, h_sgn

  def apply_kernel(self, scaling: np.ndarray, eps: float, axis=0):
    if not self._online:
      return super().apply_kernel(scaling, eps, axis)
    if axis == 0:
      return self._apply_kernel_xy(self.y, scaling, eps)
    if axis == 1:
      return self._apply_kernel_yx(self.x, scaling, eps)

  def transport_from_potentials(self, f, g):
    if not self._online:
      return super().transport_from_potentials(f, g)
    return self._transport_from_potentials_xy(self.x, f, g, self.epsilon)

  def transport_from_scalings(self, u, v):
    if not self._online:
      return super().transport_from_scalings(u, v)
    return self._transport_from_scalings_xy(self.x, u, v, self.epsilon)

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates the geometries used for a divergence computation."""
    x, y = args
    couples = [(x, y), (x, x)] if static_b else [(x, y), (x, x), (y, y)]
    return tuple(cls(*xy, **kwargs) for xy in couples)

  def tree_flatten(self):
    return ((self.x, self.y, self._cost_fn, self.epsilon),
            {'online': self._online})

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children[:3], epsilon=children[-1], **aux_data)
