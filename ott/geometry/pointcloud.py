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
"""A geometry defined using 2 point clouds and a cost function between them."""
from typing import Optional

import jax
import jax.numpy as jnp
from ott.geometry import costs
from ott.geometry import geometry
from ott.geometry import geometry_lr
from ott.geometry import ops


@jax.tree_util.register_pytree_node_class
class PointCloud(geometry.Geometry):
  """Defines geometry for 2 pointclouds (possibly 1 vs itself) using CostFn."""

  def __init__(self,
               x: jnp.ndarray,
               y: Optional[jnp.ndarray] = None,
               cost_fn: Optional[costs.CostFn] = None,
               power: float = 2.0,
               online: bool = False,
               **kwargs):
    """Creates a geometry from two point clouds, using CostFn.

    Creates a geometry, specifying a cost function passed as CostFn type object.
    When the number of points is large, setting the `online` flag to `True`
    implies that cost and kernel matrices used to update potentials or scalings
    will be recomputed on the fly, rather than stored in memory. More precisely,
    when setting `online`, the cost function will be partially cached by storing
    norm values for each point in both point clouds, but the pairwise cost
    function evaluations won't be. The sum of norms + the pairwise cost term is
    raised to `power`.

    Args:
      x : n x d array of n d-dimensional vectors
      y : m x d array of m d-dimensional vectors
      cost_fn: a CostFn function between two points in dimension d.
      power: a power to raise (norm(x) + norm(y) + cost(x,y)) **
      online: whether to run the online version of the computation or not. The
        online computation is particularly useful for big point clouds such that
        their cost matrix does not fit in memory.
      **kwargs: other optional parameters to be passed on to superclass
      initializer, notably those related to epsilon regularization.
    """

    self._cost_fn = costs.Euclidean() if cost_fn is None else cost_fn
    self._axis_norm = 0 if callable(self._cost_fn.norm) else None

    self.x = x
    self.y = y if y is not None else x

    self.power = power
    self._online = online

    super().__init__(**kwargs)

  @property
  def _norm_x(self):
    if self._axis_norm == 0:
      return self._cost_fn.norm(self.x)
    elif self._axis_norm is None:
      return 0

  @property
  def _norm_y(self):
    if self._axis_norm == 0:
      return self._cost_fn.norm(self.y)
    elif self._axis_norm is None:
      return 0

  @property
  def cost_matrix(self):
    if self._online:
      return None
    cost_matrix = self._cost_fn.all_pairs_pairwise(self.x, self.y)
    if self._axis_norm is not None:
      cost_matrix += self._norm_x[:, jnp.newaxis] + self._norm_y[jnp.newaxis, :]
    return cost_matrix ** (0.5 * self.power)

  @property
  def kernel_matrix(self):
    if self._online:
      return None
    return jnp.exp(-self.cost_matrix / self.epsilon)

  @property
  def shape(self):
    return (self.x.shape[0] if self.x is not None else 0,
            self.y.shape[0] if self.y is not None else 0)

  @property
  def is_symmetric(self):
    return self.y is None or (jnp.all(self.x.shape == self.y.shape) and
                              jnp.all(self.x == self.y))

  @property
  def is_squared_euclidean(self):
    return isinstance(self._cost_fn, costs.Euclidean) and self.power == 2.0

  @property
  def is_online(self) -> bool:
    return self._online

  def apply_lse_kernel(self,
                       f: jnp.ndarray,
                       g: jnp.ndarray,
                       eps: float,
                       vec: jnp.ndarray = None,
                       axis: int = 0) -> jnp.ndarray:
    if not self._online:
      return super().apply_lse_kernel(f, g, eps, vec, axis)

    app = jax.vmap(
        _apply_lse_kernel_xy,
        in_axes=[
            None, 0, None, self._axis_norm, None, 0, None, None, None, None
        ])

    if axis == 0:
      h_res, h_sgn = app(self.x, self.y, self._norm_x, self._norm_y, f, g, eps,
                         vec, self._cost_fn, self.power)
      h_res = eps * h_res - jnp.where(jnp.isfinite(g), g, 0)
    if axis == 1:
      h_res, h_sgn = app(self.y, self.x, self._norm_y, self._norm_x, g, f, eps,
                         vec, self._cost_fn, self.power)
      h_res = eps * h_res - jnp.where(jnp.isfinite(f), f, 0)
    return h_res, h_sgn

  def apply_kernel(self,
                   scaling: jnp.ndarray,
                   eps: Optional[float] = None,
                   axis: int = 0):
    if eps is None:
      eps = self.epsilon

    if not self._online:
      return super().apply_kernel(scaling, eps, axis)

    app = jax.vmap(_apply_kernel_xy, in_axes=[
        None, 0, None, self._axis_norm, None, None, None, None])
    if axis == 0:
      return app(self.x, self.y, self._norm_x, self._norm_y, scaling, eps,
                 self._cost_fn, self.power)
    if axis == 1:
      return app(self.y, self.x, self._norm_y, self._norm_x, scaling, eps,
                 self._cost_fn, self.power)

  def transport_from_potentials(self, f, g):
    if not self._online:
      return super().transport_from_potentials(f, g)
    transport = jax.vmap(_transport_from_potentials_xy, in_axes=[
        None, 0, None, self._axis_norm, None, 0, None, None, None])
    return transport(self.y, self.x, self._norm_y, self._norm_x, g, f,
                     self.epsilon, self._cost_fn, self.power)

  def transport_from_scalings(self, u, v):
    if not self._online:
      return super().transport_from_scalings(u, v)
    transport = jax.vmap(_transport_from_scalings_xy, in_axes=[
        None, 0, None, self._axis_norm, None, 0, None, None, None])
    return transport(self.y, self.x, self._norm_y, self._norm_x, v, u,
                     self.epsilon, self._cost_fn, self.power)

  def apply_cost(self,
                 arr: jnp.ndarray,
                 axis: int = 0,
                 fn=None) -> jnp.ndarray:
    """Applies cost matrix to array (vector or matrix).

    This function applies the geometry's cost matrix, to perform either
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b] matrix resulting from the (optional) elementwise
    application of fn to each entry of the `cost_matrix`.

    Args:
      arr: jnp.ndarray [num_a or num_b, batch], vector that will be multiplied
        by the cost matrix.
      axis: standard cost matrix if axis=1, transpose if 0
      fn: function optionally applied to cost matrix element-wise, before the
        apply

    Returns:
      A jnp.ndarray, [num_b, batch] if axis=0 or [num_a, batch] if axis=1
    """
    if fn is None:
      return self.vec_apply_cost(arr, axis, fn=fn)
    # Switch to efficient computation for the squared euclidean case.
    return jnp.where(jnp.logical_and(self.is_squared_euclidean,
                                     geometry.is_affine(fn)),
                     self.vec_apply_cost(arr, axis, fn=fn),
                     self._apply_cost(arr, axis, fn=fn))

  def _apply_cost(self,
                  arr: jnp.ndarray,
                  axis: int = 0,
                  fn=None) -> jnp.ndarray:
    """See apply_cost."""
    if self._online:
      app = jax.vmap(_apply_cost_xy, in_axes=[
          None, 0, None, self._axis_norm, None, None, None, None])
      if axis == 0:
        return app(self.x, self.y, self._norm_x, self._norm_y, arr,
                   self._cost_fn, self.power, fn)
      if axis == 1:
        return app(self.y, self.x, self._norm_y, self._norm_x, arr,
                   self._cost_fn, self.power, fn)
    else:
      return super().apply_cost(arr, axis, fn)

  def vec_apply_cost(self,
                     arr: jnp.ndarray,
                     axis: int = 0,
                     fn=None) -> jnp.ndarray:
    """Applies the geometry's cost matrix in a vectorised way.

    This performs either:
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b]

    This function can be used when the cost matrix is squared euclidean
    and fn is a linear map.

    Args:
      arr: jnp.ndarray [num_a or num_b, p], vector that will be multiplied
        by the cost matrix.
      axis: standard cost matrix if axis=1, transport if 0
      fn: function optionally applied to cost matrix element-wise, before the
        apply

    Returns:
      A jnp.ndarray, [num_b, p] if axis=0 or [num_a, p] if axis=1
    """
    rank = len(arr.shape)
    x, y = (self.x, self.y) if axis == 0 else (self.y, self.x)
    nx, ny = jnp.array(self._norm_x), jnp.array(self._norm_y)
    nx, ny = (nx, ny) if axis == 0 else (ny, nx)

    applied_cost = jnp.dot(nx, arr).reshape(1, -1)
    applied_cost += ny.reshape(-1, 1) * jnp.sum(arr, axis=0).reshape(1, -1)
    cross_term = -2.0 * jnp.dot(y, jnp.dot(x.T, arr))
    applied_cost += cross_term[:, None] if rank == 1 else cross_term
    return fn(applied_cost) if fn else applied_cost

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates the geometries used for a divergence computation."""
    x, y = args
    couples = [(x, y), (x, x)] if static_b else [(x, y), (x, x), (y, y)]
    return tuple(cls(*xy, **kwargs) for xy in couples)

  def tree_flatten(self):
    return ((self.x, self.y, self._epsilon, self._cost_fn),
            {'online': self._online, 'power': self.power})
  # Passing self.power in aux_data to be able to condition on it.

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    eps, fn = children[2:]
    return cls(*children[:2], epsilon=eps, cost_fn=fn, **aux_data)

  def to_LRCGeometry(self, scale=1.0):
    """Convert sqEuc. PointCloud to LRCGeometry if useful, and rescale."""
    if self.is_squared_euclidean:
      (n, m), d = self.shape, self.x.shape[1]
      if n * m > (n + m) * d:  # here apply_cost using LRCGeometry preferable.
        cost_1 = jnp.concatenate(
            (jnp.sum(self.x ** 2, axis=1, keepdims=True),
             jnp.ones((self.shape[0], 1)),
             -jnp.sqrt(2) * self.x),
            axis=1)
        cost_2 = jnp.concatenate(
            (jnp.ones((self.shape[1], 1)),
             jnp.sum(self.y ** 2, axis=1, keepdims=True),
             jnp.sqrt(2) * self.y),
            axis=1)
        cost_1 *= jnp.sqrt(scale)
        cost_2 *= jnp.sqrt(scale)

        return geometry_lr.LRCGeometry(
            cost_1=cost_1,
            cost_2=cost_2,
            epsilon=self._epsilon_init,
            relative_epsilon=self._relative_epsilon,
            scale=self._scale,
            **self._kwargs)
      else:
        self.x *= jnp.sqrt(scale)
        self.y *= jnp.sqrt(scale)
        return self
    else:
      raise ValueError('Cannot turn non-sq-Euclidean geometry into low-rank')


def _apply_lse_kernel_xy(x, y, norm_x, norm_y, f, g, eps,
                         vec, cost_fn, cost_pow):
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow)
  return ops.logsumexp((f + g - c) / eps, b=vec, return_sign=True, axis=-1)


def _transport_from_potentials_xy(x, y, norm_x, norm_y, f, g, eps, cost_fn,
                                  cost_pow):
  return jnp.exp((f + g - _cost(x, y, norm_x, norm_y, cost_fn, cost_pow)) / eps)


def _apply_kernel_xy(x, y, norm_x, norm_y, vec, eps, cost_fn, cost_pow):
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow)
  return jnp.dot(jnp.exp(-c / eps), vec)


def _transport_from_scalings_xy(x, y, norm_x, norm_y, u, v, eps, cost_fn,
                                cost_pow):
  return jnp.exp(- _cost(x, y, norm_x, norm_y, cost_fn, cost_pow) / eps) * u * v


def _cost(x, y, norm_x, norm_y, cost_fn, cost_pow):
  one_line_pairwise = jax.vmap(cost_fn.pairwise, in_axes=[0, None])
  return (norm_x + norm_y + one_line_pairwise(x, y)) ** (0.5 * cost_pow)


def _apply_cost_xy(x, y, norm_x, norm_y, vec, cost_fn, cost_pow, fn=None):
  """Applies [num_b, num_a] fn(cost) matrix (or transpose) to vector.

  Applies [num_b, num_a] ([num_a, num_b] if axis=1 from `apply_cost`)
  fn(cost) matrix (or transpose) to vector.

  Args:
    x: jnp.ndarray [num_a, d], first pointcloud
    y: jnp.ndarray [num_b, d], second pointcloud
    norm_x: jnp.ndarray [num_a,], (squared) norm as defined in by cost_fn
    norm_y: jnp.ndarray [num_b,], (squared) norm as defined in by cost_fn
    vec: jnp.ndarray [num_a,] ([num_b,] if axis=1 from `apply_cost`) vector
    cost_fn: a CostFn function between two points in dimension d.
    cost_pow: a power to raise (norm(x) + norm(y) + cost(x,y)) **
    fn: function optionally applied to cost matrix element-wise, before the
     apply

  Returns:
    A jnp.ndarray corresponding to cost x vector
  """
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow)
  return jnp.dot(c, vec) if fn is None else jnp.dot(fn(c), vec)


