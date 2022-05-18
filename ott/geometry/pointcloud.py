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
from typing import Optional, Union

import math

import jax
import jax.numpy as jnp
from ott.geometry import costs
from ott.geometry import geometry
from ott.geometry import low_rank
from ott.geometry import ops


@jax.tree_util.register_pytree_node_class
class PointCloud(geometry.Geometry):
  """Defines geometry for 2 pointclouds (possibly 1 vs itself) using CostFn."""

  def __init__(self,
               x: jnp.ndarray,
               y: Optional[jnp.ndarray] = None,
               cost_fn: Optional[costs.CostFn] = None,
               power: float = 2.0,
               online: Union[bool, int] = False,
               scale_cost: Optional[Union[bool, float, str]] = None,
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
        their cost matrix does not fit in memory. This is done by batching
        :meth:`apply_lse_kernel`. If `True`, batch size of 1024 is used.
      scale_cost: option to rescale the cost matrix. Implemented scalings are
        'median', 'mean', 'max_cost', 'max_norm' and 'max_bound'.
        Alternatively, a float factor can be given to rescale the cost such
        that ``cost_matrix /= scale_cost``. If `True`, use 'mean'.
      **kwargs: other optional parameters to be passed on to superclass
      initializer, notably those related to epsilon regularization.
    """

    self._cost_fn = costs.Euclidean() if cost_fn is None else cost_fn
    self._axis_norm = 0 if callable(self._cost_fn.norm) else None

    self.x = x
    self.y = self.x if y is None else y

    if online is True:
      online = 1024
    if online:
      assert online > 0, f"`online={online}` must be positive."
      n, m = self.shape
      self._bs = min(
          online, online, *(() + ((n,) if n else ()) + ((m,) if m else ())))
      # use `floor` instead of `ceil` and handle the rest seperately
      self._x_nsplit = int(math.floor(n / self._bs))
      self._y_nsplit = int(math.floor(m / self._bs))
    else:
      self._bs = self._x_nsplit = self._y_nsplit = None

    self._online = online
    self.power = power
    super().__init__(**kwargs)
    self._scale_cost = "mean" if scale_cost is True else scale_cost

  @property
  def _norm_x(self):
    if self._axis_norm == 0:
      return self._cost_fn.norm(self.x)
    elif self._axis_norm is None:
      return jnp.zeros(self.x.shape[0])

  @property
  def _norm_y(self):
    if self._axis_norm == 0:
      return self._cost_fn.norm(self.y)
    elif self._axis_norm is None:
      return jnp.zeros(self.y.shape[0])

  @property
  def cost_matrix(self):
    if self._online:
      return None
    cost_matrix = self.compute_cost_matrix()
    return cost_matrix * self.scale_cost

  @property
  def kernel_matrix(self):
    if self._online:
      return None
    return jnp.exp(-self.cost_matrix / self.epsilon)

  @property
  def shape(self):
    # in the process of flattening/unflattening in vmap, `__init__`
    # can be called with dummy objects
    # we optionally access `shape` in order to get the batch size
    try:
      return (self.x.shape[0] if self.x is not None else 0,
              self.y.shape[0] if self.y is not None else 0)
    except AttributeError:
      return 0, 0

  @property
  def is_symmetric(self):
    return self.y is None or (jnp.all(self.x.shape == self.y.shape) and
                              jnp.all(self.x == self.y))

  @property
  def is_squared_euclidean(self):
    return isinstance(self._cost_fn, costs.Euclidean) and self.power == 2.0

  @property
  def is_online(self) -> Union[bool, int]:
    return self._online is not None and self._online # for backward compatibility

  @property
  def scale_cost(self):
    """Computes the factor to scale the cost matrix."""
    if isinstance(self._scale_cost, float):
      return 1.0 / self._scale_cost
    elif self._scale_cost == 'max_cost':
      if self.is_online:
        return self.compute_summary_online(self._scale_cost)
      else:
        return jax.lax.stop_gradient(1.0 / jnp.max(self.compute_cost_matrix()))
    elif self._scale_cost == 'mean':
      if self.is_online:
        return self.compute_summary_online(self._scale_cost)
      else:
        if isinstance(self.shape[0], int) and (self.shape[0] > 0):
          return jax.lax.stop_gradient(
              1.0 / jnp.mean(self.compute_cost_matrix()))
    elif self._scale_cost == 'median':
      if not self.is_online:
        return jax.lax.stop_gradient(
            1.0 / jnp.median(self.compute_cost_matrix()))
      else:
        raise NotImplementedError("""Using the median as scaling factor for
            the cost matrix with the online mode is not implemented.""")
    elif self._scale_cost == 'max_norm':
      if self._cost_fn.norm is not None:
        return jax.lax.stop_gradient(
            1.0 / jnp.maximum(self._cost_fn.norm(self.x).max(),
                              self._cost_fn.norm(self.y).max()))
      else:
        return 1.0
    elif self._scale_cost == 'max_bound':
      if self.is_squared_euclidean:
        x_argmax = jnp.argmax(self._norm_x)
        y_argmax = jnp.argmax(self._norm_y)
        max_bound = (self._norm_x[x_argmax] + self._norm_y[y_argmax]
                     + 2 * jnp.sqrt(
                         self._norm_x[x_argmax] * self._norm_y[y_argmax]))
        return jax.lax.stop_gradient(1.0 / max_bound)
      else:
        return 1.0
    elif isinstance(self._scale_cost, str):
      raise ValueError(f'Scaling {self._scale_cost} not implemented.')
    else:
      return 1.0

  def compute_cost_matrix(self):
    cost_matrix = self._cost_fn.all_pairs_pairwise(self.x, self.y)
    if self._axis_norm is not None:
      cost_matrix += self._norm_x[:, jnp.newaxis] + self._norm_y[jnp.newaxis, :]
    return cost_matrix ** (0.5 * self.power)

  def apply_lse_kernel(self,
                       f: jnp.ndarray,
                       g: jnp.ndarray,
                       eps: float,
                       vec: jnp.ndarray = None,
                       axis: int = 0) -> jnp.ndarray:
    def body0(carry, i: int):
      f, g, eps, vec = carry
      y = jax.lax.dynamic_slice(
          self.y, (i * self._bs, 0), (self._bs, self.y.shape[1]))
      g_ = jax.lax.dynamic_slice(g, (i * self._bs,), (self._bs,))
      if self._axis_norm is None:
        norm_y = self._norm_y
      else:
        norm_y = jax.lax.dynamic_slice(
            self._norm_y, (i * self._bs,), (self._bs,))
      h_res, h_sgn = app(self.x, y, self._norm_x, norm_y, f, g_, eps, vec,
                         self._cost_fn, self.power, self.scale_cost)
      return carry, (h_res, h_sgn)

    def body1(carry, i: int):
      f, g, eps, vec = carry
      x = jax.lax.dynamic_slice(
          self.x, (i * self._bs, 0), (self._bs, self.x.shape[1]))
      f_ = jax.lax.dynamic_slice(f, (i * self._bs,), (self._bs,))
      if self._axis_norm is None:
        norm_x = self._norm_x
      else:
        norm_x = jax.lax.dynamic_slice(
            self._norm_x, (i * self._bs,), (self._bs,))
      h_res, h_sgn = app(self.y, x, self._norm_y, norm_x, g, f_, eps, vec,
                         self._cost_fn, self.power, self.scale_cost)
      return carry, (h_res, h_sgn)

    def finalize(i: int):
      if axis == 0:
        norm_y = self._norm_y if self._axis_norm is None else self._norm_y[i:]
        return app(self.x, self.y[i:], self._norm_x, norm_y, f, g[i:], eps,
                   vec, self._cost_fn, self.power, self.scale_cost)
      norm_x = self._norm_x if self._axis_norm is None else self._norm_x[i:]
      return app(self.y, self.x[i:], self._norm_y, norm_x,
                 g, f[i:], eps, vec, self._cost_fn, self.power, self.scale_cost)

    if not self._online:
      return super().apply_lse_kernel(f, g, eps, vec, axis)

    app = jax.vmap(
        _apply_lse_kernel_xy,
        in_axes=[None, 0, None, self._axis_norm, None, 0, None, None, None,
                 None, None]
    )

    if axis == 0:
      fun = body0
      v, n = g, self._y_nsplit
    elif axis == 1:
      fun = body1
      v, n = f, self._x_nsplit
    else:
      raise ValueError(axis)

    _, (h_res, h_sign) = jax.lax.scan(
        fun, init=(f, g, eps, vec), xs=jnp.arange(n))
    h_res, h_sign = jnp.concatenate(h_res), jnp.concatenate(h_sign)
    h_res_rest, h_sign_rest = finalize(n * self._bs)
    h_res = jnp.concatenate([h_res, h_res_rest])
    h_sign = jnp.concatenate([h_sign, h_sign_rest])

    return eps * h_res - jnp.where(jnp.isfinite(v), v, 0), h_sign

  def apply_kernel(self,
                   scaling: jnp.ndarray,
                   eps: Optional[float] = None,
                   axis: int = 0):
    if eps is None:
      eps = self.epsilon

    if not self._online:
      return super().apply_kernel(scaling, eps, axis)

    app = jax.vmap(_apply_kernel_xy, in_axes=[
        None, 0, None, self._axis_norm, None, None, None, None, None])
    if axis == 0:
      return app(self.x, self.y, self._norm_x, self._norm_y, scaling, eps,
                 self._cost_fn, self.power, self.scale_cost)
    if axis == 1:
      return app(self.y, self.x, self._norm_y, self._norm_x, scaling, eps,
                 self._cost_fn, self.power, self.scale_cost)

  def transport_from_potentials(self, f, g):
    if not self._online:
      return super().transport_from_potentials(f, g)
    transport = jax.vmap(_transport_from_potentials_xy, in_axes=[
        None, 0, None, self._axis_norm, None, 0, None, None, None, None])
    return transport(self.y, self.x, self._norm_y, self._norm_x, g, f,
                     self.epsilon, self._cost_fn, self.power, self.scale_cost)

  def transport_from_scalings(self, u, v):
    if not self._online:
      return super().transport_from_scalings(u, v)
    transport = jax.vmap(_transport_from_scalings_xy, in_axes=[
        None, 0, None, self._axis_norm, None, 0, None, None, None, None])
    return transport(self.y, self.x, self._norm_y, self._norm_x, v, u,
                     self.epsilon, self._cost_fn, self.power, self.scale_cost)

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
      return self._apply_cost(arr, axis, fn=fn)
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
          None, 0, None, self._axis_norm, None, None, None, None, None])
      if axis == 0:
        return app(self.x, self.y, self._norm_x, self._norm_y, arr,
                   self._cost_fn, self.power, self.scale_cost, fn)
      if axis == 1:
        return app(self.y, self.x, self._norm_y, self._norm_x, arr,
                   self._cost_fn, self.power, self.scale_cost, fn)
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
    return (fn(applied_cost) * self.scale_cost if fn
            else applied_cost * self.scale_cost)

  def leading_slice(self, t: jnp.ndarray, i: int) -> jnp.ndarray:
    start_indices = [i * self._bs] + (t.ndim - 1) * [0]
    slice_sizes = [self._bs] + list(t.shape[1:])
    return jax.lax.dynamic_slice(t, start_indices, slice_sizes)

  def compute_summary_online(self, summary: str) -> float:
    """Compute mean or max of cost matrix online, i.e. without instantiating it.

    Args:
      summary: str, can be 'mean' or 'max_cost'

    Returns:
      summary statistics
    """
    scale_cost = 1.0

    def body0(carry, i: int):
      vec, = carry
      y = self.leading_slice(self.y, i)
      if self._axis_norm is None:
        norm_y = self._norm_y
      else:
        norm_y = self.leading_slice(self._norm_y, i)
      h_res = app(
          self.x, y, self._norm_x, norm_y, vec,
          self._cost_fn, self.power, scale_cost)
      return carry, h_res

    def body1(carry, i: int):
      vec, = carry
      x = self.leading_slice(self.x, i)
      if self._axis_norm is None:
        norm_x = self._norm_x
      else:
        norm_x = self.leading_slice(self._norm_x, i)
      h_res = app(
          self.y, x, self._norm_y, norm_x, vec,
          self._cost_fn, self.power, scale_cost)
      return carry, h_res

    def finalize(i: int):
      if batch_for_y:
        norm_y = self._norm_y if self._axis_norm is None else self._norm_y[i:]
        return app(
            self.x, self.y[i:], self._norm_x, norm_y, vec,
            self._cost_fn, self.power, scale_cost)
      norm_x = self._norm_x if self._axis_norm is None else self._norm_x[i:]
      return app(
          self.y, self.x[i:], self._norm_y, norm_x, vec,
          self._cost_fn, self.power, scale_cost)

    if summary == 'mean':
      fn = _apply_cost_xy
    elif summary == 'max_cost':
      fn = _apply_max_xy
    else:
      raise ValueError(
          f'Scaling method {summary} does not exist for online mode.')
    app = jax.vmap(
        fn,
        in_axes=[None, 0, None, self._axis_norm, None, None, None, None]
    )

    batch_for_y = self.shape[0] < self.shape[1]
    if batch_for_y:
      fun = body0
      n = self._y_nsplit
      vec = jnp.ones(self.shape[0]) / (self.shape[1] * self.shape[0])
    else:
      fun = body1
      n = self._x_nsplit
      vec = jnp.ones(self.shape[1]) / (self.shape[1] * self.shape[0])

    _, val = jax.lax.scan(
        fun, init=(vec,), xs=jnp.arange(n))
    val = jnp.concatenate(val).squeeze()
    val_rest = finalize(n * self._bs)
    val_res = jnp.concatenate([val, val_rest])

    if summary == 'mean':
      return 1.0 / jnp.sum(val_res)
    elif summary == 'max_cost':
      return 1.0 / jnp.max(val_res)
    else:
      raise ValueError(
          f'Scaling method {summary} does not exist for online mode.')

  def barycenter(self, weights):
    """Compute barycenter of points in self.x using weights, valid for p=2.0 """
    assert self.power == 2.0
    return self._cost_fn.barycenter(self.x, weights)

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates the geometries used for a divergence computation."""
    x, y = args
    couples = [(x, y), (x, x)] if static_b else [(x, y), (x, x), (y, y)]
    return tuple(cls(*xy, **kwargs) for xy in couples)

  def tree_flatten(self):
    return ((self.x, self.y, self._epsilon, self._cost_fn),
            {'online': self._online, 'power': self.power,
             'scale_cost': self._scale_cost})
  # Passing self.power in aux_data to be able to condition on it.

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    eps, fn = children[2:]
    return cls(*children[:2], epsilon=eps, cost_fn=fn, **aux_data)

  def to_LRCGeometry(self, scale=1.0):
    """Converts sqEuc. PointCloud to LRCGeometry if useful, and rescale."""
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

        return low_rank.LRCGeometry(
            cost_1=cost_1,
            cost_2=cost_2,
            epsilon=self._epsilon_init,
            relative_epsilon=self._relative_epsilon,
            scale=self._scale_epsilon,
            scale_cost=self._scale_cost,
            **self._kwargs)
      else:
        self.x *= jnp.sqrt(scale)
        self.y *= jnp.sqrt(scale)
        return self
    else:
      raise ValueError('Cannot turn non-sq-Euclidean geometry into low-rank')


def _apply_lse_kernel_xy(
    x, y, norm_x, norm_y, f, g, eps, vec, cost_fn, cost_pow, scale_cost):
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return ops.logsumexp((f + g - c) / eps, b=vec, return_sign=True, axis=-1)


def _transport_from_potentials_xy(
    x, y, norm_x, norm_y, f, g, eps, cost_fn, cost_pow, scale_cost):
  return jnp.exp((
      f + g - _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)) / eps)


def _apply_kernel_xy(
    x, y, norm_x, norm_y, vec, eps, cost_fn, cost_pow, scale_cost):
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return jnp.dot(jnp.exp(-c / eps), vec)


def _transport_from_scalings_xy(
    x, y, norm_x, norm_y, u, v, eps, cost_fn, cost_pow, scale_cost):
  return jnp.exp(- _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
                 * scale_cost / eps) * u * v


def _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost):
  one_line_pairwise = jax.vmap(cost_fn.pairwise, in_axes=[0, None])
  return ((norm_x + norm_y + one_line_pairwise(x, y)) ** (0.5 * cost_pow)
          * scale_cost)


def _apply_cost_xy(
    x, y, norm_x, norm_y, vec, cost_fn, cost_pow, scale_cost, fn=None):
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
    scale_cost: scaling factor of the cost matrix.
    fn: function optionally applied to cost matrix element-wise, before the
      apply.

  Returns:
    A jnp.ndarray corresponding to cost x vector
  """
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return jnp.dot(c, vec) if fn is None else jnp.dot(fn(c), vec)


def _apply_max_xy(
    x, y, norm_x, norm_y, vec, cost_fn, cost_pow, scale_cost):
  del vec
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow, scale_cost)
  return jnp.max(jnp.abs(c))
