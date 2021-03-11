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
"""A geometry defined using 2 point clouds and a cost function between them."""
from typing import Optional, Union

import jax
import jax.numpy as jnp
from ott.geometry import costs
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry


@jax.tree_util.register_pytree_node_class
class PointCloud(geometry.Geometry):
  """Defines geometry for 2 pointclouds (possibly 1 vs itself) using CostFn."""

  def __init__(self,
               x: jnp.ndarray,
               y: Optional[jnp.ndarray] = None,
               epsilon: Union[epsilon_scheduler.Epsilon, float] = 1e-2,
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
      epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
      cost_fn: a CostFn function between two points in dimension d.
      power: a power to raise (norm(x) + norm(y) + cost(x,y)) **
      online: whether to run the online version of the computation or not. The
        online computation is particularly useful for big point clouds such that
        their cost matrix does not fit in memory.
      **kwargs: optional parameters to be passed on to epsilon scheduler.
    """
    super().__init__(epsilon=epsilon, **kwargs)

    self._cost_fn = costs.Euclidean() if cost_fn is None else cost_fn
    self._axis_norm = 0 if callable(self._cost_fn.norm) else None

    self.x = x
    self.y = y if y is not None else x

    self.power = power
    self._online = online

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
    return cost_matrix

  @property
  def kernel_matrix(self):
    if self._online:
      return None
    return jnp.exp(-self.cost_matrix / self.epsilon)

  @property
  def shape(self):
    return self.x.shape[0], self.y.shape[0]

  @property
  def is_symmetric(self):
    return self._y is None or (jnp.all(self.x.shape == self.y.shape) and
                               jnp.all(self.x == self.y))

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

  def apply_kernel(self, scaling: jnp.ndarray, eps: float, axis=0):
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

  # TODO(lpapaxanthos,cuturi) define an apply Cost function

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

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates the geometries used for a divergence computation."""
    x, y = args
    couples = [(x, y), (x, x)] if static_b else [(x, y), (x, x), (y, y)]
    return tuple(cls(*xy, **kwargs) for xy in couples)

  def tree_flatten(self):
    return ((self.x, self.y, self.epsilon, self._cost_fn, self.power),
            {'online': self._online})

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    eps, fn, power = children[2:]
    return cls(*children[:2], epsilon=eps, cost_fn=fn, power=power, **aux_data)


def _apply_lse_kernel_xy(x, y, norm_x, norm_y, f, g, eps,
                         vec, cost_fn, cost_pow):
  c = _cost(x, y, norm_x, norm_y, cost_fn, cost_pow)
  return jax.scipy.special.logsumexp((f + g - c) / eps, b=vec, return_sign=True,
                                     axis=-1)


def _transport_from_potentials_xy(
    x, y, norm_x, norm_y, f, g, eps, cost_fn, cost_pow):
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
