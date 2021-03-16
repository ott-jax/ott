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

"""Some utility functions for transport computation."""

import jax.numpy as jnp
from ott.core import sinkhorn
from ott.geometry import geometry
from ott.geometry import pointcloud


class Transport:
  """An interface to transport problems.

  Attributes:
    geom: the ground geometry underlying the regularized transport problem.
    a: jnp.ndarray<float> the weights of the source.
    b: jnp.ndarray<float> the weights of the target.
    reg_ot_cost: if defined the regularized transport cost.
    matrix: the transport matrix (if the geometry allows its computation).
  """

  def __init__(self, *args, a=None, b=None, **kwargs):
    """Initialization.

    Args:
      *args: can be either a single argument, the geometry.Geometry instance, or
        for convenience only two jnp.ndarray<float> corresponding to two point
        clouds. In that case the regularization parameter epsilon must be set in
        the kwargs.
      a: the weights of the source.
      b: the weights of the target.
      **kwargs: the keyword arguments passed to the sinkhorn algorithm. If the
        first argument is made of two arrays, kwargs must contain epsilon.

    Raises:
      A ValueError in the case the Geometry cannot be defined by the input
      parameters.
    """
    if len(args) == 1:
      if not isinstance(args[0], geometry.Geometry):
        raise ValueError('A transport problem must be defined by either a '
                         'single geometry, or two arrays.')
      self.geom = args[0]
    else:
      pc_kw = {}
      for key in ['epsilon', 'cost_fn', 'power', 'online']:
        value = kwargs.pop(key, None)
        if value is not None:
          pc_kw[key] = value
      self.geom = pointcloud.PointCloud(*args, **pc_kw)

    num_a, num_b = self.geom.shape
    self.a = jnp.ones((num_a,)) / num_a if a is None else a
    self.b = jnp.ones((num_b,)) / num_b if b is None else b
    self._f = None
    self._g = None
    self._kwargs = kwargs
    self.reg_ot_cost = None
    self.solve()

  def solve(self):
    """Runs the sinkhorn algorithm to solve the transport problem."""
    out = sinkhorn.sinkhorn(self.geom, self.a, self.b, **self._kwargs)
    # TODO(oliviert): figure out how to warn the user if no convergence.
    # So far we always set the values, even if not converged.
    # TODO(oliviert, cuturi): handles cases where it has not converged.
    self._f = out.f
    self._g = out.g
    self.reg_ot_cost = out.reg_ot_cost

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix if it can be instantiated."""
    try:
      return self.geom.transport_from_potentials(self._f, self._g)
    except ValueError:
      u = self.geom.scaling_from_potential(self._f)
      v = self.geom.scaling_from_potential(self._g)
      return self.geom.transport_from_scalings(u, v)

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Applies the transport to a ndarray; axis=1 for its transpose."""
    try:
      return self.geom.apply_transport_from_potentials(
          self._f, self._g, inputs, axis=axis)
    except ValueError:
      u = self.geom.scaling_from_potential(self._f)
      v = self.geom.scaling_from_potential(self._g)
      return self.geom.apply_transport_from_scalings(u, v, inputs, axis=axis)
