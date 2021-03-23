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

"""Implements the sinkhorn divergence."""
import collections
from typing import Optional, Type, Dict, Any

from jax import numpy as jnp
from ott.core import sinkhorn
from ott.geometry import geometry

SinkhornDivergenceOutput = collections.namedtuple(
    'SinkhornDivergenceOutput',
    ['divergence', 'potentials', 'geoms', 'errors', 'converged'])


def sinkhorn_divergence(
    geom: Type[geometry.Geometry],
    *args,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    static_b: bool = False,
    **kwargs):
  """Computes Sinkhorn divergence defined by a geometry, weights, parameters.

  Args:
    geom: a geometry class.
    *args: arguments to the prepare_divergences method that is specific to each
      geometry.
    a: jnp.ndarray<float>[n]: the weight of each input point. The sum of
      all elements of b must match that of a to converge.
    b: jnp.ndarray<float>[m]: the weight of each target point. The sum of
      all elements of b must match that of a to converge.
    sinkhorn_kwargs: Optionally a dict containing the keywords arguments for
      calls to the `sinkhorn` function, that is called twice if static_b else
      three times.
    static_b: if True, divergence of measure b against itself is NOT computed
    **kwargs: keywords arguments to the generic class. This is specific to each
      geometry.

  Returns:
    tuple: (sinkhorn divergence value, three pairs of potentials, three costs)
  """
  geometries = geom.prepare_divergences(*args, static_b=static_b, **kwargs)
  geometries = (geometries + (None,) * max(0, 3 - len(geometries)))[:3]

  num_a, num_b = geometries[0].shape
  a = jnp.ones((num_a,)) / num_a if a is None else a
  b = jnp.ones((num_b,)) / num_b if b is None else b
  div_kwargs = {} if sinkhorn_kwargs is None else sinkhorn_kwargs
  return _sinkhorn_divergence(*geometries, a, b, **div_kwargs)


def _sinkhorn_divergence(
    geometry_xy: geometry.Geometry,
    geometry_xx: geometry.Geometry,
    geometry_yy: Optional[geometry.Geometry],
    a: jnp.ndarray,
    b: jnp.ndarray,
    **kwargs):
  """Computes the (unbalanced) sinkhorn divergence for the wrapper function.

    This definition includes a correction depending on the total masses of each
    measure, as defined in https://arxiv.org/pdf/1910.12958.pdf (15).

  Args:
    geometry_xy: a Cost object able to apply kernels with a certain epsilon,
    between the views X and Y.
    geometry_xx: a Cost object able to apply kernels with a certain epsilon,
    between elements of the view X.
    geometry_yy: a Cost object able to apply kernels with a certain epsilon,
    between elements of the view Y.
    a: jnp.ndarray<float>[n]: the weight of each input point. The sum of
     all elements of b must match that of a to converge.
    b: jnp.ndarray<float>[m]: the weight of each target point. The sum of
     all elements of b must match that of a to converge.
    **kwargs: Arguments to sinkhorn_iterations.
  Returns:
    SinkhornDivergenceOutput named tuple.
  """
  geoms = (geometry_xy, geometry_xx, geometry_yy)
  out = [
      sinkhorn.SinkhornOutput(None, None, 0, None, None) if geom is None
      else sinkhorn.sinkhorn(geom, marginals[0], marginals[1], **kwargs)
      for (geom, marginals) in zip(geoms, [[a, b], [a, a], [b, b]])
  ]
  div = (out[0].reg_ot_cost - 0.5 * (out[1].reg_ot_cost + out[2].reg_ot_cost)
         + 0.5 * geometry_xy.epsilon * (jnp.sum(a) - jnp.sum(b))**2)
  return SinkhornDivergenceOutput(div, tuple([s.f, s.g] for s in out), geoms,
                                  tuple(s.errors for s in out),
                                  tuple(s.converged for s in out))
