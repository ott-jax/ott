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

"""Implements the sinkhorn divergence."""

import collections
from typing import Any, Dict, Mapping, Optional, Type

import jax
from jax import numpy as jnp
from ott.core import sinkhorn
from ott.core import segment
from ott.geometry import geometry
from ott.geometry import pointcloud

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
    share_epsilon: bool = True,
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
    share_epsilon: if True, enforces that the same epsilon regularizer is shared
      for all 2 or 3 terms of the Sinkhorn divergence. In that case, the epsilon
      will be by default that used when comparing x to y (contained in the first
      geometry). This flag is set to True by default, because in the default
      setting, the epsilon regularization is a function of the mean of the cost
      matrix.
    **kwargs: keywords arguments to the generic class. This is specific to each
      geometry.

  Returns:
    tuple: (sinkhorn divergence value, three pairs of potentials, three costs)
  """
  geometries = geom.prepare_divergences(*args, static_b=static_b, **kwargs)
  num_a, num_b = geometries[0].shape

  geometries = (geometries + (None,) * max(0, 3 - len(geometries)))[:3]
  if share_epsilon:
    geometries = (geometries[0],) + tuple(
            geom.copy_epsilon(geometries[0]) if geom is not None else None
            for geom in geometries[1 : (2 if static_b else 3)]
        )

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
    **kwargs: Arguments to sinkhorn.
  Returns:
    SinkhornDivergenceOutput named tuple.
  """
  # When computing a Sinkhorn divergence, the (x,y) terms and (x,x) / (y,y)
  # terms are computed independently. The user might want to pass some
  # sinkhorn_kwargs to parameterize sinkhorn's behavior, but those should
  # only apply to the (x,y) part. For the (x,x) / (y,y) part we fall back
  # on a simpler choice (parallel_dual_updates + momentum 0.5) that is known
  # to work well in such settings. In the future we might want to give some
  # freedom on setting parameters for the (x,x)/(y,y) part.
  # Since symmetric terms are computed assuming a = b, the linear systems
  # arising in implicit differentiation (if used) of the potentials computed for
  # the symmetric parts should be marked as symmetric.
  kwargs_symmetric = kwargs.copy()
  kwargs_symmetric.update(
      parallel_dual_updates=True,
      momentum=0.5,
      chg_momentum_from=0,
      anderson_acceleration=0,
      implicit_solver_symmetric=True)

  out_xy = sinkhorn.sinkhorn(geometry_xy, a, b, **kwargs)
  out_xx = sinkhorn.sinkhorn(geometry_xx, a, a, **kwargs_symmetric)
  if geometry_yy is None:
    out_yy = sinkhorn.SinkhornOutput(errors=jnp.array([]), reg_ot_cost=0)
  else:
    out_yy = sinkhorn.sinkhorn(geometry_yy, b, b, **kwargs_symmetric)

  div = (out_xy.reg_ot_cost - 0.5 * (out_xx.reg_ot_cost + out_yy.reg_ot_cost)
         + 0.5 * geometry_xy.epsilon * (jnp.sum(a) - jnp.sum(b))**2)
  out = (out_xy, out_xx, out_yy)
  return SinkhornDivergenceOutput(div, tuple([s.f, s.g] for s in out),
                                  (geometry_xy, geometry_xx, geometry_yy),
                                  tuple(s.errors for s in out),
                                  tuple(s.converged for s in out))


def segment_sinkhorn_divergence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    segment_ids_x: Optional[jnp.ndarray] = None,
    segment_ids_y: Optional[jnp.ndarray] = None,
    num_segments: Optional[int] = None,
    indices_are_sorted: Optional[bool] = None,
    num_per_segment_x: Optional[jnp.ndarray] = None,
    num_per_segment_y: Optional[jnp.ndarray] = None,
    weights_x: Optional[jnp.ndarray] = None,
    weights_y: Optional[jnp.ndarray] = None,
    sinkhorn_kwargs: Optional[Mapping[str, Any]] = None,
    static_b: bool = False,
    share_epsilon: bool = True,
    **kwargs) -> jnp.ndarray:
  """Computes Sinkhorn divergence between subsets of data with pointcloud.
  
  The second interface assumes `x` and `y` are segmented contiguously.

  In all cases, both `x` and `y` should contain the same number of segments.
  Each segment will be separately run through the sinkhorn divergence using
  array padding.
  
  Args:
    x: Array of input points, of shape [num_x, feature]. Multiple segments are
      held in this single array.
    y: Array of target points, of shape [num_y, feature].
    segment_ids_x: (1st interface) The segment ID for which each row of x
      belongs. This is a similar interface to `jax.ops.segment_sum`.
    segment_ids_y: (1st interface) The segment ID for which each row of y
      belongs.
    num_segments: (1st interface) Number of segments. This is required for JIT
      compilation to work. If not given, it will be computed from the data as
      the max segment ID.
    indices_are_sorted: (1st interface) Whether `segment_ids_x` and
      `segment_ids_y` are sorted. Default false.
    num_per_segment_x: (2nd interface) Number of points in each segment in `x`.
      For example, [100, 20, 30] would imply that `x` is segmented into three
      arrays of length `[100]`, `[20]`, and `[30]` respectively.
    num_per_segment_y: (2nd interface) Number of points in each segment in `y`.
    weights_x: Weights of each input points, arranged in the same segmented
      order as `x`.
    weights_y: Weights of each input points, arranged in the same segmented
      order as `y`.
    sinkhorn_kwargs: Optionally a dict containing the keywords arguments for
      calls to the `sinkhorn` function, that is called twice if static_b else
      three times.
    static_b: if True, divergence of measure b against itself is NOT computed
    share_epsilon: if True, enforces that the same epsilon regularizer is shared
      for all 2 or 3 terms of the Sinkhorn divergence. In that case, the epsilon
      will be by default that used when comparing x to y (contained in the first
      geometry). This flag is set to True by default, because in the default
      setting, the epsilon regularization is a function of the mean of the cost
      matrix.
    **kwargs: keywords arguments to the generic class. This is specific to each
      geometry.

  Returns:
    An array of sinkhorn divergence values for each segment.
  """
  use_segment_ids = segment_ids_x is not None
  if use_segment_ids:
    assert segment_ids_y is not None
  else:
    assert num_per_segment_x is not None
    assert num_per_segment_y is not None

  segmented_x, segmented_weights_x, num_segments_x = segment.segment_point_cloud(
    x, weights_x,
    segment_ids_x, num_segments, indices_are_sorted,
    num_per_segment_x)
  
  segmented_y, segmented_weights_y, num_segments_y = segment.segment_point_cloud(
    y, weights_y,
    segment_ids_y, num_segments, indices_are_sorted,
    num_per_segment_y)
  
  assert num_segments_x == num_segments_y

  def single_segment_sink_div(padded_x, padded_y, padded_weight_x,
                              padded_weight_y):
    return sinkhorn_divergence(
        pointcloud.PointCloud,
        padded_x,
        padded_y,
        a=padded_weight_x,
        b=padded_weight_y,
        sinkhorn_kwargs=sinkhorn_kwargs,
        static_b=static_b,
        share_epsilon=share_epsilon,
        **kwargs).divergence

  v_sink_div = jax.vmap(single_segment_sink_div, in_axes=[0, 0, 0, 0])

  segmented_divergences = v_sink_div(segmented_x, segmented_y,
                                     segmented_weights_x, segmented_weights_y)
  return segmented_divergences
