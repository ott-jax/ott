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
from types import MappingProxyType
from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, Type

from jax import numpy as jnp

from ott.core import segment, sinkhorn
from ott.geometry import costs, geometry, pointcloud


class SinkhornDivergenceOutput(NamedTuple):
  divergence: float
  potentials: Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]
  geoms: Tuple[geometry.Geometry, geometry.Geometry, geometry.Geometry]
  errors: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray],
                Optional[jnp.ndarray]]
  converged: Tuple[bool, bool, bool]


def sinkhorn_divergence(
    geom: Type[geometry.Geometry],
    *args: Any,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    static_b: bool = False,
    share_epsilon: bool = True,
    **kwargs: Any,
) -> SinkhornDivergenceOutput:
  """Compute Sinkhorn divergence defined by a geometry, weights, parameters.

  Args:
    geom: Type of the geometry.
    args: Positional arguments to
      :meth:`~ott.geometry.geometry.Geometry.prepare_divergences` that is
      specific to each geometry.
    a: the weight of each input point. The sum of all elements of `a` must
      match that of `b` to converge.
    b: the weight of each target point. The sum of all elements of `b` must
      match that of `a` to converge.
    sinkhorn_kwargs: keywords arguments for :func:`~ott.core.sinkhorn.sinkhorn`
      that is called twice if ``static_b = True`` else 3 times.
    static_b: if True, divergence of measure `b` against itself is **not**
      computed.
    share_epsilon: if True, enforces that the same epsilon regularizer is shared
      for all 2 or 3 terms of the Sinkhorn divergence. In that case, the epsilon
      will be by default that used when comparing x to y (contained in the first
      geometry). This flag is set to True by default, because in the default
      setting, the epsilon regularization is a function of the mean of the cost
      matrix.
    kwargs: keywords arguments to the generic class. This is specific to each
      geometry.

  Returns:
    Sinkhorn divergence value, three pairs of potentials, three costs.
  """
  geometries = geom.prepare_divergences(*args, static_b=static_b, **kwargs)
  num_a, num_b = geometries[0].shape

  geometries = (geometries + (None,) * max(0, 3 - len(geometries)))[:3]
  if share_epsilon:
    geometries = (geometries[0],) + tuple(
        geom.copy_epsilon(geometries[0]) if geom is not None else None
        for geom in geometries[1:(2 if static_b else 3)]
    )

  a = jnp.ones((num_a,)) / num_a if a is None else a
  b = jnp.ones((num_b,)) / num_b if b is None else b
  return _sinkhorn_divergence(*geometries, a, b, **sinkhorn_kwargs)


def _sinkhorn_divergence(
    geometry_xy: geometry.Geometry,
    geometry_xx: geometry.Geometry,
    geometry_yy: Optional[geometry.Geometry],
    a: jnp.ndarray,
    b: jnp.ndarray,
    **kwargs: Any,
) -> SinkhornDivergenceOutput:
  """Compute the (unbalanced) sinkhorn divergence for the wrapper function.

    This definition includes a correction depending on the total masses of each
    measure, as defined in :sejourne:19:, eq. 15.

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
    kwargs: Keyword arguments to :func:`ott.core.sinkhorn.sinkhorn`.

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
      implicit_solver_symmetric=True
  )

  out_xy = sinkhorn.sinkhorn(geometry_xy, a, b, **kwargs)
  out_xx = sinkhorn.sinkhorn(geometry_xx, a, a, **kwargs_symmetric)
  if geometry_yy is None:
    out_yy = sinkhorn.SinkhornOutput(errors=jnp.array([]), reg_ot_cost=0)
  else:
    out_yy = sinkhorn.sinkhorn(geometry_yy, b, b, **kwargs_symmetric)

  div = (
      out_xy.reg_ot_cost - 0.5 * (out_xx.reg_ot_cost + out_yy.reg_ot_cost) +
      0.5 * geometry_xy.epsilon * (jnp.sum(a) - jnp.sum(b)) ** 2
  )
  out = (out_xy, out_xx, out_yy)
  return SinkhornDivergenceOutput(
      div, tuple([s.f, s.g] for s in out),
      (geometry_xy, geometry_xx, geometry_yy), tuple(s.errors for s in out),
      tuple(s.converged for s in out)
  )


def segment_sinkhorn_divergence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    num_segments: Optional[int] = None,
    max_measure_size: Optional[int] = None,
    cost_fn: Optional[costs.CostFn] = None,
    segment_ids_x: Optional[jnp.ndarray] = None,
    segment_ids_y: Optional[jnp.ndarray] = None,
    indices_are_sorted: Optional[bool] = None,
    num_per_segment_x: Optional[jnp.ndarray] = None,
    num_per_segment_y: Optional[jnp.ndarray] = None,
    weights_x: Optional[jnp.ndarray] = None,
    weights_y: Optional[jnp.ndarray] = None,
    sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    static_b: bool = False,
    share_epsilon: bool = True,
    **kwargs: Any
) -> jnp.ndarray:
  """Compute sinkhorn divergence between subsets of vectors given in `x` & `y`.

  Helper function designed to compute Sinkhorn divergences between several point
  clouds of varying size, in parallel, using padding for efficiency.
  In practice, The inputs `x` and `y` (and their weight vectors `weights_x` and
  `weights_y`) are assumed to be large weighted point clouds, that describe
  points taken from multiple measures. To extract several subsets of points, we
  provide two interfaces. The first interface assumes that a vector of id's is
  passed, describing for each point of `x` (resp. `y`) to which measure the
  point belongs to. The second interface assumes that `x` and `y` were simply
  formed by concatenating several measures contiguously, and that only indices
  that segment these groups are needed to recover them.

  For both interfaces, both `x` and `y` should contain the same total number of
  segments. Each segment will be padded as necessary, all segments rearranged as
  a tensor, and `vmap` used to evaluate sinkhorn divergences in parallel.

  Args:
    x: Array of input points, of shape [num_x, feature]. Multiple segments are
      held in this single array.
    y: Array of target points, of shape [num_y, feature].
    num_segments: Number of segments contained in x and y. Providing this number
      is required for JIT compilation to work, see also
      :func:`~ott.core.segment.segment_point_cloud`.
    max_measure_size: Total size of measures after padding. Should ideally be
      set to an upper bound on points clouds processed with the segment
      interface. Should also be smaller than total length of `x` or `y`.
      Providing this number is required for JIT compilation to work.
    cost_fn: Cost function, defaults to :class:`~ott.core.costs.Euclidean`.
    segment_ids_x: **1st interface** The segment ID for which each row of x
      belongs. This is a similar interface to :func:`jax.ops.segment_sum`.
    segment_ids_y: **1st interface** The segment ID for which each row of y
      belongs.
    indices_are_sorted: **1st interface** Whether `segment_ids_x` and
      `segment_ids_y` are sorted. Default false.
    num_per_segment_x: **2nd interface** Number of points in each segment in
      `x`. For example, [100, 20, 30] would imply that `x` is segmented into
      three arrays of length `[100]`, `[20]`, and `[30]` respectively.
    num_per_segment_y: **2nd interface** Number of points in each segment in
      `y`.
    weights_x: Weights of each input points, arranged in the same segmented
      order as `x`.
    weights_y: Weights of each input points, arranged in the same segmented
      order as `y`.
    sinkhorn_kwargs: Optionally a dict containing the keywords arguments for
      calls to the `sinkhorn` function, called three times to evaluate for each
      segment the sinkhorn regularized OT cost between `x`/`y`, `x`/`x`, and
      `y`/`y` (except when `static_b` is `True`, in which case `y`/`y` is not
      evaluated)
    static_b: if True, divergence of measure b against itself is NOT computed
    share_epsilon: if True, enforces that the same epsilon regularizer is shared
      for all 2 or 3 terms of the Sinkhorn divergence. In that case, the epsilon
      will be by default that used when comparing x to y (contained in the first
      geometry). This flag is set to True by default, because in the default
      setting, the epsilon regularization is a function of the mean of the cost
      matrix.
    kwargs: keywords arguments passed to form
      :class:`ott.geometry.pointcloud.PointCloud` geometry objects from the
      subsets of points and masses selected in `x` and `y`, this could be for
      instance entropy regularization float, scheduler or normalization.
  Returns:
    An array of sinkhorn divergence values for each segment.
  """
  # instantiate padding vector
  dim = x.shape[1]
  if cost_fn is None:
    # default padder
    padding_vector = costs.CostFn.padder(dim=dim)
  else:
    padding_vector = cost_fn.padder(dim=dim)

  def eval_fn(
      padded_x: jnp.ndarray,
      padded_y: jnp.ndarray,
      padded_weight_x: jnp.ndarray,
      padded_weight_y: jnp.ndarray,
  ) -> float:
    mask_x = padded_weight_x > 0.
    mask_y = padded_weight_y > 0.
    return sinkhorn_divergence(
        pointcloud.PointCloud,
        padded_x,
        padded_y,
        a=padded_weight_x,
        b=padded_weight_y,
        sinkhorn_kwargs=sinkhorn_kwargs,
        static_b=static_b,
        share_epsilon=share_epsilon,
        cost_fn=cost_fn,
        src_mask=mask_x,
        tgt_mask=mask_y,
        **kwargs
    ).divergence

  return segment._segment_interface(
      x,
      y,
      eval_fn,
      num_segments=num_segments,
      max_measure_size=max_measure_size,
      segment_ids_x=segment_ids_x,
      segment_ids_y=segment_ids_y,
      indices_are_sorted=indices_are_sorted,
      num_per_segment_x=num_per_segment_x,
      num_per_segment_y=num_per_segment_y,
      weights_x=weights_x,
      weights_y=weights_y,
      padding_vector=padding_vector
  )
