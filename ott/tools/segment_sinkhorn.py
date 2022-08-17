# Copyright 2022 The OTT Authors.
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
"""Segmented sinkhorn utility."""
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from jax import numpy as jnp

from ott.core import segment, sinkhorn
from ott.geometry import costs, pointcloud


def segment_sinkhorn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    num_segments: Optional[int] = None,
    max_measure_size: Optional[int] = None,
    cost_fn: Optional[costs.CostFn] = None,
    segment_ids_x: Optional[jnp.ndarray] = None,
    segment_ids_y: Optional[jnp.ndarray] = None,
    indices_are_sorted: Optional[bool] = None,
    num_per_segment_x: Tuple[int] = None,
    num_per_segment_y: Tuple[int] = None,
    weights_x: Optional[jnp.ndarray] = None,
    weights_y: Optional[jnp.ndarray] = None,
    sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any
) -> jnp.ndarray:
  """Compute `reg_ot_cost` between subsets of vectors described in `x` & `y`.

  Helper function designed to compute Sinkhorn regularized OT cost between
  several point clouds of varying size, in parallel, using padding.
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
  a tensor, and :func:`jax.vmap` used to evaluate sinkhorn divergences in
  parallel.

  Args:
    x: Array of input points, of shape [num_x, feature]. Multiple segments are
      held in this single array.
    y: Array of target points, of shape [num_y, feature].
    num_segments: Number of segments contained in x and y. Providing this number
      is required for JIT compilation to work, see also
      :func:`~ott.core.segment.segment_point_cloud`.
    max_measure_size: Total size of measures after padding. Should ideally be
      set to an upper bound on points clouds processed with the segment
      interface. Providing this number is required for JIT compilation to work.
    cost_fn: Cost function, defaults to :class:`~ott.core.costs.Euclidean`.
    segment_ids_x: **1st interface** The segment ID for which each row of x
      belongs. This is a similar interface to `jax.ops.segment_sum`.
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
      evaluated).
    kwargs: keywords arguments passed to form
      :class:`ott.geometry.pointcloud.PointCloud` geometry objects from the
      subsets of points and masses selected in `x` and `y`, possibly a
      :class:`ott.geometry.costs.CostFn` or an entropy regularizer.

  Returns:
    An array of sinkhorn reg_ot_cost for each segment.
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
    return sinkhorn.sinkhorn(
        pointcloud.PointCloud(
            padded_x,
            padded_y,
            cost_fn=cost_fn,
            src_mask=mask_x,
            tgt_mask=mask_y,
            **kwargs
        ),
        a=padded_weight_x,
        b=padded_weight_y,
        **sinkhorn_kwargs
    ).reg_ot_cost

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
