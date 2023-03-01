#
# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

__all__ = ["segment_point_cloud"]


def segment_point_cloud(
    x: jnp.ndarray,
    a: Optional[jnp.ndarray] = None,
    num_segments: Optional[int] = None,
    max_measure_size: Optional[int] = None,
    segment_ids: Optional[jnp.ndarray] = None,
    indices_are_sorted: bool = False,
    num_per_segment: Optional[Tuple[int, ...]] = None,
    padding_vector: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Segment and pad as needed the entries of a point cloud.

  There are two interfaces:

  #. use ``segment_ids``, and optionally ``indices_are_sorted`` to describe
     for each data point in the matrix to which segment it belongs to.
  #. use ``num_per_segment`` which describes contiguous segments.

  If using the first interface, ``num_segments`` is required for jitting.
  Assumes ``range(0, num_segments)`` are the segment ids.

  In both cases, jitting requires defining a ``max_measure_size``, the
  upper bound on the maximal size of measures, which will be used for padding.

  Args:
    x: Array of input points, of shape ``[num_x, ndim]``.
      Multiple segments are held in this single array.
    a: Array of shape ``[num_x,]`` containing the weights (within each measure)
      of all the points.
    num_segments: Number of segments. Required for jitting.
      If `None` and using the second interface, it will be computed as
      ``len(num_per_segment)``.
    max_measure_size: Overall size of padding. Required for jitting.
      If `None` and using the second interface, it will be computed as
      ``max(num_per_segment)``.
    segment_ids: **1st interface** The segment ids for which each row of ``x``
      belongs. This is a similar interface to :func:`jax.ops.segment_sum`.
    indices_are_sorted: **1st interface** Whether ``segment_ids`` are sorted.
    num_per_segment: **2nd interface** Number of points in each segment.
      For example, `[100, 20, 30]` would imply that ``x`` is segmented into 3
      arrays of length `[100]`, `[20]`, and `[30]`, respectively.
      Must be a tuple and not a :class:`jax.numpy.ndarray` to allow jitting.
      This means changes in ``num_per_segment`` will re-trigger compilation.
    padding_vector: vector to be used to pad point cloud matrices. Most likely
      to be zero, but can be adjusted to be other values to avoid errors or
      over/underflow in cost matrix that could be problematic (even these values
      are not supposed to be taken given their corresponding masses are 0).
      See also :func:`~ott.geometry.costs.CostFn._padder`.
      If ``None``, vector of 0s of shape ``[1, ndim]`` is used.

  Returns:
    Segmented ``x`` as an array of shape
    ``[num_measures, max_measure_size, ndim]`` and ``a`` as an array of shape
    ``[num_measures, max_measure_size]``.
  """
  num, dim = x.shape
  use_segment_ids = segment_ids is not None
  if use_segment_ids:
    assert num_segments is not None, "Please specify `num_segments`."
    assert max_measure_size is not None, "Please specify `max_measure_size`."
    num_per_segment = jax.ops.segment_sum(
        jnp.ones_like(segment_ids),
        segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted
    )
  else:
    assert num_per_segment is not None, "Please specify `num_per_segment`."
    if max_measure_size is None:
      max_measure_size = max(num_per_segment)
    if num_segments is None:
      num_segments = len(num_per_segment)
    else:
      assert num_segments == len(num_per_segment)
    # conversion to facilitate computation of default weight below.
    num_per_segment = jnp.array(num_per_segment)
    segment_ids = jnp.arange(num_segments).repeat(
        num_per_segment, total_repeat_length=num
    )

  if a is None:
    a = jnp.array(
        (1.0 /
         num_per_segment).repeat(num_per_segment, total_repeat_length=num)
    )

  if padding_vector is None:
    padding_vector = jnp.zeros((1, dim))

  x = jnp.concatenate((x, padding_vector))
  a = jnp.concatenate((a, jnp.zeros((1,))))
  segmented_a, segmented_x = [], []

  for i in range(num_segments):
    idx = jnp.where(segment_ids == i, jnp.arange(num), num + 1)
    idx = jax.lax.dynamic_slice(jnp.sort(idx), (0,), (max_measure_size,))

    # segment the weights
    segmented_a.append(a.at[idx].get())
    # segment the positions
    segmented_x.append(x.at[idx].get())

  segmented_a = jnp.stack(segmented_a)
  segmented_x = jnp.stack(segmented_x)

  return segmented_x, segmented_a


def _segment_interface(
    x: jnp.ndarray,
    y: jnp.ndarray,
    eval_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                      jnp.ndarray],
    num_segments: Optional[int] = None,
    max_measure_size: Optional[int] = None,
    segment_ids_x: Optional[jnp.ndarray] = None,
    segment_ids_y: Optional[jnp.ndarray] = None,
    indices_are_sorted: bool = False,
    num_per_segment_x: Optional[jnp.ndarray] = None,
    num_per_segment_y: Optional[jnp.ndarray] = None,
    weights_x: Optional[jnp.ndarray] = None,
    weights_y: Optional[jnp.ndarray] = None,
    padding_vector: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Wrapper to segment two point clouds and return parallel evaluations.

  Utility function that segments two point clouds using the approach outlined
  in `segment_point_cloud` and evaluates `eval_fn` on pairs of segmented point
  clouds.
  """
  use_segment_ids = segment_ids_x is not None
  if use_segment_ids:
    assert segment_ids_y is not None
  else:
    assert num_per_segment_x is not None
    assert num_per_segment_y is not None

  segmented_x, segmented_weights_x = segment_point_cloud(
      x,
      a=weights_x,
      num_segments=num_segments,
      max_measure_size=max_measure_size,
      segment_ids=segment_ids_x,
      indices_are_sorted=indices_are_sorted,
      num_per_segment=num_per_segment_x,
      padding_vector=padding_vector
  )

  segmented_y, segmented_weights_y = segment_point_cloud(
      y,
      a=weights_y,
      num_segments=num_segments,
      max_measure_size=max_measure_size,
      segment_ids=segment_ids_y,
      indices_are_sorted=indices_are_sorted,
      num_per_segment=num_per_segment_y,
      padding_vector=padding_vector
  )

  v_eval = jax.vmap(eval_fn, in_axes=[0] * 4)
  return v_eval(
      segmented_x,
      segmented_y,
      segmented_weights_x,
      segmented_weights_y,
  )
