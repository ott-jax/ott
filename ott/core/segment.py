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
"""Prepare point clouds for parallel computations."""
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
from jax import numpy as jnp


def segment_point_cloud(
    x: jnp.ndarray,
    a: Optional[jnp.ndarray] = None,
    segment_ids: Optional[jnp.ndarray] = None,
    num_segments: Optional[int] = None,
    indices_are_sorted: Optional[bool] = None,
    num_per_segment: Optional[jnp.ndarray] = None,
    max_measure_size: Optional[int] = None,
    padding_vector: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
  """Segment and pad as needed the entries of a point cloud.

  There are two interfaces: either use `segment_ids`, and optionally
  `num_segments` and `indices_are_sorted`, to describe for each
  data point in the matrix to which segment each point corresponds to,
  OR use `num_per_segment`, which describes contiguous segments.

  If using the first interface, `num_segments` is required for JIT compilation.
  Assumes range(0, `num_segments`) are the segment ids.

  In both cases, jitting requires defining a max_measure_size, the
  upper bound on the maximal size of measures, which will be used for padding.

  Args:
    y: a matrix merging the points of all measures.
    a: a vector containing the weights (within each measure) of all the points.
    segment_ids: describe for each point to which measure it belongs.
    num_segments: total number of measures.
    indices_are_sorted: flag indicating indices in segment_ids are sorted.
    num_per_segment: number of points in each segment, if contiguous.
    max_measure_size: max number of points in each segment (for efficient jit)
    padding_vector: Vector of shape ``[1, dim]`` used for padding.
      If ``None``, use a vector of 0s.

  Returns:
    Segmented ``x``, ``y`` and the number segments.
  """
  num, dim = x.shape
  use_segment_ids = segment_ids is not None
  if use_segment_ids:
    if num_segments is None:
      num_segments = jnp.max(segment_ids) + 1
    if indices_are_sorted is None:
      indices_are_sorted = False

    num_per_segment = jax.ops.segment_sum(
        jnp.ones_like(segment_ids),
        segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted
    )
  else:
    assert num_per_segment is not None
    assert num_segments is None or num_segments == num_per_segment.shape[0]
    num_segments = num_per_segment.shape[0]
    segment_ids = jnp.arange(num_segments).repeat(
        num_per_segment, total_repeat_length=num
    )

  if a is None:
    a = (1 / num_per_segment).repeat(num_per_segment)

  if max_measure_size is None:
    max_measure_size = jnp.max(num_per_segment)

  if padding_vector is None:
    padding_vector = jnp.zeros((1, dim))

  segmented_a = []
  segmented_x = []

  x = jnp.concatenate((x, padding_vector))
  a = jnp.concatenate((a, jnp.zeros((1,))))

  for i in range(num_segments):
    idx = jnp.where(segment_ids == i, jnp.arange(num), num + 1)
    idx = jax.lax.dynamic_slice(jnp.sort(idx), (0,), (max_measure_size,))

    # segment the weights
    z = a.at[idx].get()
    segmented_a.append(z)

    # segment the positions
    z = x.at[idx].get()
    segmented_x.append(z)

  segmented_a = jnp.stack(segmented_a)
  segmented_x = jnp.stack(segmented_x)

  return segmented_x, segmented_a, num_segments


def pad_along_axis(
    x: Sequence[jnp.ndarray],
    max_pad_size: Mapping[int, Optional[int]] = MappingProxyType({}),
    constant_values: Any = 0.0,
    **kwargs: Any,
) -> jnp.ndarray:
  """Pad and stack sequence of arrays.

  Args:
    x: Sequence of arrays to pad.
    max_pad_size: Maximum padding size along each axis. Always pads after.
      Each key specifies an axis to pad, value corresponds to its new size.
      If the value is ``None``, maximum value across all arrays is used.
    constant_values: Value to pad with.
    kwargs: Keyword arguments for :func:`jax.numpy.pad`.

  Returns:
    The padded array.
  """
  shapes = jnp.asarray([arr.shape for arr in x])
  res = []

  for arr in x:
    pad_width = []
    # TODO(michalk8): handle negative axes
    for dim in range(arr.ndim):
      max_size = max_pad_size.get(dim, arr.shape[dim])
      if max_size is None:
        max_size = jnp.max(shapes[:, dim])
      # if negative, `jnp.pad` will raise
      pad_width.append((0, max_size - arr.shape[dim]))
    padded = jnp.pad(
        arr,
        pad_width=pad_width,
        mode='constant',
        constant_values=constant_values,
        **kwargs
    )
    res.append(padded)

  return jnp.asarray(res)
