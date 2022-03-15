"""Prepare point clouds for parallel computations."""


from typing import Any, Dict, Mapping, Optional, Type
import jax
from jax import numpy as jnp

def segment_point_cloud(
  x: jnp.ndarray,
  a: Optional[jnp.ndarray] = None,
  segment_ids: Optional[jnp.ndarray] = None,
  num_segments: Optional[int] = None,
  indices_are_sorted: Optional[bool] = None,
  num_per_segment: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
  """ Segment and pad as needed the entries of a.
  There are two interfaces: either use `segment_ids`, and optionally 
  `num_segments` and `indices_are_sorted`, OR use `num_per_segment`. 
  
  If using the first interface, `num_segments` is required for JIT compilation.
  Assumes range(0, `num_segments`) are the segment ids.  
  """
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
        indices_are_sorted=indices_are_sorted)
  else:
    assert num_per_segment is not None
    assert num_segments is None or num_segments == num_per_segment.shape[0]
    num_segments = num_per_segment.shape[0]
    segment_ids = jnp.arange(num_segments).repeat(num_per_segment)    

  num = x.shape[0]

  if a is None:
    a = (1 / num_per_segment).repeat(
        num_per_segment, total_repeat_length=num)

  segmented_x = jnp.stack(
      [x * (segment_ids == i)[:, None] for i in range(num_segments)])

  segmented_a = jnp.stack(
      [a * (segment_ids == i) for i in range(num_segments)])

  return segmented_x, segmented_a, num_segments