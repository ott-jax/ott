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

"""Classes defining OT problem(s) (objective function + utilities)."""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.geometry import costs
from ott.core import segment


@jax.tree_util.register_pytree_node_class
class BarycenterProblem:
  """Holds the definition of a linear regularized OT problem and some tools."""

  def __init__(self,
    y: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    segment_ids: Optional[jnp.ndarray] = None,
    num_segments: Optional[jnp.ndarray] = None,
    indices_are_sorted: Optional[bool] = None,
    num_per_segment: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[jnp.ndarray] = None):
    """Initializes a discrete BarycenterProblem 

    Args:
      y: a matrix merging the points of all measures.
      b: a vector containing the weights (within each masure) of all the points
      segment_ids: describe for each point to which measure it belongs.
      num_segments: total number of measures
      indices_are_sorted: flag indicating indices in segment_ids are sorted.
      num_per_segment: number of points in each segment, if contiguous.
      weights: weights of the barycenter problem (size num_segments)
      cost_fn: cost function used.
      epsilon: epsilon regularization used to solve reg-OT problems.
    """
    self._y = y
    self._b = b
    self._segment_ids = segment_ids
    self._num_segments = num_segments
    self._indices_are_sorted = indices_are_sorted
    self._num_per_segment = num_per_segment
    self._weights = weights
    self.cost_fn = cost_fn
    self._epsilon = epsilon
    
  def tree_flatten(self):
    return ([self.segment_y, self.segment_b, self.weights, self.cost_fn],
            None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  @property
  def segmented_y(self):
    if self._y.ndim == 3:
      return self._y
    else:  
      segmented_y, _ = segment.segment_point_cloud(
        self._y, self._segment_ids, self._num_segments,
        self._indices_are_sorted, self._num_per_segment)
    return segmented_y
  
  @property
  def segmented_b(self):
    if self._b.ndim == 2:
      return self._b
    else:
      segmented_b, _ = segment.segment_point_cloud(
      self.b, self._segment_ids, self._num_segments,
      self._indices_are_sorted, self._num_per_segment)
    return segmented_b

  @property
  def flattened_y(self):
    if self._y.ndim == 3:
      return self._y.reshape((-1,self._y.shape[-1]))
    else:  
      return self._y
  
  @property
  def flattened_b(self):
    return self._b.ravel() if self._b.ndim == 2 else self._b
    
  @property
  def num_segments(self):
    if self._y.ndim == 3:
      if self._b is not None:
        assert self._y.shape[0] == self._b.shape[0]
      return self._y.shape[0]
    else:
      _ , num_segments = segment.segment_point_cloud(
        self.b, self._segment_ids, self._num_segments,
        self._indices_are_sorted, self._num_per_segment)
    return num_segments


  @property
  def weights(self):
    if self._weights is None:
      return jnp.ones((self.num_segments,)) / self.num_segments
    else:
      assert self.weights.shape[0] == self.num_segments
      return self.weights

  @property
  def epsilon(self):
    return self._epsilon