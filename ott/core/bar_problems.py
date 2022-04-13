# coding=utf-8
# Copyright 2022 Apple Inc
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
    weights: Optional[jnp.ndarray] = None,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[jnp.ndarray] = None,
    debiased: bool = False,
    segment_ids: Optional[jnp.ndarray] = None,
    num_segments: Optional[jnp.ndarray] = None,
    indices_are_sorted: Optional[bool] = None,
    num_per_segment: Optional[jnp.ndarray] = None,
    max_measure_size: Optional[int] = None):
    """Initializes a discrete BarycenterProblem 

    Args:
      y: a matrix merging the points of all measures.
      b: a vector containing the weights (within each masure) of all the points
      weights: weights of the barycenter problem (size num_segments)
      cost_fn: cost function used.
      epsilon: epsilon regularization used to solve reg-OT problems.
      debiased: whether the problem is debiased, in the sense that
        the regularized transportation cost of barycenter to itself will
        be considered when computing gradient. Note that if the debiased option
        is used, the barycenter size (used in call function) needs to be smaller
        than the max_measure_size parameter below, for parallelization to
        operate efficiently.
      segment_ids: describe for each point to which measure it belongs.
      num_segments: total number of measures
      indices_are_sorted: flag indicating indices in segment_ids are sorted.
      num_per_segment: number of points in each segment, if contiguous.
      max_measure_size: max number of points in each segment (for efficient jit)
    """
    self._y = y
    self._b = b
    self._weights = weights
    self.cost_fn = costs.Euclidean() if cost_fn is None else cost_fn
    self.epsilon = epsilon
    self.debiased = debiased
    self._segment_ids = segment_ids
    self._num_segments = num_segments
    self._indices_are_sorted = indices_are_sorted
    self._num_per_segment = num_per_segment
    self._max_measure_size = max_measure_size
    
  def tree_flatten(self):
    return ([self._y, self._b, self._weights],
            {
            'cost_fn' : self.cost_fn, 
            'epsilon' : self.epsilon,
            'debiased': self.debiased, 
            'segment_ids' : self._segment_ids, 
            'num_segments' : self._num_segments,
            'indices_are_sorted' : self._indices_are_sorted,
            'num_per_segment' : self._num_per_segment,
            'max_measure_size' : self._max_measure_size})

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  @property
  def segmented_y_b(self):
    if self._y is None or (self._y.ndim == 3 and self._b.ndim == 2):      
      return self.add_slice_for_debiased(self._y, self._b)
    else:  
      segmented_y, segmented_b, _ = segment.segment_point_cloud(
        self._y, self._b, self._segment_ids, self._num_segments,
        self._indices_are_sorted, self._num_per_segment,
        self.max_measure_size)
    return self.add_slice_for_debiased(segmented_y, segmented_b)
  
  def add_slice_for_debiased(self, y, b):
    if y is None or b is None:
      return y, b    
    if self.debiased:  
      n, dim = y.shape[1], y.shape[2]
      y = jnp.concatenate((y, jnp.zeros((1, n, dim))), axis=0)
      b = jnp.concatenate((b, jnp.zeros((1, n,))), axis=0)    
    return y, b

  @property
  def flattened_y(self):
    if self._y is not None and self._y.ndim == 3:
      return self._y.reshape((-1,self._y.shape[-1]))
    else:  
      return self._y
  
  @property
  def flattened_b(self):
    if self._b is not None and self._b.ndim == 2:
      return self._b.ravel()
    else:
      return self._b
    
  @property
  def max_measure_size(self):
    if self._max_measure_size is not None:
      return self._max_measure_size
    if self._y is not None and self._y.ndim == 3:
      return self._y.shape[1]
    else:
      if self._num_per_segment is None:
        if self._num_segments is None:
          num_segments = jnp.max(self._segment_ids) + 1
        if self._indices_are_sorted is None:
          indices_are_sorted = False
        num_per_segment = jax.ops.segment_sum(
          jnp.ones_like(self._segment_ids), self._segment_ids,
          num_segments=num_segments, indices_are_sorted=indices_are_sorted)
        return jnp.max(num_per_segment)  
      else:
        return jnp.max(self._num_per_segment)

  @property
  def num_segments(self):
    if self._y is None:
      return 0
    if self._y.ndim == 3:
      if self._b is not None:
        assert self._y.shape[0] == self._b.shape[0]
      return self._y.shape[0]
    else:
      _ , _, num_segments = segment.segment_point_cloud(
        self._y, self._b, self._segment_ids, self._num_segments,
        self._indices_are_sorted, self._num_per_segment,
        self.max_measure_size)
    return num_segments


  @property
  def weights(self):
    if self._weights is None:
      weights = jnp.ones((self.num_segments,)) / self.num_segments
    else:
      assert self.weights.shape[0] == self.num_segments
      assert jnp.isclose(jnp.sum(self.weights), 1.0)
      weights = self.weights
    if self.debiased:
      weights = jnp.concatenate((weights, jnp.array([-0.5])))    
    return weights
