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

"""Soft sort operators."""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ott.tools import transport


def transport_for_sort(inputs: jnp.ndarray,
                       weights: jnp.ndarray,
                       target_weights: jnp.ndarray,
                       kwargs) -> jnp.ndarray:
  """Solves reg. OT, from inputs to a weighted family of increasing values.

  Args:
    inputs: jnp.ndarray[num_points]. Must be one dimensional.
    weights: jnp.ndarray[num_points]. Weight vector `a` for input values.
    target_weights: jnp.ndarray[num_targets]: Weight vector of the target
      measure. It may be of different size than `weights`.
    kwargs: a dictionary holding the keyword arguments for `sinkhorn` and
      `PointCloud`.

  Returns:
    A jnp.ndarray<float> representing the transport matrix of the inputs onto
    the underlying sorted target.
  """
  shape = inputs.shape
  if len(shape) > 2 or (len(shape) == 2 and shape[1] != 1):
    raise ValueError(
        'Shape ({shape}) not supported. The input should be one-dimensional.')

  x = jnp.expand_dims(jnp.squeeze(inputs), axis=1)
  x = jax.nn.sigmoid((x - jnp.mean(x)) / (jnp.std(x) + 1e-10))
  a = jnp.squeeze(weights)
  b = jnp.squeeze(target_weights)
  num_targets = b.shape[0]
  y = jnp.linspace(0.0, 1.0, num_targets)[:, jnp.newaxis]
  return transport.Transport(x, y, a=a, b=b, **kwargs)


def apply_on_axis(op, inputs, axis, *args):
  """Applies a differentiable operator on a given axis of the input.

  Args:
    op: a differentiable operator (can be ranks, quantile, etc.)
    inputs: jnp.ndarray<float> of any shape.
    axis: the axis (int) or tuple of ints on which to apply the operator. If
      several axes are passed the operator, those are merged as a single
      dimension.
    *args: other positional arguments to the operator.

  Returns:
    A jnp.ndarray holding the output of the differentiable operator on the given
    axis.
  """
  axis = (axis,) if isinstance(axis, int) else axis
  num_points = np.prod(np.array(inputs.shape)[tuple([axis])])
  permutation = np.arange(len(inputs.shape))
  axis = tuple(permutation[a] for a in axis)
  permutation = tuple(sorted(set(permutation) - set(axis)) + sorted(axis))
  inputs = jnp.transpose(inputs, permutation)

  batch_fn = jax.vmap(op, in_axes=(0,) + (None,) * len(args))
  result = batch_fn(jnp.reshape(inputs, (-1, num_points)), *args)
  shrink = len(axis)
  result = jnp.reshape(result, inputs.shape[:-shrink] + result.shape[-1:])

  permutation = tuple(range(len(result.shape)))
  rank = len(result.shape) - 1
  axis = min(axis)
  permutation = permutation[:axis] + (rank,) + permutation[axis:-1]
  result = jnp.transpose(result, permutation)
  return result


def _sort(inputs: jnp.ndarray, topk, kwargs) -> jnp.ndarray:
  """Applies the soft sort operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  if 0 < topk < num_points:
    start_index = 1
    b = jnp.concatenate([
        jnp.array([(num_points - topk) / num_points]),
        jnp.ones(topk, dtype=inputs.dtype) / num_points
    ])
  else:
    start_index = 0
    b = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(inputs, a, b, kwargs)
  out = 1.0 / b * ot.apply(inputs, axis=0)
  return out[start_index:]


def sort(inputs: jnp.ndarray,
         axis: int = -1,
         topk: int = -1,
         **kwargs) -> jnp.ndarray:
  """Applies the soft sort operator on a given axis of the input.

  Args:
    inputs: jnp.ndarray<float> of any shape.
    axis: the axis on which to apply the operator.
    topk: if set to a positive value, the returned vector will be only the topk
      values. This also reduces the complexity and speed of soft sorting.
    **kwargs: keyword arguments of the PointCloud class. See
      pointcloud.py for more details.

  Returns:
    A jnp.ndarray of the same shape as the input with soft sorted values on the
    given axis.
  """
  return apply_on_axis(_sort, inputs, axis, topk, kwargs)


def _ranks(inputs: jnp.ndarray, kwargs) -> jnp.ndarray:
  """Applies the soft ranks operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  b = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(inputs, a, b, kwargs)
  out = 1.0 / a * ot.apply(jnp.arange(num_points), axis=1)
  return jnp.reshape(out, inputs.shape)


def ranks(inputs: jnp.ndarray,
          axis: int = -1,
          **kwargs) -> jnp.ndarray:
  """Applies the sof trank operator on input tensor.

  Args:
    inputs: a jnp.ndarray<float> of any shape.
    axis: the axis on which to apply the soft ranks operator.
    **kwargs: extra arguments to the underlying `PointCloud` geometry object as
      well as the sinkhorn parameters.

  Returns:
    A jnp.ndarray<float> of the same shape as inputs, with the ranks.
  """
  return apply_on_axis(_ranks, inputs, axis, kwargs)


def quantile(inputs: jnp.ndarray,
             axis: int = -1,
             level: float = 0.5,
             weight: float = 0.05,
             **kwargs) -> jnp.ndarray:
  """Applies the soft quantile operator on the input tensor.

  For instance:

  x = jax.random.uniform(rng, (1000,))
  q = quantile(x, 0.5, 0.01)

  Then q will be computed as a mean over the 10 median points of x.
  Therefore, there is a tradeoff between accuracy and gradient.

  Args:
   inputs: a jnp.ndarray<float> of any shape.
   axis: the axis on which to apply the operator.
   level: the value of the quantile level to be computed. 0.5 for median.
   weight: the weight of the quantile in the transport problem.
   **kwargs: extra arguments to the underlying `PointCloud` geometry object.

  Returns:
    A jnp.ndarray, which has the same shape as the input, except on the give
    axis on which the dimension is 1.
  """
  def _quantile(inputs: jnp.ndarray,
                level: float,
                weight: float,
                kwargs: Dict[str, Any]) -> jnp.ndarray:
    num_points = inputs.shape[0]
    a = jnp.ones((num_points,)) / num_points
    b = jnp.array([level - weight / 2, weight, 1.0 - weight / 2 - level])
    ot = transport_for_sort(inputs, a, b, kwargs)
    out = 1.0 / b * ot.apply(jnp.squeeze(inputs), axis=0)
    return out[1:2]

  return apply_on_axis(_quantile, inputs, axis, level, weight, kwargs)


def _quantile_normalization(inputs: jnp.ndarray,
                            targets: jnp.ndarray,
                            weights: float,
                            kwargs: Dict[str, Any]) -> jnp.ndarray:
  """Applies soft quantile normalization on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(inputs, a, weights, kwargs)
  return 1.0 / a * ot.apply(targets, axis=1)


def quantile_normalization(inputs: jnp.ndarray,
                           targets: jnp.ndarray,
                           weights: Optional[jnp.ndarray] = None,
                           axis: int = -1,
                           **kwargs) -> jnp.ndarray:
  """Renormalizes inputs so that its quantiles match those of targets/weights.

  The idea of quantile normalization is to map the inputs to values so that the
  distribution of transformed values matches the distribution of target values.
  In a sense, we want to keep the inputs in the same order, but apply the values
  of the target.

  Args:
    inputs: the inputs array of any shape.
    targets: the target values of dimension 1. The targets must be sorted.
    weights: if set, the weights or the target.
    axis: the axis along which to apply the transformation on the inputs.
    **kwargs: extra arguments to the underlying `PointCloud` geometry object as
      well as the sinkhorn algorithm.

  Returns:
    A jnp.ndarray, which has the same shape as the input, except on the give
    axis on which the dimension is 1.

  Raises:
    A ValueError in case the weights and the targets are both set and not of
    compatible shapes.
  """
  if weights is not None and weights.shape != targets.shape:
    raise ValueError('The target weights and targets values should have the '
                     f'same shape: {targets.shape} != {weights.shape}')
  if weights is None:
    num_targets = targets.shape[0]
    weights = jnp.ones((num_targets,)) / num_targets

  op = _quantile_normalization
  return apply_on_axis(op, inputs, axis, targets, weights, kwargs)


def sort_with(inputs: jnp.ndarray,
              criterion: jnp.ndarray,
              topk: int = -1,
              **kwargs) -> jnp.ndarray:
  """Sort an array according to a criterion.

  Given `batch` vectors of dimension `dim`, this function produces `topk` (or
  `batch` if `topk` is set to -1, as by default) vectors of size `dim` that will
  be convex combinations of all vectors, ranked from smallest to largest.
  Vectors with the largest indices will contain convex combinations of those
  vectors with highest criterion, vectors with smaller indices will contain
  combinations of vectors with smaller criterion.

  Args:
    inputs: the inputs as a jnp.ndarray[batch, dim].
    criterion: the values according to which to sort the inputs. It has shape
      [batch, 1].
    topk: The number of outputs to keep.
    **kwargs: extra arguments to the underlying `PointCloud` geometry object as
      well as the sinkhorn parameters.

  Returns:
    A jnp.ndarray[batch | topk, dim].
  """
  num_points = criterion.shape[0]
  weights = jnp.ones(num_points, dtype=criterion.dtype) / num_points
  if 0 < topk < num_points:
    start_index = 1
    target_weights = jnp.concatenate([
        jnp.array([(num_points - topk) / num_points]),
        jnp.ones(topk, dtype=inputs.dtype) / num_points
    ])
  else:
    start_index = 0
    target_weights = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(criterion, weights, target_weights, kwargs)
  # Applies the topk on each of the dimensions of the inputs.
  sort_fn = jax.vmap(
      lambda x: (1.0 / target_weights * ot.apply(x, axis=0))[start_index:],
      in_axes=(1,), out_axes=1)
  return sort_fn(inputs)


def _quantize(inputs: jnp.ndarray,
              num_levels: int,
              kwargs: Dict[str, Any]) -> jnp.ndarray:
  """Applies the soft quantization operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  b = jnp.ones((num_levels,)) / num_levels
  ot = transport_for_sort(inputs, a, b, kwargs)
  return 1.0 / a * ot.apply(1.0 / b * ot.apply(inputs), axis=1)


def quantize(inputs: jnp.ndarray,
             num_levels: int,
             axis: int = -1,
             **kwargs):
  """Soft quantizes an inputs according to some levels along the given axis."""
  return apply_on_axis(_quantize, inputs, axis, num_levels, kwargs)
