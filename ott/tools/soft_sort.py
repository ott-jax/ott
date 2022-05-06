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

"""Soft sort operators."""

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from ott.tools import transport


def transport_for_sort(
    inputs: jnp.ndarray,
    weights: jnp.ndarray,
    target_weights: jnp.ndarray,
    squashing_fun: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    epsilon: float = 1e-2,
    **kwargs) -> jnp.ndarray:
  r"""Solves reg. OT, from inputs to a weighted family of increasing values.

  Args:
    inputs: jnp.ndarray[num_points]. Must be one dimensional.
    weights: jnp.ndarray[num_points]. Weight vector `a` for input values.
    target_weights: jnp.ndarray[num_targets]: Weight vector of the target
      measure. It may be of different size than `weights`.
    squashing_fun: function taking an array to squash all its entries in [0,1].
      sigmoid of whitened values by default. Can be set to be the identity by
      passing ``squashing_fun = lambda x : x`` instead.
    epsilon: the regularization parameter.
    **kwargs: keyword arguments for `sinkhorn` and / or `PointCloud`.

  Returns:
    A jnp.ndarray<float> num_points x num_target transport matrix, from all
    inputs onto the sorted target.
  """
  shape = inputs.shape
  if len(shape) > 2 or (len(shape) == 2 and shape[1] != 1):
    raise ValueError(
        'Shape ({shape}) not supported. The input should be one-dimensional.')

  x = jnp.expand_dims(jnp.squeeze(inputs), axis=1)
  if squashing_fun is None:
    squashing_fun = lambda z: jax.nn.sigmoid(
        (z - jnp.mean(z)) / (jnp.std(z) + 1e-10))
  x = squashing_fun(x)
  a = jnp.squeeze(weights)
  b = jnp.squeeze(target_weights)
  num_targets = b.shape[0]
  y = jnp.linspace(0.0, 1.0, num_targets)[:, jnp.newaxis]

  return transport.solve(x, y, a=a, b=b, epsilon=epsilon, **kwargs)


def apply_on_axis(op, inputs, axis, *args, **kwargs):
  """Applies a differentiable operator on a given axis of the input.

  Args:
    op: a differentiable operator (can be ranks, quantile, etc.)
    inputs: jnp.ndarray<float> of any shape.
    axis: the axis (int) or tuple of ints on which to apply the operator. If
      several axes are passed the operator, those are merged as a single
      dimension.
    *args: other positional arguments to the operator.
    **kwargs: other positional arguments to the operator.


  Returns:
    A jnp.ndarray holding the output of the differentiable operator on the given
    axis.
  """
  op_inner = functools.partial(op, **kwargs)
  axis = (axis,) if isinstance(axis, int) else axis
  num_points = np.prod(np.array(inputs.shape)[tuple([axis])])
  permutation = np.arange(len(inputs.shape))
  axis = tuple(permutation[a] for a in axis)
  permutation = tuple(sorted(set(permutation) - set(axis)) + sorted(axis))
  inputs = jnp.transpose(inputs, permutation)

  batch_fn = jax.vmap(op_inner, in_axes=(0,) + (None,) * len(args))
  result = batch_fn(jnp.reshape(inputs, (-1, num_points)), *args)
  shrink = len(axis)
  result = jnp.reshape(result, inputs.shape[:-shrink] + result.shape[-1:])

  permutation = tuple(range(len(result.shape)))
  rank = len(result.shape) - 1
  axis = min(axis)
  permutation = permutation[:axis] + (rank,) + permutation[axis:-1]
  result = jnp.transpose(result, permutation)
  return result


def _sort(inputs: jnp.ndarray, topk, num_targets, **kwargs) -> jnp.ndarray:
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
    num_targets = num_points if num_targets is None else num_targets
    start_index = 0
    b = jnp.ones((num_targets,)) / num_targets
  ot = transport_for_sort(inputs, a, b, **kwargs)
  out = 1.0 / b * ot.apply(inputs, axis=0)
  return out[start_index:]


def sort(inputs: jnp.ndarray,
         axis: int = -1,
         topk: int = -1,
         num_targets: Optional[int] = None,
         **kwargs) -> jnp.ndarray:
  r"""Applies the soft sort operator on a given axis of the input.

  Args:
    inputs: jnp.ndarray<float> of any shape.
    axis: the axis on which to apply the operator.
    topk: if set to a positive value, the returned vector will only contain
      the topk values. This also reduces the complexity of soft sorting.
    num_targets: if topk is not specified, num_targets defines the number of
      (composite) sorted values computed from the inputs (each value is a convex
      combination of values recorded in the inputs, provided in increasing
      order). If not specified, ``num_targets`` is set by default to be the size
      of the slices of the input that are sorted, i.e. the number of composite
      sorted values is equal to that of the inputs that are sorted.
    **kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in [0,1] (sigmoid of whitened values by default) to
      solve the optimal transport problem; ``cost_fn``, used in ``PointCloud``,
      that defines the ground cost function to transport from ``inputs`` to the
      ``num_targets`` target values (squared Euclidean distance by default, see
      ``pointcloud.py`` for more details); ``epsilon`` values as well as other
      parameters to shape the ``sinkhorn`` algorithm.

  Returns:
    A jnp.ndarray of the same shape as the input with soft sorted values on the
    given axis.
  """
  return apply_on_axis(_sort, inputs, axis, topk, num_targets, **kwargs)


def _ranks(inputs: jnp.ndarray, num_targets, **kwargs) -> jnp.ndarray:
  """Applies the soft ranks operator on a one dimensional array."""
  num_points = inputs.shape[0]
  num_targets = num_points if num_targets is None else num_targets
  a = jnp.ones((num_points,)) / num_points
  b = jnp.ones((num_targets,)) / num_targets
  ot = transport_for_sort(inputs, a, b, **kwargs)
  out = 1.0 / a * ot.apply(jnp.arange(num_targets), axis=1)
  return jnp.reshape(out, inputs.shape)


def ranks(inputs: jnp.ndarray,
          axis: int = -1,
          num_targets: Optional[int] = None,
          **kwargs) -> jnp.ndarray:
  r"""Applies the soft trank operator on input tensor.

  Args:
    inputs: a jnp.ndarray<float> of any shape.
    axis: the axis on which to apply the soft ranks operator.
    num_targets: num_targets defines the number of targets used to compute a
      composite ranks for each value in ``inputs``: that soft rank will be a
      convex combination of values in [0,...,``(num_targets-2)/num_targets``,1]
      specified by the optimal transport between values in ``inputs`` towards
      those values. If not specified, ``num_targets`` is set by default to be
      the size of the slices of the input that are sorted.
    **kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in [0,1] (sigmoid of whitened values by default) to
      solve the optimal transport problem; ``cost_fn``, used in ``PointCloud``,
      that defines the ground cost function to transport from ``inputs`` to the
      ``num_targets`` target values (squared Euclidean distance by default, see
      ``pointcloud.py`` for more details); ``epsilon`` values as well as other
      parameters to shape the ``sinkhorn`` algorithm.

  Returns:
    A jnp.ndarray<float> of the same shape as inputs, with the ranks.
  """
  return apply_on_axis(_ranks, inputs, axis, num_targets, **kwargs)


def quantile(inputs: jnp.ndarray,
             axis: int = -1,
             level: float = 0.5,
             weight: float = 0.05,
             **kwargs) -> jnp.ndarray:
  r"""Applies the soft quantile operator on the input tensor.

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
   **kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in [0,1] (sigmoid of whitened values by default) to
      solve the optimal transport problem; ``cost_fn``, used in ``PointCloud``,
      that defines the ground cost function to transport from ``inputs`` to the
      ``num_targets`` target values (squared Euclidean distance by default, see
      ``pointcloud.py`` for more details); ``epsilon`` values as well as other
      parameters to shape the ``sinkhorn`` algorithm.
  Returns:
    A jnp.ndarray, which has the same shape as the input, except on the give
    axis on which the dimension is 1.
  """
  # TODO(cuturi,oliviert) option to compute several quantiles at once, as in tf.
  def _quantile(inputs: jnp.ndarray,
                level: float,
                weight: float,
                **kwargs) -> jnp.ndarray:
    num_points = inputs.shape[0]
    a = jnp.ones((num_points,)) / num_points
    b = jnp.array([level - weight / 2, weight, 1.0 - weight / 2 - level])
    ot = transport_for_sort(inputs, a, b, **kwargs)
    out = 1.0 / b * ot.apply(jnp.squeeze(inputs), axis=0)
    return out[1:2]

  return apply_on_axis(_quantile, inputs, axis, level, weight, **kwargs)


def _quantile_normalization(inputs: jnp.ndarray,
                            targets: jnp.ndarray,
                            weights: float,
                            **kwargs) -> jnp.ndarray:
  """Applies soft quantile normalization on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(inputs, a, weights, **kwargs)
  return 1.0 / a * ot.apply(targets, axis=1)


def quantile_normalization(inputs: jnp.ndarray,
                           targets: jnp.ndarray,
                           weights: Optional[jnp.ndarray] = None,
                           axis: int = -1,
                           **kwargs) -> jnp.ndarray:
  r"""Renormalizes inputs so that its quantiles match those of targets/weights.

  The idea of quantile normalization is to map the inputs to values so that the
  distribution of transformed values matches the distribution of target values.
  In a sense, we want to keep the inputs in the same order, but apply the values
  of the target.

  Args:
    inputs: the inputs array of any shape.
    targets: the target values of dimension 1. The targets must be sorted.
    weights: if set, the weights or the target.
    axis: the axis along which to apply the transformation on the inputs.
    **kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in [0,1] (sigmoid of whitened values by default) to
      solve the optimal transport problem; ``cost_fn``, used in ``PointCloud``,
      that defines the ground cost function to transport from ``inputs`` to the
      ``num_targets`` target values (squared Euclidean distance by default, see
      ``pointcloud.py`` for more details); ``epsilon`` values as well as other
      parameters to shape the ``sinkhorn`` algorithm.
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
  return apply_on_axis(op, inputs, axis, targets, weights, **kwargs)


def sort_with(inputs: jnp.ndarray,
              criterion: jnp.ndarray,
              topk: int = -1,
              **kwargs) -> jnp.ndarray:
  r"""Sort a multidimensional array according to a real valued criterion.

  Given ``batch`` vectors of dimension `dim`, to which, for each, a real value
  ``criterion`` is associated, this function produces ``topk`` (or
  ``batch`` if ``topk`` is set to -1, as by default) composite vectors of size
  ``dim`` that will be convex combinations of all vectors, ranked from smallest
  to largest criterion. Composite vectors with the largest indices will contain
  convex combinations of those vectors with highest criterion, vectors with
  smaller indices will contain combinations of vectors with smaller criterion.

  Args:
    inputs: the inputs as a jnp.ndarray[batch, dim].
    criterion: the values according to which to sort the inputs. It has shape
      [batch, 1].
    topk: The number of outputs to keep.
    **kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in [0,1] (sigmoid of whitened values by default) to
      solve the optimal transport problem; ``cost_fn``, used in ``PointCloud``,
      that defines the ground cost function to transport from ``inputs`` to the
      ``num_targets`` target values (squared Euclidean distance by default, see
      ``pointcloud.py`` for more details); ``epsilon`` values as well as other
      parameters to shape the ``sinkhorn`` algorithm.

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
  ot = transport_for_sort(criterion, weights, target_weights, **kwargs)
  # Applies the topk on each of the dimensions of the inputs.
  sort_fn = jax.vmap(
      lambda x: (1.0 / target_weights * ot.apply(x, axis=0))[start_index:],
      in_axes=(1,), out_axes=1)
  return sort_fn(inputs)


def _quantize(inputs: jnp.ndarray,
              num_levels: int,
              **kwargs) -> jnp.ndarray:
  """Applies the soft quantization operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  b = jnp.ones((num_levels,)) / num_levels
  ot = transport_for_sort(inputs, a, b, **kwargs)
  return 1.0 / a * ot.apply(1.0 / b * ot.apply(inputs), axis=1)


def quantize(inputs: jnp.ndarray,
             num_levels: int = 10,
             axis: int = -1,
             **kwargs):
  r"""Soft quantizes an input according using num_levels values along axis.

  The quantization operator consists in concentrating several values around
  a few predefined ``num_levels``. The soft quantization operator proposed here
  does so by carrying out a `soft` concentration that is differentiable. The
  ``inputs`` values are first soft-sorted, resulting in ``num_levels`` values.
  In a second step, the ``inputs`` values are then provided again a composite
  value that is equal (for each) to a convex combination of those soft-sorted
  values using the transportation matrix. As the regularization parameter
  ``epsilon`` of regularized optimal transport goes to 0, this operator recovers
  the expected behavior of quantization, namely each value in ``inputs`` is
  assigned a single level. When using ``epsilon>0`` the bheaviour is similar but
  differentiable.

  Args:
    inputs: the inputs as a jnp.ndarray[batch, dim].
    num_levels: number of levels available to quantize the signal.
    axis: axis along which quantization is carried out.
    **kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in [0,1] (sigmoid of whitened values by default) to
      solve the optimal transport problem; ``cost_fn``, used in ``PointCloud``,
      that defines the ground cost function to transport from ``inputs`` to the
      ``num_targets`` target values (squared Euclidean distance by default, see
      ``pointcloud.py`` for more details); ``epsilon`` values as well as other
      parameters to shape the ``sinkhorn`` algorithm.

  Returns:
    A jnp.ndarray of the same size as ``inputs``.
  """
  return apply_on_axis(_quantize, inputs, axis, num_levels, **kwargs)
