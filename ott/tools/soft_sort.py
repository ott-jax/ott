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

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ott.core import sinkhorn
from ott.geometry import pointcloud


def sinkhorn_for_sort(inputs: jnp.ndarray,
                      weights: jnp.ndarray,
                      target_weights: jnp.ndarray,
                      sinkhorn_kw,
                      pointcloud_kw) -> jnp.ndarray:
  """Runs sinkhorn on a fixed increasing target.

  Args:
    inputs: jnp.ndarray[num_points]. Must be one dimensional.
    weights: jnp.ndarray[num_points]. The weights 'a' for the inputs.
    target_weights: jnp.ndarray[num_targets]: the weights of the targets. It may
      be of a different size than the weights.
    sinkhorn_kw: a dictionary holding the sinkhorn keyword arguments. See
      sinkhorn.py for more details.
    pointcloud_kw: a dictionary holding the keyword arguments of the
      PointCloud class. See pointcloud.py for more details.

  Returns:
    A jnp.ndarray<float> representing the transport matrix of the inputs onto
    the underlying sorted target.
  """
  shape = inputs.shape
  if len(shape) > 2 or (len(shape) == 2 and shape[1] != 1):
    raise ValueError(
        "Shape ({shape}) not supported. The input should be one-dimensional.")

  x = jnp.expand_dims(jnp.squeeze(inputs), axis=1)
  x = jax.nn.sigmoid((x - jnp.mean(x)) / (jnp.std(x) + 1e-10))
  a = jnp.squeeze(weights)
  b = jnp.squeeze(target_weights)
  num_targets = b.shape[0]
  y = jnp.linspace(0.0, 1.0, num_targets)[:, jnp.newaxis]
  geom = pointcloud.PointCloud(x, y, **pointcloud_kw)
  res = sinkhorn.sinkhorn(geom, a, b, **sinkhorn_kw)
  return geom.transport_from_potentials(res.f, res.g)


def apply_on_axis(op, inputs, axis, *args):
  """Applies a differentiable operator on a given axis of the input.

  Args:
    op: a differentiable operator (can be softranks, softquantiles, etc.)
    inputs: jnp.ndarray<float> of any shape.
    axis: the axis on which to apply the operator.
    *args: other positional arguments to the operator.

  Returns:
    A jnp.ndarray holding the output of the differentiable operator on the given
    axis.
  """
  original_shape = inputs.shape
  num_points = original_shape[axis]
  permutation = np.arange(len(original_shape))
  permutation[axis], permutation[-1] = permutation[-1], permutation[axis]
  inputs = jnp.transpose(inputs, permutation)

  batch_fn = jax.vmap(op, in_axes=(0,) + (None,) * len(args))
  result = batch_fn(jnp.reshape(inputs, (-1, num_points)), *args)
  result = jnp.reshape(result, inputs.shape[:-1] + result.shape[-1:])
  result = jnp.transpose(result, permutation)
  return result


def _softsort(inputs: jnp.ndarray, sinkhorn_kw, kwargs) -> jnp.ndarray:
  """Applies the soft sort operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  b = jnp.ones((num_points,)) / num_points
  soft_permutation = sinkhorn_for_sort(inputs, a, b, sinkhorn_kw, kwargs)
  out = 1.0 / b * jnp.matmul(soft_permutation.T, jnp.squeeze(inputs))
  return jnp.reshape(out, inputs.shape)


def softsort(inputs: jnp.ndarray,
             axis: int = -1,
             sinkhorn_kw=None,
             **kwargs) -> jnp.ndarray:
  """Applies the soft sort operator on a given axis of the input.

  Args:
    inputs: jnp.ndarray<float> of any shape.
    axis: the axis on which to apply the operator.
    sinkhorn_kw: a dictionary holding the sinkhorn keyword arguments. See
      sinkhorn.py for more details.
    **kwargs: keyword arguments of the PointCloud class. See
      pointcloud.py for more details.

  Returns:
    A jnp.ndarray of the same shape as the input with soft sorted values on the
    given axis.
  """
  sinkhorn_kw = {} if sinkhorn_kw is None else sinkhorn_kw
  return apply_on_axis(_softsort, inputs, axis, sinkhorn_kw, kwargs)


def _softranks(inputs: jnp.ndarray, sinkhorn_kw, kwargs) -> jnp.ndarray:
  """Applies the soft ranks operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  b = jnp.ones((num_points,)) / num_points
  soft_permutation = sinkhorn_for_sort(inputs, a, b, sinkhorn_kw, kwargs)
  out = 1.0 / a * jnp.matmul(soft_permutation, jnp.arange(num_points))
  return jnp.reshape(out, inputs.shape)


def softranks(inputs: jnp.ndarray,
              axis=-1,
              sinkhorn_kw=None,
              **kwargs) -> jnp.ndarray:
  """Applies the softrank operator on input tensor.

  Args:
    inputs: a jnp.ndarray<float> of any shape.
    axis: the axis on which to apply the soft ranks operator.
    sinkhorn_kw: keyword argument to the sinkhorn routine.
    **kwargs: extra arguments to the underlying EuclideanGeometry.

  Returns:
    A jnp.ndarray<float> of the same shape as inputs, with the ranks.
  """
  sinkhorn_kw = {} if sinkhorn_kw is None else sinkhorn_kw
  return apply_on_axis(_softranks, inputs, axis, sinkhorn_kw, kwargs)


def _softquantile(inputs: jnp.ndarray,
                  level: float,
                  weight: float,
                  sinkhorn_kw: Dict[str, Any],
                  kwargs: Dict[str, Any]
                  ) -> jnp.ndarray:
  """Applies the soft quantile operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  b = jnp.array([level - weight / 2, weight, 1.0 - weight / 2 - level])
  soft_permutation = sinkhorn_for_sort(inputs, a, b, sinkhorn_kw, kwargs)
  out = 1.0 / b * jnp.matmul(soft_permutation.T, jnp.squeeze(inputs))
  return out[1:2]


def softquantile(inputs: jnp.ndarray,
                 axis: int = -1,
                 level: float = 0.5,
                 weight: float = 0.05,
                 sinkhorn_kw=None,
                 **kwargs) -> jnp.ndarray:
  """Applies the softquantile operator on input tensor.

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
   sinkhorn_kw: keyword argument to the sinkhorn routine.
   **kwargs: extra arguments to the underlying EuclideanGeometry.

  Returns:
    A jnp.ndarray, which has the same shape as the input, except on the give
    axis on which the dimension is 1.
  """
  sinkhorn_kw = {} if sinkhorn_kw is None else sinkhorn_kw
  return apply_on_axis(
      _softquantile, inputs, axis, level, weight, sinkhorn_kw, kwargs)
