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

# Lint as: python3
"""A class describing low-rank geometries."""
from typing import Union, Optional
import jax
import jax.numpy as jnp
from ott.geometry import geometry


@jax.tree_util.register_pytree_node_class
class LRCGeometry(geometry.Geometry):
  r"""Low-rank Cost Geometry defined by two factors.
  """

  def __init__(self,
               cost_1: jnp.ndarray,
               cost_2: jnp.ndarray,
               bias: float = 0.0,
               scale_cost: Optional[Union[bool, float, str]] = None,
               batch_size: Optional[int] = None,
               **kwargs
               ):
    r"""Initializes a geometry by passing it low-rank factors.

    Args:
      cost_1: jnp.ndarray<float>[num_a, r]
      cost_2: jnp.ndarray<float>[num_b, r]
      bias: constant added to entire cost matrix.
      scale_cost: option to rescale the cost matrix. Implemented scalings are
        'max_bound', 'mean' and 'max_cost'. Alternatively, a float
        factor can be given to rescale the cost such that
        ``cost_matrix /= scale_cost``. If `True`, use 'mean'.
      batch_size: optional size of the batch to compute online (without
        instantiating the matrix) the scale factor ``scale_cost`` of the
        ``cost_matrix`` when ``scale_cost=max_cost``. If set to ``None``, the
        batch size is set to 1024 or to the largest number of samples between
        ``cost_1`` and ``cost_2`` if smaller than 1024.
      **kwargs: additional kwargs to Geometry
    """
    assert cost_1.shape[1] == cost_2.shape[1]
    self._cost_1 = cost_1
    self._cost_2 = cost_2
    self._bias = bias
    self._kwargs = kwargs

    super().__init__(**kwargs)
    self._scale_cost = "mean" if scale_cost is True else scale_cost
    self.batch_size = batch_size

  @property
  def cost_1(self):
    return self._cost_1 * jnp.sqrt(self.scale_cost)

  @property
  def cost_2(self):
    return self._cost_2 * jnp.sqrt(self.scale_cost)

  @property
  def bias(self):
    return self._bias * self.scale_cost

  @property
  def cost_rank(self):
    return self._cost_1.shape[1]

  @property
  def cost_matrix(self):
    """Returns cost matrix if requested."""
    return jnp.matmul(self.cost_1, self.cost_2.T) + self.bias

  @property
  def shape(self):
    return (self._cost_1.shape[0], self._cost_2.shape[0])

  @property
  def is_symmetric(self):
    return (self._cost_1.shape[0] == self._cost_2.shape[0] and
            jnp.all(self._cost_1 == self._cost_2))

  @property
  def scale_cost(self):
    if isinstance(self._scale_cost, float):
      return 1.0 / self._scale_cost
    elif self._scale_cost == 'max_bound':
      x_norm = self._cost_1[:, 0].max()
      y_norm = self._cost_2[:, 1].max()
      max_bound = x_norm + y_norm + 2 * jnp.sqrt(
        x_norm * y_norm
      )
      return jax.lax.stop_gradient(1.0 / (max_bound + self._bias))
    elif self._scale_cost == 'mean':
      factor1 = jnp.dot(jnp.ones(self.shape[0]), self._cost_1)
      factor2 = jnp.dot(self._cost_2.T, jnp.ones(self.shape[1]))
      mean = (jnp.dot(factor1, factor2) / (self.shape[0] * self.shape[1])
              + self._bias)
      return jax.lax.stop_gradient(1.0 / mean)
    elif self._scale_cost == 'max_cost':
      return jax.lax.stop_gradient(1.0 / self.compute_max_cost())
    elif isinstance(self._scale_cost, str):
      raise ValueError(f'Scaling {self._scale_cost} not provided.')
    else:
      return 1.0

  def apply_square_cost(self, arr: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Applies elementwise-square of cost matrix to array (vector or matrix)."""
    (n, m), r = self.shape, self.cost_rank
    # When applying square of a LRCgeometry, one can either elementwise square
    # the cost matrix, or instantiate an augmented (rank^2) LRCGeometry
    # and apply it. First is O(nm), the other is O((n+m)r^2).
    if n * m < (n + m) * r**2:  #  better use regular apply
      return super().apply_square_cost(arr, axis)
    else:
      new_cost_1 = self.cost_1[:, :, None] * self.cost_1[:, None, :]
      new_cost_2 = self.cost_2[:, :, None] * self.cost_2[:, None, :]
      return LRCGeometry(
          cost_1=new_cost_1.reshape((n, r**2)),
          cost_2=new_cost_2.reshape((m, r**2))).apply_cost(arr, axis)

  def _apply_cost_to_vec(self,
                         vec: jnp.ndarray,
                         axis: int = 0,
                         fn=None) -> jnp.ndarray:
    """Applies [num_a, num_b] fn(cost) (or transpose) to vector.

    Args:
      vec: jnp.ndarray [num_a,] ([num_b,] if axis=1) vector
      axis: axis on which the reduction is done.
      fn: function optionally applied to cost matrix element-wise, before the
        doc product

    Returns:
      A jnp.ndarray corresponding to cost x vector
    """
    def efficient_apply(vec, axis, fn):
      c1 = self.cost_1 if axis == 1 else self.cost_2
      c2 = self.cost_2 if axis == 1 else self.cost_1
      c2 = fn(c2) if fn is not None else c2
      bias = fn(self.bias) if fn is not None else self.bias
      out = jnp.dot(c1, jnp.dot(c2.T, vec))
      return out + bias * jnp.sum(vec) * jnp.ones_like(out)

    return jax.lax.cond(
      fn is None or geometry.is_linear(fn),
      lambda _: efficient_apply(vec, axis, fn),
      lambda obj: super(obj.__class__, obj)._apply_cost_to_vec(vec, axis, fn),
      self
    )

  def compute_max_cost(self) -> float:
    """Computes the maximum of the cost matrix.

    Three cases are taken into account:
    - If the number of samples of ``cost_1`` and ``cost_2`` are both smaller
    than 1024 and if ``batch_size`` is ``None``, the ``cost_matrix`` is
    computed to obtain its maximum entry.
    - If one of the number of samples of ``cost_1`` or ``cost_2`` is larger
    than 1024 and if ``batch_size`` is ``None``, then the maximum of the
    cost matrix is calculated by batch. The batches are created on the longest
    axis of the cost matrix and their size is fixed to 1024.
    - If ``batch_size`` is provided as a float, then the maximum of the cost
    matrix is calculated by batch. The batches are created on the longest axis
    of the cost matrix and their size if fixed by ``batch_size``.

    Returns:
      Maximum of the cost matrix.
    """
    batch_for_y = self.shape[1] > self.shape[0]

    n = self.shape[1] if batch_for_y else self.shape[0]
    p = self._cost_2.shape[1] if batch_for_y else self._cost_1.shape[1]
    carry = ((self._cost_1, self._cost_2) if batch_for_y
             else (self._cost_2, self._cost_1))

    if self.batch_size:
      batch_size = min(self.batch_size, n)
    else:
      batch_size = min(1024, max(self.shape[0], self.shape[1]))
    n_batch = n // batch_size

    def body(carry, slice_idx):
      cost1, cost2 = carry
      cost2_slice = jax.lax.dynamic_slice(
          cost2, (slice_idx * batch_size, 0), (batch_size, p))
      out_slice = jnp.max(jnp.dot(cost2_slice, cost1.T))
      return carry, out_slice

    def finalize(carry):
      cost1, cost2 = carry
      out_slice = jnp.dot(cost2[n_batch * batch_size:], cost1.T)
      return out_slice

    _, out = jax.lax.scan(body, carry, jnp.arange(n_batch))
    last_slice = finalize(carry)
    max_value = jnp.max(jnp.concatenate((out, last_slice.reshape(-1))))
    return max_value + self._bias

  def tree_flatten(self):
    return (self._cost_1, self._cost_2, self._kwargs), {
        'bias': self._bias, 'scale_cost': self._scale_cost,
        'batch_size': self.batch_size}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children[:-1], **children[-1], **aux_data)


def add_lrc_geom(geom1: LRCGeometry, geom2: LRCGeometry):
  """Add geometry in geom1 to that in geom2, keeping other geom1 params."""
  return LRCGeometry(
      cost_1=jnp.concatenate((geom1.cost_1, geom2.cost_1), axis=1),
      cost_2=jnp.concatenate((geom1.cost_2, geom2.cost_2), axis=1),
      **geom1._kwargs)
