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

# Lint as: python3
"""A class describing common operations for a cost."""
import itertools
from typing import Callable, Iterable, Optional, Sequence

import jax
import jax.numpy as np
import numpy as onp

from ott.core.ground_geometry import geometry


@jax.tree_util.register_pytree_node_class
class Grid(geometry.Geometry):
  """Class to implement separable separable cost on a grid as ground metric.

  This class implements a geometry in which probability measures are supported
  on a d-dimensional cartesian grid (a cartesian product of a list of d values).
  The transportation cost between points in the grid is assumed to be separable,
  namely a sum of coordinate-wise cost functions.

  In such a regime, and despite the fact that the number n of points in such
  grids is exponential in the dimension of the grid, applying a kernel in the
  context of regularized optimal transport can be carried out in time that is
  of the order of n^(1+1/d).
  """

  def __init__(
      self,
      x: Optional[Sequence[np.ndarray]] = None,
      grid_size: Optional[Sequence[int]] = None,
      cost_fns: Optional[Iterable[Callable[[float, float], float]]] = None,
      **kwargs):
    """Create instance of Euclidean cost to power p.

    Args:
      x : d arrays of varying sizes, locations of the grid
      grid_size: Sequence of integers describing grid sizes
      cost_fns: d functions taking two floats and outputting a float. If none
        the Euclidean distance is assumed by default
      **kwargs: passed to parent class.
    """
    super().__init__(None, None, **kwargs)
    if x is not None:
      self.x = x
      self.grid_size = [len(xs) for xs in self.x]
    elif grid_size is not None:
      self.grid_size = grid_size
      self.x = [np.arange(n) / np.maximum(1, n - 1) for n in grid_size]
    else:
      raise ValueError('Expected either grid locations or grid specifications')
    self._grid_dimension = len(self.x)
    if cost_fns is None:
      cost_fns = [lambda x, y: (x - y)**2]
    self.cost_fns = cost_fns
    # kwargs used in tree_flatten
    self.kwargs = {'x': x, 'grid_size': grid_size, 'cost_fns': cost_fns}
    self.kwargs.update(kwargs)

  @property
  def cost_matrices(self):
    # computes cost matrices along each dimension of the grid
    cost_matrices = []
    for dimension, cost_fn in itertools.zip_longest(
        range(self._grid_dimension), self.cost_fns,
        fillvalue=self.cost_fns[-1]):
      x_values = self.x[dimension]
      cost_matrix = jax.vmap(lambda x1: jax.vmap(lambda y1: cost_fn(x1, y1))  # pylint: disable=cell-var-from-loop
                             (x_values))(x_values)  # pylint: disable=cell-var-from-loop
      cost_matrices.append(cost_matrix)
    return cost_matrices

  @property
  def kernel_matrices(self):
    # computes kernel matrices from cost matrices grid
    kernel_matrices = []
    for cost_matrix in self.cost_matrices:
      kernel_matrices.append(np.exp(-cost_matrix/self.epsilon))
    return kernel_matrices

  @property
  def shape(self):
    num_a = onp.prod(self.grid_size)
    return num_a, num_a

  # TODO(lpapaxanthos): add properties for cost_matrix and kernel_matrix

  # Reimplemented functions to be used in regularized OT
  def apply_lse_kernel(self,
                       f: np.ndarray,
                       g: np.ndarray,
                       eps: float,
                       vec: np.ndarray = None,
                       axis: int = 0):
    """Applies grid kernel in log space. see parent for description."""
    # More implementation details in https://arxiv.org/pdf/1708.01955.pdf

    f, g = np.reshape(f, self.grid_size), np.reshape(g, self.grid_size)

    if vec is not None:
      vec = np.reshape(vec, self.grid_size)

    if axis == 0:
      f, g = g, f

    for dimension in range(self._grid_dimension):
      g, vec = self._apply_lse_kernel_one_dimension(dimension, f, g, eps, vec)
      g -= f
    if vec is None:
      vec = np.array(1.0)
    return g.ravel(), vec.ravel()

  def _apply_lse_kernel_one_dimension(self, dimension, f, g, eps, vec=None):
    indices = onp.arange(self._grid_dimension)
    indices[dimension], indices[0] = 0, dimension
    f, g = np.transpose(f, indices), np.transpose(g, indices)
    centered_cost = (f[:, np.newaxis, ...] + g[np.newaxis, ...]
                     - np.expand_dims(
                         self.cost_matrices[dimension],
                         axis=tuple(range(2, 1 + self._grid_dimension)))
                     ) / eps

    if vec is not None:
      vec = np.transpose(vec, indices)
      softmax_res, softmax_sgn = jax.scipy.special.logsumexp(
          centered_cost, b=vec, axis=1, return_sign=True)
      return eps * np.transpose(softmax_res, indices), np.transpose(
          softmax_sgn, indices)
    else:
      softmax_res = jax.scipy.special.logsumexp(centered_cost, axis=1)
      return eps * np.transpose(softmax_res, indices), None

  def apply_kernel(self, scaling: np.ndarray, eps=None, axis=None):
    """Applies kernel on grid using sub-kernel matrices on each slice."""
    scaling = np.reshape(scaling, self.grid_size)
    indices = list(range(1, self._grid_dimension))
    for dimension in range(self._grid_dimension):
      ind = indices.copy()
      ind.insert(dimension, 0)
      scaling = np.tensordot(
          self.kernel_matrices[dimension], scaling,
          axes=([0], [dimension])).transpose(ind)
    return scaling.ravel()

  # TODO(cuturi) this should be handled with care, as it will likely blow up
  def transport_from_potentials(self, f: np.ndarray, g: np.ndarray, axis=0):
    raise ValueError('Grid geometry cannot instantiate a transport matrix, use',
                     ' apply_transport_from_potentials(...).')

  def transport_from_scalings(self, f: np.ndarray, g: np.ndarray, axis=0):
    raise ValueError('Grid geometry cannot instantiate a transport matrix, use',
                     ' apply_transport_from_scalings(...)')

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates the geometries used for a divergence computation."""
    grid_size = kwargs.pop('grid_size', None)
    x = kwargs.pop('x', args)

    sep_grid = cls(x=x, grid_size=grid_size, **kwargs)
    size = 2 if static_b else 3
    return tuple(sep_grid for _ in range(size))

  def tree_flatten(self):
    return ((), self.kwargs)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    out = cls(*children, **aux_data) if children else cls(**aux_data)
    return  out
