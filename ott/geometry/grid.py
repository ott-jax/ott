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
"""Implements a geometry class for points supported on a cartesian product."""
import itertools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.geometry import geometry
from ott.geometry import ops


@jax.tree_util.register_pytree_node_class
class Grid(geometry.Geometry):
  r"""Class describing the geometry of points taken in a cartestian product.

  This class implements a geometry in which probability measures are supported
  on a :math:`d`-dimensional cartesian grid, a cartesian product of :math:`d`
  lists of values, each list being itself of size :math:`n_i`.

  The transportation cost between points in the grid is assumed to be separable,
  namely a sum of coordinate-wise cost functions, as in

  .. math::

    cost(x,y) = \sum_{i=1}^d cost_i(x_i, y_i)

  where :math:`cost_i`: R x R → R.

  In such a regime, and despite the fact that the total number :math:`n_{total}`
  of points in the grid is exponential :math:`d` (namely :math:`\prod_i n_i`),
  applying a kernel in the context of regularized optimal transport can be
  carried out in time that is of the order of :math:`n_{total}^{(1+1/d)}` using
  convolutions, either in the original domain or log-space domain. This class
  precomputes :math:`d` :math:`n_i` x :math:`n_i` cost matrices (one per
  dimension) and implements these two operations by carrying out these
  convolutions one dimension at a time.
  """

  def __init__(
      self,
      x: Optional[Sequence[jnp.ndarray]] = None,
      grid_size: Optional[Sequence[int]] = None,
      cost_fns: Optional[Sequence[costs.CostFn]] = None,
      num_a: Optional[int] = None,
      grid_dimension: Optional[int] = None,
      **kwargs):
    """Create instance of grid using either locations or sizes.

    Args:
      x : list of arrays of varying sizes, describing the locations of the grid.
        Locations are provided as a list of jnp.ndarrays, that is :math:`d`
        vectors of (possibly varying) size :math:`n_i`. The resulting grid
        is the Cartesian product of these vectors.
      grid_size: tuple of integers describing grid sizes, namely
        :math:`(n_1,...,n_d)`. This will only be used if x is None.
        In that case the grid will be assumed to lie in the hypercube
        :math:`[0,1]^d`, with the :math:`d` dimensions, described as points
        regularly sampled in [0,1].
      cost_fns: a sequence of :math:`d` costs.CostFn's, each being a cost taking
        two reals as inputs to output a real number.
      num_a: total size of grid. This parameters will be computed from other
        inputs and used in the flatten/unflatten functions.
      grid_dimension: dimension of grid. This parameters will be computed from
        other inputs and used in the flatten/unflatten functions.
      **kwargs: other optional parameters to be passed on to superclass
        initializer, notably those related to epsilon regularization.

    """
    if (grid_size is not None and x is not None and num_a is not None and
        grid_dimension is not None):
      self.grid_size = tuple(map(int, grid_size))
      self.x = x
      self.num_a = num_a
      self.grid_dimension = grid_dimension
    elif x is not None:
      self.x = x
      self.grid_size = tuple([xs.shape[0] for xs in x])
      self.num_a = np.prod(np.array(self.grid_size))
      self.grid_dimension = len(self.x)
    elif grid_size is not None:
      self.grid_size = tuple(map(int, grid_size))
      self.x = tuple([jnp.linspace(0, 1, n) for n in self.grid_size])
      self.num_a = np.prod(np.array(grid_size))
      self.grid_dimension = len(self.grid_size)
    else:
      raise ValueError('Input either grid_size t-uple or grid locations x.')

    if cost_fns is None:
      cost_fns = [costs.Euclidean()]
    self.cost_fns = cost_fns
    self.kwargs = {'num_a': self.num_a, 'grid_size': self.grid_size,
                   'grid_dimension': self.grid_dimension}

    super().__init__(**kwargs)

  @property
  def cost_matrices(self):
    # computes cost matrices along each dimension of the grid
    cost_matrices = []
    for dimension, cost_fn in itertools.zip_longest(
        range(self.grid_dimension), self.cost_fns,
        fillvalue=self.cost_fns[-1]):
      x_values = self.x[dimension][:, jnp.newaxis]
      cost_matrix = jax.vmap(lambda x1: jax.vmap(lambda y1: cost_fn(x1, y1))  # pylint: disable=cell-var-from-loop
                             (x_values))(x_values)  # pylint: disable=cell-var-from-loop
      cost_matrices.append(cost_matrix)
    return cost_matrices

  @property
  def kernel_matrices(self):
    # computes kernel matrices from cost matrices grid
    kernel_matrices = []
    for cost_matrix in self.cost_matrices:
      kernel_matrices.append(jnp.exp(-cost_matrix / self.epsilon))
    return kernel_matrices

  @property
  def median_cost_matrix(self):
    raise NotImplementedError('Median cost not implemented for grids')

  @property
  def shape(self):
    return self.num_a, self.num_a

  @property
  def is_symmetric(self):
    return True

  # Reimplemented functions to be used in regularized OT
  def apply_lse_kernel(self,
                       f: jnp.ndarray,
                       g: jnp.ndarray,
                       eps: float,
                       vec: Optional[jnp.ndarray] = None,
                       axis: int = 0):
    """Applies grid kernel in log space. See notes in parent class for use case.

    Reshapes vector inputs below as grids, applies kernels onto each slice, and
    then expands the outputs as vectors.

    More implementation details in https://arxiv.org/pdf/1708.01955.pdf

    Args:
      f: jnp.ndarray, a vector of potentials
      g: jnp.ndarray, a vector of potentials
      eps: float, regularization strength
      vec: jnp.ndarray, if needed, a vector onto which apply the kernel weighted
        by f and g.
      axis: axis (0 or 1) along which summation should be carried out.

    Returns:
      a vector, the result of kernel applied in lse space onto vec.
    """
    f, g = jnp.reshape(f, self.grid_size), jnp.reshape(g, self.grid_size)

    if vec is not None:
      vec = jnp.reshape(vec, self.grid_size)

    if axis == 0:
      f, g = g, f

    for dimension in range(self.grid_dimension):
      g, vec = self._apply_lse_kernel_one_dimension(dimension, f, g, eps, vec)
      g -= jnp.where(jnp.isfinite(f), f, 0)
    if vec is None:
      vec = jnp.array(1.0)
    return g.ravel(), vec.ravel()

  def _apply_lse_kernel_one_dimension(self, dimension, f, g, eps, vec=None):
    """Helper function to permute axis & apply the kernel on a single slice."""
    indices = np.arange(self.grid_dimension)
    indices[dimension], indices[0] = 0, dimension
    f, g = jnp.transpose(f, indices), jnp.transpose(g, indices)
    centered_cost = (f[:, jnp.newaxis, ...] + g[jnp.newaxis, ...]
                     - jnp.expand_dims(
                         self.cost_matrices[dimension],
                         axis=tuple(range(2, 1 + self.grid_dimension)))
                     ) / eps

    if vec is not None:
      vec = jnp.transpose(vec, indices)
      softmax_res, softmax_sgn = ops.logsumexp(
          centered_cost, b=vec, axis=1, return_sign=True)
      return eps * jnp.transpose(softmax_res, indices), jnp.transpose(
          softmax_sgn, indices)
    else:
      softmax_res = ops.logsumexp(centered_cost, axis=1)
      return eps * jnp.transpose(softmax_res, indices), None

  def _apply_cost_to_vec(self,
                         vec: jnp.ndarray,
                         axis: int = 0,
                         fn=None) -> jnp.ndarray:
    r"""Applies grid's cost matrix (without instantiating it) to a vector.

    The `apply_cost` operation on grids rests on the following identity.
    If it were to be cast as a [num_a, num_a] matrix, the corresponding cost
    matrix :math:`C` would be a sum of grid_dimension matrices, each of the form
    (here for the j-th slice)
    :math:`\tilde{C}_j : = 1_{n_1} \otimes \dots \otimes C_j \otimes 1_{n_d}`
    where each :math:`1_{n}` is the :math:`n\times n` square matrix full of 1's.

    Applying :math:`\tilde{C}_j` on a vector grid consists in carrying a tensor
    multiplication on the dimension of that vector reshaped as a grid, followed
    by a summation on all other axis while keeping dimensions. That identity is
    a generalization of the formula
    :math:`(1_{n} \otimes A) vec(X) = vec( A X 1_n)`
    where the last multiplication by the matrix of ones is equivalent to
    summation while keeping dimensions.

    Args:
      vec: jnp.ndarray, flat vector of total size prod(grid_size).
      axis: axis 0 if applying transpose costs, 1 if using the original cost.
      fn: function optionally applied to cost matrix element-wise, before the
        dot product.

    Returns:
      A jnp.ndarray corresponding to cost x matrix
    """
    vec = jnp.reshape(vec, self.grid_size)
    accum_vec = jnp.zeros_like(vec)
    indices = list(range(1, self.grid_dimension))
    for dimension, cost in enumerate(self.cost_matrices):
      ind = indices.copy()
      ind.insert(dimension, 0)
      if axis == 0:
        cost = cost.T
      accum_vec += jnp.sum(
          jnp.tensordot(cost, vec, axes=([0], [dimension])),
          axis=indices, keepdims=True).transpose(ind)
    return accum_vec.ravel()

  def apply_kernel(self,
                   scaling: jnp.ndarray,
                   eps: Optional[float] = None,
                   axis: Optional[int] = None):
    """Applies grid kernel on scaling vector.

    See notes in parent class for use.

    Reshapes scaling vector as a grid, applies kernels onto each slice, and
    then ravels backs the output as a vector.

    More implementation details in https://arxiv.org/pdf/1708.01955.pdf

    Args:
      scaling: jnp.ndarray, a vector of scaling (>0) values.
      eps: float, regularization strength
      axis: axis (0 or 1) along which summation should be carried out.

    Returns:
      a vector, the result of kernel applied onto scaling.
    """
    scaling = jnp.reshape(scaling, self.grid_size)
    indices = list(range(1, self.grid_dimension))
    for dimension, kernel in enumerate(self.kernel_matrices):
      ind = indices.copy()
      ind.insert(dimension, 0)
      kernel = kernel if eps is None else kernel ** (self.epsilon / eps)
      scaling = jnp.tensordot(kernel, scaling,
                              axes=([0], [dimension])).transpose(ind)
    return scaling.ravel()

  def transport_from_potentials(self, f: jnp.ndarray, g: jnp.ndarray, axis=0):
    raise ValueError('Grid geometry cannot instantiate a transport matrix, use',
                     ' apply_transport_from_potentials(...) if you wish to ',
                     ' apply the transport matrix to a vector, or use a point '
                     ' cloud geometry instead')

  def transport_from_scalings(self, f: jnp.ndarray, g: jnp.ndarray, axis=0):
    raise ValueError('Grid geometry cannot instantiate a transport matrix, use',
                     ' apply_transport_from_scalings(...) if you wish to ',
                     ' apply the transport matrix to a vector, or use a point '
                     ' cloud geometry instead')

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates the geometries used for a divergence computation."""
    grid_size = kwargs.pop('grid_size', None)
    x = kwargs.pop('x', args)

    sep_grid = cls(x=x, grid_size=grid_size, **kwargs)
    size = 2 if static_b else 3
    return tuple(sep_grid for _ in range(size))

  def tree_flatten(self):
    return (self.x, self.cost_fns, self._epsilon), self.kwargs

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(
        x=children[0], cost_fns=children[1], epsilon=children[2], **aux_data)
