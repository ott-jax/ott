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
import functools
from typing import Optional, Union

import jax
import jax.numpy as np


@jax.tree_util.register_pytree_node_class
class Epsilon:
  """A generic class to encapsulate the regularization parameter epsilon."""

  def __init__(self,
               target: float = 1e-2,
               init: float = 1.0,
               decay: float = 1.0):
    self.target = target
    self._init = init
    self._decay = decay

  def at(self, iteration: Optional[int] = 1) -> float:
    if iteration is None:
      return self.target
    init = np.where(self._decay < 1.0, self._init, self.target)
    decay = np.where(self._decay < 1.0, self._decay, 1.0)
    return np.maximum(init * decay**iteration, self.target)

  def done(self, eps):
    return eps == self.target

  def done_at(self, iteration):
    return self.done(self.at(iteration))

  def tree_flatten(self):
    return (self.target, self._init, self._decay), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)

  @classmethod
  def make(cls, *args, **kwargs):
    """Create or return a Epsilon instance."""
    if isinstance(args[0], cls):
      return args[0]
    else:
      return cls(*args, **kwargs)


@jax.tree_util.register_pytree_node_class
class Geometry:
  """Base class to define ground costs/kernels used in optimal transport."""

  def __init__(self,
               cost_matrix: Optional[np.ndarray] = None,
               kernel_matrix: Optional[np.ndarray] = None,
               epsilon: Union[Epsilon, float] = 1e-2,
               **kwargs
               ):
    """Initializes a simple ground cost.

    Args:
      cost_matrix: np.ndarray<float>[num_a, num_b]: a cost matrix storing n x m
        costs.
      kernel_matrix: np.ndarray<float>[num_a, num_b]: a kernel matrix storing
        n x m kernel values.
      epsilon: a regularization parameter.
      **kwargs: additional kwargs to epsilon.
    """
    self._cost_matrix = cost_matrix
    self._kernel_matrix = kernel_matrix
    self._epsilon = Epsilon.make(epsilon, **kwargs)

  @property
  def cost_matrix(self):
    return self._cost_matrix

  @property
  def kernel_matrix(self):
    if self._kernel_matrix is None:
      return np.exp(-(self._cost_matrix / self.epsilon))
    return self._kernel_matrix

  @property
  def epsilon(self):
    return self._epsilon.target

  @property
  def shape(self):
    mat = self.kernel_matrix if self.cost_matrix is None else self.cost_matrix
    return mat.shape

  # Generic functions to include geometry in Sinkhorn iterations. Particular
  # cases of geometries
  def apply_lse_kernel(self,
                       f: np.ndarray,
                       g: np.ndarray,
                       eps: float,
                       vec: np.ndarray = None,
                       axis: int = 0) -> np.ndarray:
    """Applies kernel in log domain on pair of dual potential variables.

    This function applies the ground geometry's kernel in log domain, using
    a stabilized formulation. At a high level, this iteration performs either

    output = eps * log (K exp(g/eps) )    (1)
    or
    output = eps * log (K' exp(f/eps) )   (2)

    K is implicitly exp(-cost_matrix/eps).

    To carry this out in a stabilized way, we take advantage of the fact that
    the entries of the matrix f[:,*] + g[*,:] - C are all negative, and
    therefore their exponential never overflows, to add and subtract f and g
    in iterations 1 & 2 respectively.

    Args:
      f: np.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: np.ndarray [num_b,] , potential of size num_cols of cost_matrix
      eps: float, regularization strength
      vec: np.ndarray [num_a or num_b,] , when not None, this has the effect
        of doing log-Kernel computations with an addition multiplication of
        each line/row by a vector. This is carried out by adding weights to
        the log-sum-exp
      axis: summing over axis 0 when doing (2), or over axis 1 when doing (1)
    Returns:
      A np.ndarray corresponding to output above, depending on axis.
    """
    w_res, w_sgn = self._softmax(f, g, eps, vec, axis)
    return w_res - (f if axis == 1 else g), w_sgn

  def apply_kernel(self, scaling: np.ndarray, eps=None, axis=0):
    """Applies kernel on positive scaling vector.

    This function applies the ground geometry's kernel, to perform either
    output = K v    (1)
    output = K'u   (2)
    where K is [num_a, num_b]

    Args:
      scaling: np.ndarray [num_a or num_b] , scaling of size num_rows
        or num_cols of kernel_matrix
      eps: passed for consistency, not used yet.
      axis: standard kernel product if axis is 1, transpose if 0.
    Returns:
      a np.ndarray corresponding to output above, depending on axis.
    """
    kernel = self.kernel_matrix if axis == 1 else self.kernel_matrix.T
    return np.dot(kernel, scaling)

  def marginal_from_potentials(self, f: np.ndarray,
                               g: np.ndarray,
                               axis: int = 0) -> np.ndarray:
    """Outputs marginal of transportation matrix from potentials.

    This applies first lse kernel in the standard way, removes the
    correction used to stabilise computations, and lifts this with an exp to
    recover either of the marginals corresponding to the transport map induced
    by potentials.
    Args:
      f: np.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: np.ndarray [num_b,] , potential of size num_cols of cost_matrix
      axis: axis along which to integrate, returns marginal on other axis.
    Returns:
      a vector of marginals of the transport matrix.
    """
    return np.exp((self.apply_lse_kernel(f, g, self.epsilon, axis=axis)[0] +
                   (f if axis == 1 else g)) / self.epsilon)

  def marginal_from_scalings(self, u: np.ndarray, v: np.ndarray, axis=0):
    """Outputs marginal of transportation matrix from scalings."""
    u, v = (v, u) if axis == 0 else (u, v)
    return u * self.apply_kernel(v, eps=self.epsilon, axis=axis)

  def transport_from_potentials(self, f, g):
    """Outputs transport matrix from potentials."""
    return np.exp(self._center(f, g) / self.epsilon)

  def transport_from_scalings(self, u, v):
    """Outputs transport matrix from pair of scalings."""
    return self.kernel_matrix * u[:, np.newaxis] * v[np.newaxis, :]

  # Functions that are not supposed to be changed by inherited classes.
  # These are the point of entry for Sinkhorn's algorithm to use a geometry.
  def error(self, f_u, g_v, target, iteration, axis=0, default_value=1.0,
            lse_mode: bool = True):
    if lse_mode:
      marginal = self.marginal_from_potentials(f_u, g_v, axis=axis)
    else:
      marginal = self.marginal_from_scalings(f_u, g_v, axis=axis)
    result = np.max(np.abs(marginal - target) / target, axis=None)
    return np.where(self._epsilon.done_at(iteration), result, default_value)

  def update_potential(self, f, g, log_marginal, iteration=None, axis=0):
    """Updates potentials in log space Sinkhorn iteration.

    Args:
      f: np.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: np.ndarray [num_b,] , potential of size num_cols of cost_matrix
      log_marginal: targeted marginal
      iteration: used to compute epsilon from schedule, if provided.
      axis: axis along which the update should be carried out.

    Returns:
      new potential value, g if axis=0, f if axis is 1.
    """
    eps = self._epsilon.at(iteration)
    return eps * log_marginal - self.apply_lse_kernel(f, g, eps, axis=axis)[0]

  def update_scaling(self, scaling, marginal, iteration=None, axis=0):
    eps = self._epsilon.at(iteration)
    return marginal / self.apply_kernel(scaling, eps, axis=axis)

  # Helper functions
  def _center(self, f: np.ndarray, g: np.ndarray):
    return f[:, np.newaxis] + g[np.newaxis, :] - self.cost_matrix

  def _softmax(self, f, g, eps, vec, axis):
    if vec is not None:
      if axis == 0:
        vec = vec.reshape((vec.size, 1))
      lse_output = jax.scipy.special.logsumexp(
          self._center(f, g) / eps, b=vec, axis=axis, return_sign=True)
      return eps * lse_output[0], lse_output[1]
    else:
      return eps * jax.scipy.special.logsumexp(
          self._center(f, g) / eps, b=vec, axis=axis), np.array(1.0)

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_potentials(self, f, g, vec, axis):
    """Applies lse_kernel while keeping track of signs."""
    lse_res, lse_sgn = self.apply_lse_kernel(
        f, g, self.epsilon, vec=vec, axis=axis)
    lse_res += f if axis == 1 else g
    return lse_sgn * np.exp(lse_res / self.epsilon)

  # wrapper to allow default option for axis.
  def apply_transport_from_potentials(self,
                                      f: np.ndarray,
                                      g: np.ndarray,
                                      vec: np.ndarray,
                                      axis: int = 0) -> np.ndarray:
    """Applies transport matrix computed from potentials to a (batched) vec.

    This approach does not instantiate the transport matrix itself.
    Computations are done in log space, and take advantage of the
    (b=, return_sign=True) parameters of logsumexp. Should be more stable
    than instantiating the transportation matrix from potentials and multiply.

    Args:
      f: np.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: np.ndarray [num_b,] , potential of size num_cols of cost_matrix
      vec: np.ndarray [batch, num_a or num_b], vector that will be multiplied
        by transport matrix corresponding to potentials f, g, and geom.
      axis: axis to differentiate left (0) or right (1) multiply.
    Returns:
      array of the size of vec.
    """
    return self._apply_transport_from_potentials(f, g, vec, axis)

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_scalings(self, u, v, vec, axis):
    u, v = (u, v * vec) if axis == 0 else (v, u * vec)
    return u * self.apply_kernel(v, axis=axis)

  # wrapper to allow default option for axis
  def apply_transport_from_scalings(self,
                                    u: np.ndarray,
                                    v: np.ndarray,
                                    vec: np.ndarray,
                                    axis: int = 0) -> np.ndarray:
    """Applies transport matrix computed from scalings to a (batched) vec.

    This approach does not instantiate the transport matrix itself.
    Args:
      u: np.ndarray [num_a,] , scaling of size num_rows of cost_matrix
      v: np.ndarray [num_b,] , scaling of size num_cols of cost_matrix
      vec: np.ndarray [batch, num_a or num_b], vector that will be multiplied
        by transport matrix corresponding to scalings u, v, and geom.
      axis: axis to differentiate left (0) or right (1) multiply.
    Returns:
      array of the size of vec.
    """
    return self._apply_transport_from_scalings(u, v, vec, axis)

  def potential_from_scaling(self, scaling: np.ndarray) -> np.ndarray:
    return self.epsilon * np.log(scaling)

  def scaling_from_potential(self, potential: np.ndarray) -> np.ndarray:
    return np.exp(potential / self.epsilon)

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates the geometries used for a divergence computation."""
    size = 2 if static_b else 3
    nones = [None, None, None]
    kernel_matrices = kwargs.pop('kernel_matrix', nones)
    cost_matrices = kwargs.pop('cost_matrix', args)
    cost_matrices = cost_matrices if cost_matrices is not None else nones
    return tuple(
        cls(cost_matrix=arg1, kernel_matrix=arg2, **kwargs)
        for arg1, arg2, _ in zip(cost_matrices, kernel_matrices, range(size))
    )

  def tree_flatten(self):
    return (self.cost_matrix, self.kernel_matrix, self._epsilon), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)
