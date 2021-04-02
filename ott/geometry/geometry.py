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
"""A class describing operations used to instantiate and use a geometry."""
import functools
from typing import Optional, Union, Sequence

import jax
import jax.numpy as jnp
from ott.geometry import epsilon_scheduler


@jax.tree_util.register_pytree_node_class
class Geometry:
  """Base class to define ground costs/kernels used in optimal transport."""

  def __init__(self,
               cost_matrix: Optional[jnp.ndarray] = None,
               kernel_matrix: Optional[jnp.ndarray] = None,
               epsilon: Union[epsilon_scheduler.Epsilon, float] = 1e-2,
               **kwargs
               ):
    """Initializes a geometry by passing it a cost matrix or a kernel matrix.

    Args:
      cost_matrix: jnp.ndarray<float>[num_a, num_b]: a cost matrix storing n x m
        costs.
      kernel_matrix: jnp.ndarray<float>[num_a, num_b]: a kernel matrix storing
        n x m kernel values.
      epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
      **kwargs: additional kwargs to epsilon.
    """
    self._cost_matrix = cost_matrix
    self._kernel_matrix = kernel_matrix
    self._epsilon = epsilon_scheduler.Epsilon.make(epsilon, **kwargs)

  @property
  def cost_matrix(self):
    if self._cost_matrix is None:
      return -self.epsilon * jnp.log(self._kernel_matrix)
    return self._cost_matrix

  @property
  def median_cost_matrix(self):
    return jnp.median(self.cost_matrix[:])

  @property
  def kernel_matrix(self):
    if self._kernel_matrix is None:
      return jnp.exp(-(self._cost_matrix / self.epsilon))
    return self._kernel_matrix

  @property
  def epsilon(self):
    return self._epsilon.target

  @property
  def shape(self):
    mat = self.kernel_matrix if self.cost_matrix is None else self.cost_matrix
    return mat.shape if mat is not None else None

  @property
  def is_symmetric(self):
    mat = self.kernel_matrix if self.cost_matrix is None else self.cost_matrix
    return (mat.shape[0] == mat.shape[1] and jnp.all(mat == mat.T)
            ) if mat is not None else False

  # The functions below are at the core of Sinkhorn iterations.
  def apply_lse_kernel(self,
                       f: jnp.ndarray,
                       g: jnp.ndarray,
                       eps: float,
                       vec: jnp.ndarray = None,
                       axis: int = 0) -> jnp.ndarray:
    """Applies kernel in log domain on pair of dual potential variables.

    This function applies the ground geometry's kernel in log domain, using
    a stabilized formulation. At a high level, this iteration performs either

    output = eps * log (K (exp(g / eps) * vec) )    (1)
    or
    output = eps * log (K'(exp(f / eps) * vec))   (2)

    K is implicitly exp(-cost_matrix/eps).

    To carry this out in a stabilized way, we take advantage of the fact that
    the entries of the matrix f[:,*] + g[*,:] - C are all negative, and
    therefore their exponential never overflows, to add (and subtract after)
    f and g in iterations 1 & 2 respectively.

    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      eps: float, regularization strength
      vec: jnp.ndarray [num_a or num_b,] , when not None, this has the effect
        of doing log-Kernel computations with an addition elementwise
        multiplication of exp(g /eps) by a vector. This is carried out by
        adding weights to the log-sum-exp function, and needs to handle signs
        separately.
      axis: summing over axis 0 when doing (2), or over axis 1 when doing (1)
    Returns:
      A jnp.ndarray corresponding to output above, depending on axis.
    """
    w_res, w_sgn = self._softmax(f, g, eps, vec, axis)
    remove = f if axis == 1 else g
    return w_res - jnp.where(jnp.isfinite(remove), remove, 0), w_sgn

  def apply_kernel(self, scaling: jnp.ndarray, eps=None, axis=0):
    """Applies kernel on positive scaling vector.

    This function applies the ground geometry's kernel, to perform either
    output = K v    (1)
    output = K'u   (2)
    where K is [num_a, num_b]

    Args:
      scaling: jnp.ndarray [num_a or num_b] , scaling of size num_rows
        or num_cols of kernel_matrix
      eps: passed for consistency, not used yet.
      axis: standard kernel product if axis is 1, transpose if 0.
    Returns:
      a jnp.ndarray corresponding to output above, depending on axis.
    """
    del eps
    kernel = self.kernel_matrix if axis == 1 else self.kernel_matrix.T
    return jnp.dot(kernel, scaling)

  def marginal_from_potentials(self, f: jnp.ndarray,
                               g: jnp.ndarray,
                               axis: int = 0) -> jnp.ndarray:
    """Outputs marginal of transportation matrix from potentials.

    This applies first lse kernel in the standard way, removes the
    correction used to stabilise computations, and lifts this with an exp to
    recover either of the marginals corresponding to the transport map induced
    by potentials.
    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      axis: axis along which to integrate, returns marginal on other axis.
    Returns:
      a vector of marginals of the transport matrix.
    """
    h = (f if axis == 1 else g)
    z = self.apply_lse_kernel(f, g, self.epsilon, axis=axis)[0]
    return jnp.where(jnp.isfinite(h),
                     jnp.exp((z + h)/self.epsilon),
                     0)

  def marginal_from_scalings(self, u: jnp.ndarray, v: jnp.ndarray, axis=0):
    """Outputs marginal of transportation matrix from scalings."""
    u, v = (v, u) if axis == 0 else (u, v)
    return u * self.apply_kernel(v, eps=self.epsilon, axis=axis)

  def transport_from_potentials(self, f, g):
    """Outputs transport matrix from potentials."""
    return jnp.exp(self._center(f, g) / self.epsilon)

  def transport_from_scalings(self, u, v):
    """Outputs transport matrix from pair of scalings."""
    return self.kernel_matrix * u[:, jnp.newaxis] * v[jnp.newaxis, :]

  # Functions that are not supposed to be changed by inherited classes.
  # These are the point of entry for Sinkhorn's algorithm to use a geometry.
  def error(self,
            f_u: jnp.ndarray,
            g_v: jnp.ndarray,
            target: jnp.ndarray,
            axis: int = 0,
            norm_error: Sequence[int] = (1,),
            lse_mode: bool = True):
    """Outputs error, using 2 potentials/scalings, of transport w.r.t target marginal.

    Args:
      f_u: a vector of potentials or scalings for the first marginal.
      g_v: a vector of potentials or scalings for the second marginal.
      target: target marginal.
      axis: axis (0 or 1) along which to compute marginal.
      norm_error: (t-uple of int) p's to compute p-norm between marginal/target
      lse_mode: whether operating on scalings or potentials
    Returns:
      t-uple of floats, quantifying difference between target / marginal.
    """
    if lse_mode:
      marginal = self.marginal_from_potentials(f_u, g_v, axis=axis)
    else:
      marginal = self.marginal_from_scalings(f_u, g_v, axis=axis)
    norm_error = jnp.array(norm_error)
    error = jnp.sum(
        jnp.abs(marginal - target) ** norm_error[:, jnp.newaxis],
        axis=1) ** (1.0 / norm_error)
    return error

  def update_potential(self, f, g, log_marginal, iteration=None, axis=0):
    """Carries out one Sinkhorn update for potentials, i.e. in log space.

    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      log_marginal: targeted marginal
      iteration: used to compute epsilon from schedule, if provided.
      axis: axis along which the update should be carried out.

    Returns:
      new potential value, g if axis=0, f if axis is 1.
    """
    eps = self._epsilon.at(iteration)
    app_lse = self.apply_lse_kernel(f, g, eps, axis=axis)[0]
    return eps * log_marginal - jnp.where(jnp.isfinite(app_lse), app_lse, 0)

  def update_scaling(self, scaling, marginal, iteration=None, axis=0):
    """Carries out one Sinkhorn update for scalings, using kernel directly.

    Args:
      scaling: jnp.ndarray of num_a or num_b positive values.
      marginal: targeted marginal
      iteration: used to compute epsilon from schedule, if provided.
      axis: axis along which the update should be carried out.

    Returns:
      new scaling vector, of size num_b if axis=0, num_a if axis is 1.
    """

    eps = self._epsilon.at(iteration)
    app_kernel = self.apply_kernel(scaling, eps, axis=axis)
    return marginal / jnp.where(app_kernel > 0, app_kernel, 1.0)

  # Helper functions
  def _center(self, f: jnp.ndarray, g: jnp.ndarray):
    return f[:, jnp.newaxis] + g[jnp.newaxis, :] - self.cost_matrix

  def _softmax(self, f, g, eps, vec, axis):
    """Applies softmax row or column wise, weighted by vec."""
    if vec is not None:
      if axis == 0:
        vec = vec.reshape((vec.size, 1))
      lse_output = jax.scipy.special.logsumexp(
          self._center(f, g) / eps, b=vec, axis=axis, return_sign=True)
      return eps * lse_output[0], lse_output[1]
    else:
      lse_output = jax.scipy.special.logsumexp(
          self._center(f, g) / eps, axis=axis, return_sign=False)
      return eps * lse_output, jnp.array([1.0])

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_potentials(self, f, g, vec, axis):
    """Applies lse_kernel to arbitrary vector while keeping track of signs."""
    lse_res, lse_sgn = self.apply_lse_kernel(
        f, g, self.epsilon, vec=vec, axis=axis)
    lse_res += f if axis == 1 else g
    return lse_sgn * jnp.exp(lse_res / self.epsilon)

  # wrapper to allow default option for axis.
  def apply_transport_from_potentials(self,
                                      f: jnp.ndarray,
                                      g: jnp.ndarray,
                                      vec: jnp.ndarray,
                                      axis: int = 0) -> jnp.ndarray:
    """Applies transport matrix computed from potentials to a (batched) vec.

    This approach does not instantiate the transport matrix itself, but uses
    instead potentials to apply the transport using apply_lse_kernel, therefore
    guaranteeing stability and lower memory footprint.

    Computations are done in log space, and take advantage of the
    (b=..., return_sign=True) optional parameters of logsumexp.

    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      vec: jnp.ndarray [batch, num_a or num_b], vector that will be multiplied
        by transport matrix corresponding to potentials f, g, and geom.
      axis: axis to differentiate left (0) or right (1) multiply.
    Returns:
      ndarray of the size of vec.
    """
    if vec.ndim == 1:
      return self._apply_transport_from_potentials(
          f, g, vec[jnp.newaxis, :], axis)[0, :]
    return self._apply_transport_from_potentials(f, g, vec, axis)

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_scalings(self, u, v, vec, axis):
    u, v = (u, v * vec) if axis == 1 else (v, u * vec)
    return u * self.apply_kernel(v, eps=self.epsilon, axis=axis)

  # wrapper to allow default option for axis
  def apply_transport_from_scalings(self,
                                    u: jnp.ndarray,
                                    v: jnp.ndarray,
                                    vec: jnp.ndarray,
                                    axis: int = 0) -> jnp.ndarray:
    """Applies transport matrix computed from scalings to a (batched) vec.

    This approach does not instantiate the transport matrix itself, but
    relies instead on the apply_kernel function.

    Args:
      u: jnp.ndarray [num_a,] , scaling of size num_rows of cost_matrix
      v: jnp.ndarray [num_b,] , scaling of size num_cols of cost_matrix
      vec: jnp.ndarray [batch, num_a or num_b], vector that will be multiplied
        by transport matrix corresponding to scalings u, v, and geom.
      axis: axis to differentiate left (0) or right (1) multiply.
    Returns:
      ndarray of the size of vec.
    """
    if vec.ndim == 1:
      return self._apply_transport_from_scalings(u, v, vec[jnp.newaxis, :],
                                                 axis)[0, :]
    return self._apply_transport_from_scalings(u, v, vec, axis)

  def potential_from_scaling(self, scaling: jnp.ndarray) -> jnp.ndarray:
    return self.epsilon * jnp.log(scaling)

  def scaling_from_potential(self, potential: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(potential / self.epsilon)

  def apply_cost(self, arr: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Applies cost matrix to array (vector or matrix).

    Args:
      arr: jnp.ndarray [num_b,...] or [num_a,...], depending on axis.
      axis: axis.
    Returns:
      A jnp.ndarray corresponding to cost x matrix
    """
    return jax.vmap(lambda x: self._apply_cost_to_vec(x, axis)
                   )(jnp.atleast_2d(arr))

  def _apply_cost_to_vec(self, vec: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Applies [num_a, num_b] cost matrix to vector.

    Args:
      vec: jnp.ndarray [num_b,] ([num_a,] if axis=1) vector
      axis: axis.
    Returns:
      A jnp.ndarray corresponding to cost x vector
    """
    return jnp.dot(self.cost_matrix if axis == 0 else self.cost_matrix.T, vec)

  @classmethod
  def prepare_divergences(cls, *args, static_b: bool = False, **kwargs):
    """Instantiates 2 (or 3) geometries to compute a Sinkhorn divergence."""
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
