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
"""A class describing operations used to instantiate and use a geometry."""
import functools
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

if TYPE_CHECKING:
  from ott.geometry import low_rank

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing_extensions import Literal

from ott.geometry import epsilon_scheduler, ops


@jax.tree_util.register_pytree_node_class
class Geometry:
  r"""Base class to define ground costs/kernels used in optimal transport.

  Optimal transport problems are intrinsically geometric: they compute an
  optimal way to transport mass from one configuration onto another. To define
  what is meant by optimality of a transport requires defining a cost, of moving
  mass from one among several sources, towards one out of multiple targets.
  These sources and targets can be provided as points in vectors spaces, grids,
  or more generally exclusively described through a (dissimilarity) cost matrix,
  or almost equivalently, a (similarity) kernel matrix.

  Once that cost or kernel matrix is set, the ``Geometry`` class provides a
  basic operations to be run with the Sinkhorn algorithm.

  Args:
    cost_matrix: jnp.ndarray<float>[num_a, num_b]: a cost matrix storing n x m
      costs.
    kernel_matrix: jnp.ndarray<float>[num_a, num_b]: a kernel matrix storing n
      x m kernel values.
    epsilon: a regularization parameter.
      If a :class:`~ott.geometry.epsilon_scheduler.Epsilon` scheduler is passed,
      other parameters below are ignored in practice. If the
      parameter is a float, then this is understood to be the regularization
      that is needed, unless ``relative_epsilon`` below is ``True``, in which
      case ``epsilon`` is understood as a normalized quantity, to be scaled by
      the mean value of the :attr:`cost_matrix`.
    relative_epsilon: whether epsilon is passed relative to scale of problem,
      here understood as mean value of :attr:`cost_matrix`.
    scale_epsilon: the scale multiplier for epsilon.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean' and 'max_cost'. Alternatively, a float factor can be
      given to rescale the cost such that ``cost_matrix /= scale_cost``.
      If `True`, use 'mean'.
    tgt_mask: Mask specifying valid rows when computing some statistics of
      :attr:`cost_matrix`, see :attr:`src_mask`.
    tgt_mask: Mask specifying valid columns when computing some statistics of
      :attr:`cost_matrix`, see :attr:`tgt_mask`.
    kwargs: additional kwargs to epsilon scheduler.

  Note:
    When defining a ``Geometry`` through a ``cost_matrix``, it is important to
    select an ``epsilon`` regularization parameter that is meaningful. That
    parameter can be provided by the user, or assigned a default value through
    a simple rule, using the :attr:`mean_cost_matrix`.
  """

  def __init__(
      self,
      cost_matrix: Optional[jnp.ndarray] = None,
      kernel_matrix: Optional[jnp.ndarray] = None,
      epsilon: Union[epsilon_scheduler.Epsilon, float, None] = None,
      relative_epsilon: Optional[bool] = None,
      scale_epsilon: Optional[float] = None,
      src_mask: Optional[jnp.ndarray] = None,
      tgt_mask: Optional[jnp.ndarray] = None,
      scale_cost: Union[bool, int, float, Literal['mean', 'max_cost',
                                                  'median']] = 1.0,
      **kwargs: Any,
  ):
    self._cost_matrix = cost_matrix
    self._kernel_matrix = kernel_matrix
    self._epsilon_init = epsilon
    self._relative_epsilon = relative_epsilon
    self._scale_epsilon = scale_epsilon
    self._scale_cost = "mean" if scale_cost is True else scale_cost
    self._src_mask = src_mask
    self._tgt_mask = tgt_mask
    # Define default dictionary and update it with user's values.
    self._kwargs = {**{'init': None, 'decay': None}, **kwargs}

  @property
  def cost_rank(self) -> None:
    """Output rank of cost matrix, if any was provided."""
    return None

  @property
  def scale_epsilon(self) -> float:
    """Compute the scale of the epsilon, potentially based on data."""
    if isinstance(self._epsilon_init, epsilon_scheduler.Epsilon):
      return 1.0

    rel = self._relative_epsilon
    trigger = ((self._scale_epsilon is None) and
               ((rel is None and self._epsilon_init is None) or rel))

    if (self._scale_epsilon is None) and (trigger is not None):  # for dry run
      return jnp.where(
          trigger, jax.lax.stop_gradient(self.mean_cost_matrix), 1.0
      )
    else:
      return self._scale_epsilon

  @property
  def _epsilon(self) -> epsilon_scheduler.Epsilon:
    """Return epsilon scheduler, either passed directly or by building it."""
    if isinstance(self._epsilon_init, epsilon_scheduler.Epsilon):
      return self._epsilon_init
    eps = 5e-2 if self._epsilon_init is None else self._epsilon_init
    return epsilon_scheduler.Epsilon.make(
        eps, scale_epsilon=self.scale_epsilon, **self._kwargs
    )

  @property
  def cost_matrix(self) -> jnp.ndarray:
    """Cost matrix, recomputed from kernel if only kernel was specified."""
    if self._cost_matrix is None:
      # If no epsilon was passed on to the geometry, then assume it is one by
      # default.
      cost = -jnp.log(self._kernel_matrix)
      cost *= self.inv_scale_cost
      return cost if self._epsilon_init is None else self.epsilon * cost
    return self._cost_matrix * self.inv_scale_cost

  @property
  def median_cost_matrix(self) -> float:
    """Median of the :attr:`cost_matrix`."""
    geom = self._masked_geom(mask_value=jnp.nan)
    return jnp.nanmedian(geom.cost_matrix)  # will fail for online PC

  @property
  def mean_cost_matrix(self) -> float:
    """Mean of the :attr:`cost_matrix`."""
    tmp = self._masked_geom().apply_cost(self._n_normed_ones).squeeze()
    return jnp.sum(tmp * self._m_normed_ones)

  @property
  def kernel_matrix(self) -> jnp.ndarray:
    """Kernel matrix, either provided by user or recomputed from \
     :attr:`cost_matrix`."""
    if self._kernel_matrix is None:
      return jnp.exp(-(self._cost_matrix * self.inv_scale_cost / self.epsilon))
    return self._kernel_matrix ** self.inv_scale_cost

  @property
  def epsilon(self) -> float:
    """Epsilon regularization value."""
    return self._epsilon.target

  @property
  def shape(self) -> Tuple[int, int]:
    """Shape of the geometry."""
    mat = (
        self._kernel_matrix if self._cost_matrix is None else self._cost_matrix
    )
    if mat is not None:
      return mat.shape
    return 0, 0

  @property
  def is_squared_euclidean(self) -> bool:
    """Whether cost is computed by taking squared-Eucl. distance of points."""
    return False

  @property
  def is_online(self) -> bool:
    """Whether geometry cost/kernel should be recomputed on the fly."""
    return False

  @property
  def is_symmetric(self) -> bool:
    """Whether geometry cost/kernel is a symmetric matrix."""
    mat = self.kernel_matrix if self.cost_matrix is None else self.cost_matrix
    return (
        mat.shape[0] == mat.shape[1] and jnp.all(mat == mat.T)
    ) if mat is not None else False

  @property
  def inv_scale_cost(self) -> float:
    """Compute and return inverse of scaling factor for cost matrix."""
    if isinstance(self._scale_cost, (int, float)):
      return 1.0 / self._scale_cost
    self = self._masked_geom(mask_value=jnp.nan)
    if self._scale_cost == 'max_cost':
      return 1.0 / jnp.nanmax(self._cost_matrix)
    if self._scale_cost == 'mean':
      return 1.0 / jnp.nanmean(self._cost_matrix)
    if self._scale_cost == 'median':
      return 1.0 / jnp.nanmedian(self._cost_matrix)
    raise ValueError(f'Scaling {self._scale_cost} not implemented.')

  def _set_scale_cost(
      self, scale_cost: Optional[Union[bool, float, str]]
  ) -> "Geometry":
    # case when `geom` doesn't have `scale_cost` or doesn't need to be modified
    # `False` retains the original scale
    if scale_cost is False or scale_cost == self._scale_cost:
      return self
    children, aux_data = self.tree_flatten()
    aux_data["scale_cost"] = scale_cost
    return type(self).tree_unflatten(aux_data, children)

  def copy_epsilon(self, other: 'Geometry') -> "Geometry":
    """Copy the epsilon parameters from another geometry."""
    scheduler = other._epsilon
    self._epsilon_init = scheduler._target_init
    self._relative_epsilon = False
    self._scale_epsilon = other.scale_epsilon
    return self

  # The functions below are at the core of Sinkhorn iterations, they
  # are implemented here in their default form, either in lse (using directly
  # cost matrices in stabilized form) or kernel mode (using kernel matrices).

  def apply_lse_kernel(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      eps: float,
      vec: jnp.ndarray = None,
      axis: int = 0
  ) -> jnp.ndarray:
    r"""Apply :attr:`kernel_matrix` in log domain on a pair of dual potential variables.

    This function applies the ground geometry's kernel in log domain, using
    a stabilized formulation. At a high level, this iteration performs either:

    - output = eps * log (K (exp(g / eps) * vec))  (1)
    - output = eps * log (K'(exp(f / eps) * vec))  (2)

    K is implicitly exp(-cost_matrix/eps).

    To carry this out in a stabilized way, we take advantage of the fact that
    the entries of the matrix ``f[:,*] + g[*,:] - C`` are all negative, and
    therefore their exponential never overflows, to add (and subtract after)
    f and g in iterations 1 & 2 respectively.

    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      eps: float, regularization strength
      vec: jnp.ndarray [num_a or num_b,] , when not None, this has the effect of
        doing log-Kernel computations with an addition elementwise
        multiplication of exp(g / eps) by a vector. This is carried out by adding
        weights to the log-sum-exp function, and needs to handle signs
        separately.
      axis: summing over axis 0 when doing (2), or over axis 1 when doing (1)

    Returns:
      A jnp.ndarray corresponding to output above, depending on axis.
    """
    w_res, w_sgn = self._softmax(f, g, eps, vec, axis)
    remove = f if axis == 1 else g
    return w_res - jnp.where(jnp.isfinite(remove), remove, 0), w_sgn

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Apply :attr:`kernel_matrix` on positive scaling vector.

    This function applies the ground geometry's kernel, to perform either
    output = K v    (1)
    output = K'u   (2)
    where K is [num_a, num_b]

    Args:
      scaling: jnp.ndarray [num_a or num_b] , scaling of size num_rows or
        num_cols of kernel_matrix
      eps: passed for consistency, not used yet.
      axis: standard kernel product if axis is 1, transpose if 0.

    Returns:
      a jnp.ndarray corresponding to output above, depending on axis.
    """
    if eps is None:
      kernel = self.kernel_matrix
    else:
      kernel = self.kernel_matrix ** (self.epsilon / eps)
    kernel = kernel if axis == 1 else kernel.T

    return jnp.dot(kernel, scaling)

  def marginal_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Output marginal of transportation matrix from potentials.

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
    return jnp.exp((z + h) / self.epsilon)

  def marginal_from_scalings(
      self,
      u: jnp.ndarray,
      v: jnp.ndarray,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Output marginal of transportation matrix from scalings."""
    u, v = (v, u) if axis == 0 else (u, v)
    return u * self.apply_kernel(v, eps=self.epsilon, axis=axis)

  def transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray
  ) -> jnp.ndarray:
    """Output transport matrix from potentials."""
    return jnp.exp(self._center(f, g) / self.epsilon)

  def transport_from_scalings(
      self, u: jnp.ndarray, v: jnp.ndarray
  ) -> jnp.ndarray:
    """Output transport matrix from pair of scalings."""
    return self.kernel_matrix * u[:, jnp.newaxis] * v[jnp.newaxis, :]

  # Functions that are not supposed to be changed by inherited classes.
  # These are the point of entry for Sinkhorn's algorithm to use a geometry.

  def update_potential(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      log_marginal: jnp.ndarray,
      iteration: Optional[int] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Carry out one Sinkhorn update for potentials, i.e. in log space.

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

  def update_scaling(
      self,
      scaling: jnp.ndarray,
      marginal: jnp.ndarray,
      iteration: Optional[int] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Carry out one Sinkhorn update for scalings, using kernel directly.

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
  def _center(self, f: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    return f[:, jnp.newaxis] + g[jnp.newaxis, :] - self.cost_matrix

  def _softmax(self, f, g, eps, vec, axis):
    """Apply softmax row or column wise, weighted by vec."""
    if vec is not None:
      if axis == 0:
        vec = vec.reshape((vec.size, 1))
      lse_output = ops.logsumexp(
          self._center(f, g) / eps, b=vec, axis=axis, return_sign=True
      )
      return eps * lse_output[0], lse_output[1]
    else:
      lse_output = ops.logsumexp(
          self._center(f, g) / eps, axis=axis, return_sign=False
      )
      return eps * lse_output, jnp.array([1.0])

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_potentials(self, f, g, vec, axis):
    """Apply lse_kernel to arbitrary vector while keeping track of signs."""
    lse_res, lse_sgn = self.apply_lse_kernel(
        f, g, self.epsilon, vec=vec, axis=axis
    )
    lse_res += f if axis == 1 else g
    return lse_sgn * jnp.exp(lse_res / self.epsilon)

  # wrapper to allow default option for axis.
  def apply_transport_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    """Apply transport matrix computed from potentials to a (batched) vec.

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
          f, g, vec[jnp.newaxis, :], axis
      )[0, :]
    return self._apply_transport_from_potentials(f, g, vec, axis)

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_scalings(self, u, v, vec, axis):
    u, v = (u, v * vec) if axis == 1 else (v, u * vec)
    return u * self.apply_kernel(v, eps=self.epsilon, axis=axis)

  # wrapper to allow default option for axis
  def apply_transport_from_scalings(
      self,
      u: jnp.ndarray,
      v: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    """Apply transport matrix computed from scalings to a (batched) vec.

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
      return self._apply_transport_from_scalings(
          u, v, vec[jnp.newaxis, :], axis
      )[0, :]
    return self._apply_transport_from_scalings(u, v, vec, axis)

  def potential_from_scaling(self, scaling: jnp.ndarray) -> jnp.ndarray:
    """Compute dual potential vector from scaling vector.

    Args:
      scaling: vector.

    Returns:
      a vector of the same size.
    """
    return self.epsilon * jnp.log(scaling)

  def scaling_from_potential(self, potential: jnp.ndarray) -> jnp.ndarray:
    """Compute scaling vector from dual potential.

    Args:
      potential: vector.

    Returns:
      a vector of the same size.
    """
    finite = jnp.isfinite(potential)
    return jnp.where(
        finite, jnp.exp(jnp.where(finite, potential / self.epsilon, 0.0)), 0.0
    )

  def apply_square_cost(self, arr: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply elementwise-square of cost matrix to array (vector or matrix).

    This function applies the ground geometry's cost matrix, to perform either
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b], when the cost matrix itself is computed as a
    squared-Euclidean distance between vectors, and therefore admits an
    explicit low-rank factorization.

    Args:
      arr: array.
      axis: axis of the array on which the cost matrix should be applied.

    Returns:
      An array, [num_b, p] if axis=0 or [num_a, p] if axis=1.
    """
    return self.apply_cost(arr, axis=axis, fn=lambda x: x ** 2)

  def apply_cost(
      self,
      arr: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      **kwargs: Any
  ) -> jnp.ndarray:
    """Apply :attr:`cost_matrix` to array (vector or matrix).

    This function applies the ground geometry's cost matrix, to perform either
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b]

    Args:
      arr: jnp.ndarray [num_a or num_b, p], vector that will be multiplied by
        the cost matrix.
      axis: standard cost matrix if axis=1, transpose if 0
      fn: function to apply to cost matrix element-wise before the dot product
      kwargs: Keyword arguments for :meth:`_apply_cost_to_vec`.

    Returns:
      An array, [num_b, p] if axis=0 or [num_a, p] if axis=1
    """
    if arr.ndim == 1:
      arr = arr.reshape(-1, 1)

    app = functools.partial(self._apply_cost_to_vec, axis=axis, fn=fn, **kwargs)
    return jax.vmap(app, in_axes=1, out_axes=1)(arr)

  def _apply_cost_to_vec(
      self,
      vec: jnp.ndarray,
      axis: int = 0,
      fn=None,
      **_: Any,
  ) -> jnp.ndarray:
    """Apply ``[num_a, num_b]`` fn(cost) (or transpose) to vector.

    Args:
      vec: jnp.ndarray [num_a,] ([num_b,] if axis=1) vector
      axis: axis on which the reduction is done.
      fn: function optionally applied to cost matrix element-wise, before the
        doc product

    Returns:
      A jnp.ndarray corresponding to cost x vector
    """
    matrix = self.cost_matrix.T if axis == 0 else self.cost_matrix
    matrix = fn(matrix) if fn is not None else matrix
    return jnp.dot(matrix, vec)

  @classmethod
  def prepare_divergences(
      cls,
      *args: Any,
      static_b: bool = False,
      **kwargs: Any
  ) -> Tuple["Geometry", ...]:
    """Instantiate 2 (or 3) geometries to compute a Sinkhorn divergence."""
    size = 2 if static_b else 3
    nones = [None, None, None]
    cost_matrices = kwargs.pop('cost_matrix', args)
    kernel_matrices = kwargs.pop('kernel_matrix', nones)
    cost_matrices = cost_matrices if cost_matrices is not None else nones
    return tuple(
        cls(cost_matrix=arg1, kernel_matrix=arg2, **kwargs)
        for arg1, arg2, _ in zip(cost_matrices, kernel_matrices, range(size))
    )

  def to_LRCGeometry(
      self,
      rank: int,
      tol: float = 1e-2,
      seed: int = 0
  ) -> 'low_rank.LRCGeometry':
    r"""Factorize the cost matrix in sublinear time :cite:`indyk:19`.

    Uses the implementation of :cite:`scetbon:21`, algorithm 4.

    It holds that with probability *0.99*,
    :math:`||A - UV||_F^2 \leq || A - A_k ||_F^2 + tol \cdot ||A||_F^2`,
    where :math:`A` is ``n x m`` cost matrix, :math:`UV` the factorization
    computed in sublinear time and :math:`A_k` the best rank-k approximation.

    Args:
      rank: Target rank of the :attr:`cost_matrix`.
      tol: Tolerance of the error. The total number of sampled points is
        :math:`min(n, m,\frac{rank}{tol})`.
      seed: Random seed.

    Returns:
      Low-rank geometry.
    """
    from ott.geometry import low_rank

    assert rank > 0, f"Rank must be positive, got {rank}."
    rng = jax.random.PRNGKey(seed)
    key1, key2, key3, key4, key5 = jax.random.split(rng, 5)
    n, m = self.shape
    n_subset = min(int(rank / tol), n, m)

    i_star = jax.random.randint(key1, shape=(), minval=0, maxval=n)
    j_star = jax.random.randint(key2, shape=(), minval=0, maxval=m)

    # force `batch_size=None` since `cost_matrix` would be `None`
    ci_star = self.subset(
        i_star, None, batch_size=None
    ).cost_matrix.ravel() ** 2  # (m,)
    cj_star = self.subset(
        None, j_star, batch_size=None
    ).cost_matrix.ravel() ** 2  # (n,)

    p_row = cj_star + ci_star[j_star] + jnp.mean(ci_star)  # (n,)
    p_row /= jnp.sum(p_row)
    row_ixs = jax.random.choice(key3, n, shape=(n_subset,), p=p_row)
    # (n_subset, m)
    S = self.subset(row_ixs, None, batch_size=None).cost_matrix
    S /= jnp.sqrt(n_subset * p_row[row_ixs][:, None])

    p_col = jnp.sum(S ** 2, axis=0)  # (m,)
    p_col /= jnp.sum(p_col)
    # (n_subset,)
    col_ixs = jax.random.choice(key4, m, shape=(n_subset,), p=p_col)
    # (n_subset, n_subset)
    W = S[:, col_ixs] / jnp.sqrt(n_subset * p_col[col_ixs][None, :])

    U, _, V = jsp.linalg.svd(W)
    U = U[:, :rank]  # (n_subset, rank)
    U = (S.T @ U) / jnp.linalg.norm(W.T @ U, axis=0)  # (m, rank)

    _, d, v = jnp.linalg.svd(U.T @ U)  # (k,), (k, k)
    v = v.T / jnp.sqrt(d)[None, :]

    inv_scale = (1. / jnp.sqrt(n_subset))
    col_ixs = jax.random.choice(key5, m, shape=(n_subset,))  # (n_subset,)

    # (n, n_subset)
    A_trans = self.subset(
        None, col_ixs, batch_size=None
    ).cost_matrix * inv_scale
    B = (U[col_ixs, :] @ v * inv_scale)  # (n_subset, k)
    M = jnp.linalg.inv(B.T @ B)  # (k, k)
    V = jnp.linalg.multi_dot([A_trans, B, M.T, v.T])  # (n, k)

    return low_rank.LRCGeometry(
        cost_1=V,
        cost_2=U,
        epsilon=self._epsilon_init,
        relative_epsilon=self._relative_epsilon,
        scale=self._scale_epsilon,
        scale_cost=self._scale_cost,
        **self._kwargs
    )

  def subset(
      self, src_ixs: Optional[jnp.ndarray], tgt_ixs: Optional[jnp.ndarray],
      **kwargs: Any
  ) -> "Geometry":
    """Subset rows or columns of a geometry.

    Args:
      src_ixs: Row indices. If ``None``, use all rows.
      tgt_ixs: Column indices. If ``None``, use all columns.
      kwargs: Keyword arguments to override the initialization.

    Returns:
      The subsetted geometry.
    """

    def subset_fn(
        arr: Optional[jnp.ndarray],
        src_ixs: Optional[jnp.ndarray],
        tgt_ixs: Optional[jnp.ndarray],
    ) -> Optional[jnp.ndarray]:
      if arr is None:
        return None
      if src_ixs is not None:
        arr = arr[jnp.atleast_1d(src_ixs)]
      if tgt_ixs is not None:
        arr = arr[:, jnp.atleast_1d(tgt_ixs)]
      return arr

    return self._mask_subset_helper(
        src_ixs, tgt_ixs, fn=subset_fn, propagate_mask=True, **kwargs
    )

  def mask(
      self,
      src_mask: Optional[jnp.ndarray],
      tgt_mask: Optional[jnp.ndarray],
      mask_value: float = 0.,
  ) -> "Geometry":
    """Mask rows or columns of a geometry.

    The mask is used only when computing some statistics of the
    :attr:`cost_matrix`.

        - :attr:`mean_cost_matrix`
        - :attr:`median_cost_matrix`
        - :attr:`inv_scale_cost`

    Args:
      src_mask: Row mask. Can be specified either as a boolean array of shape
        ``[num_a,]`` or as an array of indices. If ``None``, no mask is applied.
      tgt_mask: Column mask. Can be specified either as a boolean array of shape
        ``[num_b,]`` or as an array of indices. If ``None``, no mask is applied.
      mask_value: Value to use for masking.

    Returns:
      The masked geometry.
    """

    def mask_fn(
        arr: Optional[jnp.ndarray],
        src_mask: Optional[jnp.ndarray],
        tgt_mask: Optional[jnp.ndarray],
    ) -> Optional[jnp.ndarray]:
      if arr is None:
        return arr
      assert arr.ndim == 2, arr.ndim
      if src_mask is not None:
        arr = jnp.where(src_mask[:, None], arr, mask_value)
      if tgt_mask is not None:
        arr = jnp.where(tgt_mask[None, :], arr, mask_value)
      return arr

    src_mask = self._normalize_mask(src_mask, self.shape[0])
    tgt_mask = self._normalize_mask(tgt_mask, self.shape[1])
    return self._mask_subset_helper(
        src_mask, tgt_mask, fn=mask_fn, propagate_mask=False
    )

  def _mask_subset_helper(
      self,
      src_ixs: Optional[jnp.ndarray],
      tgt_ixs: Optional[jnp.ndarray],
      *,
      fn: Callable[
          [Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]],
          Optional[jnp.ndarray]],
      propagate_mask: bool,
      **kwargs: Any,
  ) -> "Geometry":
    (cost, kernel, *children, src_mask, tgt_mask,
     kws), aux_data = self.tree_flatten()
    cost = fn(cost, src_ixs, tgt_ixs)
    kernel = fn(kernel, src_ixs, tgt_ixs)
    if propagate_mask:
      src_mask = self._normalize_mask(src_mask, self.shape[0])
      tgt_mask = self._normalize_mask(tgt_mask, self.shape[1])
      src_mask = fn(src_mask, src_ixs, None)
      tgt_mask = fn(tgt_mask, tgt_ixs, None)

    aux_data = {**aux_data, **kwargs}
    return type(self).tree_unflatten(
        aux_data, [cost, kernel] + children + [src_mask, tgt_mask, kws]
    )

  @property
  def src_mask(self) -> Optional[jnp.ndarray]:
    """Mask of shape ``[num_a,]`` to compute :attr:`cost_matrix` statistics.

    Specifically, it is used when computing:

      - :attr:`mean_cost_matrix`
      - :attr:`median_cost_matrix`
      - :attr:`inv_scale_cost`
    """
    return self._normalize_mask(self._src_mask, self.shape[0])

  @property
  def tgt_mask(self) -> Optional[jnp.ndarray]:
    """Mask of shape ``[num_b,]`` to compute :attr:`cost_matrix` statistics.

    Specifically, it is used when computing:

      - :attr:`mean_cost_matrix`
      - :attr:`median_cost_matrix`
      - :attr:`inv_scale_cost`
    """
    return self._normalize_mask(self._tgt_mask, self.shape[1])

  def _masked_geom(self, mask_value: float = 0.) -> "Geometry":
    """Mask geometry based on :attr:`src_mask` and :attr:`tgt_mask`."""
    src_mask, tgt_mask = self.src_mask, self.tgt_mask
    if src_mask is None and tgt_mask is None:
      return self
    return self.mask(src_mask, tgt_mask, mask_value=mask_value)

  @property
  def _n_normed_ones(self) -> jnp.ndarray:
    """Normalized array of shape ``[num_a,]`` \
    taking into account :attr:`src_mask`."""
    mask = self.src_mask
    arr = jnp.ones(self.shape[0]) if mask is None else mask
    return arr / jnp.sum(arr)

  @property
  def _m_normed_ones(self) -> jnp.ndarray:
    """Normalized array of shape ``[num_b,]`` \
    taking into account :attr:`tgt_mask`."""
    mask = self.tgt_mask
    arr = jnp.ones(self.shape[1]) if mask is None else mask
    return arr / jnp.sum(arr)

  @staticmethod
  def _normalize_mask(mask: Optional[Union[int, jnp.ndarray]],
                      size: int) -> Optional[jnp.ndarray]:
    """Convert array of indices to a boolean mask."""
    if mask is None:
      return None
    mask = jnp.atleast_1d(mask)
    if not jnp.issubdtype(mask, (bool, jnp.bool_)):
      mask = jnp.isin(jnp.arange(size), mask)
    assert mask.shape == (size,)
    return mask

  def tree_flatten(self):
    return (
        self._cost_matrix, self._kernel_matrix, self._epsilon_init,
        self._relative_epsilon, self._scale_epsilon, self._src_mask,
        self._tgt_mask, self._kwargs
    ), {
        'scale_cost': self._scale_cost
    }

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    *args, kwargs = children
    return cls(*args, **kwargs, **aux_data)


def is_affine(fn) -> bool:
  """Test heuristically if a function is affine."""
  x = jnp.arange(10.0)
  out = jax.vmap(jax.grad(fn))(x)
  return jnp.sum(jnp.diff(jnp.abs(out))) == 0.0


def is_linear(fn) -> bool:
  """Test heuristically if a function is linear."""
  return fn(0.0) == 0.0 and is_affine(fn)
