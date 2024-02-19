# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
  from ott.geometry import low_rank

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ott import utils
from ott.geometry import epsilon_scheduler
from ott.math import utils as mu

__all__ = ["Geometry", "is_linear", "is_affine"]


@jax.tree_util.register_pytree_node_class
class Geometry:
  r"""Base class to define ground costs/kernels used in optimal transport.

  Optimal transport problems are intrinsically geometric: they compute an
  optimal way to transport mass from one configuration onto another. To define
  what is meant by optimality of transport requires defining a cost, of moving
  mass from one among several sources, towards one out of multiple targets.
  These sources and targets can be provided as points in vectors spaces, grids,
  or more generally exclusively described through a (dissimilarity) cost matrix,
  or almost equivalently, a (similarity) kernel matrix.

  Once that cost or kernel matrix is set, the ``Geometry`` class provides a
  basic operations to be run with the Sinkhorn algorithm.

  Args:
    cost_matrix: Cost matrix of shape ``[n, m]``.
    kernel_matrix: Kernel matrix of shape ``[n, m]``.
    epsilon: Regularization parameter. If ``None`` and either
      ``relative_epsilon = True`` or ``relative_epsilon = None``, this defaults
      to the value computed in :attr:`mean_cost_matrix` / 20. If passed as a
      ``float``, then the regularizer that is ultimately used is either that
      ``float`` value (if ``relative_epsilon = False`` or ``None``) or that
      ``float`` times the :attr:`mean_cost_matrix`
      (if ``relative_epsilon = True``). Look for
      :class:`~ott.geometry.epsilon_scheduler.Epsilon` when passed as a
      scheduler.
    relative_epsilon: when `False`, the parameter ``epsilon`` specifies the
      value of the entropic regularization parameter. When `True`, ``epsilon``
      refers to a fraction of the :attr:`mean_cost_matrix`, which is computed
      adaptively from data.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean' and 'max_cost'. Alternatively, a float factor can be
      given to rescale the cost such that ``cost_matrix /= scale_cost``.
    src_mask: Mask specifying valid rows when computing some statistics of
      :attr:`cost_matrix`, see :attr:`src_mask`.
    tgt_mask: Mask specifying valid columns when computing some statistics of
      :attr:`cost_matrix`, see :attr:`tgt_mask`.

  Note:
    When defining a :class:`~ott.geometry.geometry.Geometry` through a
    ``cost_matrix``, it is important to select an ``epsilon`` regularization
    parameter that is meaningful. That parameter can be provided by the user,
    or assigned a default value through a simple rule,
    using the :attr:`mean_cost_matrix`.
  """

  def __init__(
      self,
      cost_matrix: Optional[jnp.ndarray] = None,
      kernel_matrix: Optional[jnp.ndarray] = None,
      epsilon: Optional[Union[float, epsilon_scheduler.Epsilon]] = None,
      relative_epsilon: Optional[bool] = None,
      scale_cost: Union[int, float, Literal["mean", "max_cost",
                                            "median"]] = 1.0,
      src_mask: Optional[jnp.ndarray] = None,
      tgt_mask: Optional[jnp.ndarray] = None,
  ):
    self._cost_matrix = cost_matrix
    self._kernel_matrix = kernel_matrix

    # needed for `copy_epsilon`, because of the `isinstance` check
    self._epsilon_init = epsilon if isinstance(
        epsilon, epsilon_scheduler.Epsilon
    ) else epsilon_scheduler.Epsilon(epsilon)
    self._relative_epsilon = relative_epsilon

    self._scale_cost = scale_cost

    self._src_mask = src_mask
    self._tgt_mask = tgt_mask

  @property
  def cost_rank(self) -> Optional[int]:
    """Output rank of cost matrix, if any was provided."""

  @property
  def cost_matrix(self) -> jnp.ndarray:
    """Cost matrix, recomputed from kernel if only kernel was specified."""
    if self._cost_matrix is None:
      # If no epsilon was passed on to the geometry, then assume it is one by
      # default.
      eps = jnp.finfo(self._kernel_matrix.dtype).tiny
      cost = -jnp.log(self._kernel_matrix + eps)
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
    """Kernel matrix.

    Either provided by user or recomputed from :attr:`cost_matrix`.
    """
    if self._kernel_matrix is None:
      return jnp.exp(-(self._cost_matrix * self.inv_scale_cost / self.epsilon))
    return self._kernel_matrix ** self.inv_scale_cost

  @property
  def _epsilon(self) -> epsilon_scheduler.Epsilon:
    (target, scale_eps, _, _), _ = self._epsilon_init.tree_flatten()
    rel = self._relative_epsilon

    use_mean_scale = rel is True or (rel is None and target is None)
    if scale_eps is None and use_mean_scale:
      scale_eps = jax.lax.stop_gradient(self.mean_cost_matrix)

    if isinstance(self._epsilon_init, epsilon_scheduler.Epsilon):
      return self._epsilon_init.set(scale_epsilon=scale_eps)

    return epsilon_scheduler.Epsilon(
        target=5e-2 if target is None else target, scale_epsilon=scale_eps
    )

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
  def can_LRC(self) -> bool:
    """Check quickly if casting geometry as LRC makes sense.

    This check is only carried out using basic considerations from the geometry,
    not using a rigorous check involving, e.g., SVD.
    """
    return False

  @property
  def is_squared_euclidean(self) -> bool:
    """Whether cost is computed by taking squared Euclidean distance."""
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
    if isinstance(self._scale_cost, (int, float, np.number, jax.Array)):
      return 1.0 / self._scale_cost
    self = self._masked_geom(mask_value=jnp.nan)
    if self._scale_cost == "max_cost":
      return 1.0 / jnp.nanmax(self._cost_matrix)
    if self._scale_cost == "mean":
      return 1.0 / jnp.nanmean(self._cost_matrix)
    if self._scale_cost == "median":
      return 1.0 / jnp.nanmedian(self._cost_matrix)
    raise ValueError(f"Scaling {self._scale_cost} not implemented.")

  def set_scale_cost(self, scale_cost: Union[float, str]) -> "Geometry":
    """Modify how to rescale of the :attr:`cost_matrix`."""
    # case when `geom` doesn't have `scale_cost` or doesn't need to be modified
    # `False` retains the original scale
    if scale_cost == self._scale_cost:
      return self
    children, aux_data = self.tree_flatten()
    aux_data["scale_cost"] = scale_cost
    return type(self).tree_unflatten(aux_data, children)

  def copy_epsilon(self, other: "Geometry") -> "Geometry":
    """Copy the epsilon parameters from another geometry."""
    other_epsilon = other._epsilon
    children, aux_data = self.tree_flatten()

    new_children = []
    for child in children:
      if isinstance(child, epsilon_scheduler.Epsilon):
        child = child.set(
            target=other_epsilon._target_init,
            scale_epsilon=other_epsilon._scale_epsilon
        )
      new_children.append(child)

    aux_data["relative_epsilon"] = False
    return type(self).tree_unflatten(aux_data, new_children)

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
    r"""Apply :attr:`kernel_matrix` in log domain.

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
        multiplication of exp(g / eps) by a vector. This is carried out by
        adding weights to the log-sum-exp function, and needs to handle signs
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
    correction used to stabilize computations, and lifts this with an exp to
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

  def _softmax(
      self, f: jnp.ndarray, g: jnp.ndarray, eps: float,
      vec: Optional[jnp.ndarray], axis: int
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply softmax row or column wise, weighted by vec."""
    if vec is not None:
      if axis == 0:
        vec = vec.reshape((-1, 1))
      lse_output = mu.logsumexp(
          self._center(f, g) / eps, b=vec, axis=axis, return_sign=True
      )
      return eps * lse_output[0], lse_output[1]

    lse_output = mu.logsumexp(
        self._center(f, g) / eps, axis=axis, return_sign=False
    )
    return eps * lse_output, jnp.array([1.0])

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray, vec: jnp.ndarray, axis: int
  ) -> jnp.ndarray:
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
  def _apply_transport_from_scalings(
      self, u: jnp.ndarray, v: jnp.ndarray, vec: jnp.ndarray, axis: int
  ):
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
    cost_matrices = kwargs.pop("cost_matrix", args)
    kernel_matrices = kwargs.pop("kernel_matrix", nones)
    cost_matrices = cost_matrices if cost_matrices is not None else nones
    return tuple(
        cls(cost_matrix=arg1, kernel_matrix=arg2, **kwargs)
        for arg1, arg2, _ in zip(cost_matrices, kernel_matrices, range(size))
    )

  def to_LRCGeometry(
      self,
      rank: int = 0,
      tol: float = 1e-2,
      rng: Optional[jax.Array] = None,
      scale: float = 1.0
  ) -> "low_rank.LRCGeometry":
    r"""Factorize the cost matrix using either SVD (full) or :cite:`indyk:19`.

    When `rank=min(n,m)` or `0` (by default), use :func:`jax.numpy.linalg.svd`.

    For other values, use the routine in sublinear time :cite:`indyk:19`.
    Uses the implementation of :cite:`scetbon:21`, algorithm 4.

    It holds that with probability *0.99*,
    :math:`||A - UV||_F^2 \leq || A - A_k ||_F^2 + tol \cdot ||A||_F^2`,
    where :math:`A` is ``n x m`` cost matrix, :math:`UV` the factorization
    computed in sublinear time and :math:`A_k` the best rank-k approximation.

    Args:
      rank: Target rank of the :attr:`cost_matrix`.
      tol: Tolerance of the error. The total number of sampled points is
        :math:`min(n, m,\frac{rank}{tol})`.
      rng: The PRNG key to use for initializing the model.
      scale: Value used to rescale the factors of the low-rank geometry.
        Useful when this geometry is used in the linear term of fused GW.

    Returns:
      Low-rank geometry.
    """
    from ott.geometry import low_rank
    assert rank >= 0, f"Rank must be non-negative, got {rank}."
    n, m = self.shape

    if rank == 0 or rank >= min(n, m):
      # TODO(marcocuturi): add hermitian=self.is_symmetric, currently bugging.
      u, s, vh = jnp.linalg.svd(
          self.cost_matrix,
          full_matrices=False,
          compute_uv=True,
      )

      cost_1 = u
      cost_2 = (s[:, None] * vh).T
    else:
      rng = utils.default_prng_key(rng)
      rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
      n_subset = min(int(rank / tol), n, m)

      i_star = jax.random.randint(rng1, shape=(), minval=0, maxval=n)
      j_star = jax.random.randint(rng2, shape=(), minval=0, maxval=m)

      ci_star = self.subset([i_star], None).cost_matrix.ravel() ** 2  # (m,)
      cj_star = self.subset(None, [j_star]).cost_matrix.ravel() ** 2  # (n,)

      p_row = cj_star + ci_star[j_star] + jnp.mean(ci_star)  # (n,)
      p_row /= jnp.sum(p_row)
      row_ixs = jax.random.choice(rng3, n, shape=(n_subset,), p=p_row)
      # (n_subset, m)
      s = self.subset(row_ixs, None).cost_matrix
      s /= jnp.sqrt(n_subset * p_row[row_ixs][:, None])

      p_col = jnp.sum(s ** 2, axis=0)  # (m,)
      p_col /= jnp.sum(p_col)
      # (n_subset,)
      col_ixs = jax.random.choice(rng4, m, shape=(n_subset,), p=p_col)
      # (n_subset, n_subset)
      w = s[:, col_ixs] / jnp.sqrt(n_subset * p_col[col_ixs][None, :])

      U, _, V = jsp.linalg.svd(w)
      U = U[:, :rank]  # (n_subset, rank)
      U = (s.T @ U) / jnp.linalg.norm(w.T @ U, axis=0)  # (m, rank)

      _, d, v = jnp.linalg.svd(U.T @ U)  # (k,), (k, k)
      v = v.T / jnp.sqrt(d)[None, :]

      inv_scale = (1.0 / jnp.sqrt(n_subset))
      col_ixs = jax.random.choice(rng5, m, shape=(n_subset,))  # (n_subset,)

      # (n, n_subset)
      A_trans = self.subset(None, col_ixs).cost_matrix * inv_scale
      B = (U[col_ixs, :] @ v * inv_scale)  # (n_subset, k)
      M = jnp.linalg.inv(B.T @ B)  # (k, k)
      V = jnp.linalg.multi_dot([A_trans, B, M.T, v.T])  # (n, k)
      cost_1 = V
      cost_2 = U

    return low_rank.LRCGeometry(
        cost_1=cost_1,
        cost_2=cost_2,
        epsilon=self._epsilon_init,
        relative_epsilon=self._relative_epsilon,
        scale_cost=self._scale_cost,
        scale_factor=scale,
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
      The modified geometry.
    """

    def subset_fn(
        arr: Optional[jnp.ndarray],
        src_ixs: Optional[jnp.ndarray],
        tgt_ixs: Optional[jnp.ndarray],
    ) -> Optional[jnp.ndarray]:
      if arr is None:
        return None
      if src_ixs is not None:
        arr = arr[src_ixs, ...]
      if tgt_ixs is not None:
        arr = arr[:, tgt_ixs]
      return arr  # noqa: RET504

    return self._mask_subset_helper(
        src_ixs,
        tgt_ixs,
        fn=subset_fn,
        propagate_mask=True,
        **kwargs,
    )

  def mask(
      self,
      src_mask: Optional[jnp.ndarray],
      tgt_mask: Optional[jnp.ndarray],
      mask_value: float = 0.0,
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
      return arr  # noqa: RET504

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
    (cost, kernel, eps, src_mask, tgt_mask), aux_data = self.tree_flatten()
    cost = fn(cost, src_ixs, tgt_ixs)
    kernel = fn(kernel, src_ixs, tgt_ixs)
    if propagate_mask:
      src_mask = self._normalize_mask(src_mask, self.shape[0])
      tgt_mask = self._normalize_mask(tgt_mask, self.shape[1])
      src_mask = fn(src_mask, src_ixs, None)
      tgt_mask = fn(tgt_mask, tgt_ixs, None)

    aux_data = {**aux_data, **kwargs}
    return type(self).tree_unflatten(
        aux_data, [cost, kernel, eps, src_mask, tgt_mask]
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

  @property
  def dtype(self) -> jnp.dtype:
    """The data type."""
    return (
        self._kernel_matrix if self._cost_matrix is None else self._cost_matrix
    ).dtype

  def _masked_geom(self, mask_value: float = 0.0) -> "Geometry":
    """Mask geometry based on :attr:`src_mask` and :attr:`tgt_mask`."""
    src_mask, tgt_mask = self.src_mask, self.tgt_mask
    if src_mask is None and tgt_mask is None:
      return self
    return self.mask(src_mask, tgt_mask, mask_value=mask_value)

  @property
  def _n_normed_ones(self) -> jnp.ndarray:
    """Normalized array of shape ``[num_a,]``."""
    mask = self.src_mask
    arr = jnp.ones(self.shape[0]) if mask is None else mask
    return arr / jnp.sum(arr)

  @property
  def _m_normed_ones(self) -> jnp.ndarray:
    """Normalized array of shape ``[num_b,]``."""
    mask = self.tgt_mask
    arr = jnp.ones(self.shape[1]) if mask is None else mask
    return arr / jnp.sum(arr)

  @staticmethod
  def _normalize_mask(mask: Optional[jnp.ndarray],
                      size: int) -> Optional[jnp.ndarray]:
    """Convert array of indices to a boolean mask."""
    if mask is None:
      return None
    if not jnp.issubdtype(mask, (bool, jnp.bool_)):
      mask = jnp.isin(jnp.arange(size), mask)
    assert mask.shape == (size,)
    return mask

  def tree_flatten(self):  # noqa: D102
    return (
        self._cost_matrix, self._kernel_matrix, self._epsilon_init,
        self._src_mask, self._tgt_mask
    ), {
        "scale_cost": self._scale_cost,
        "relative_epsilon": self._relative_epsilon
    }

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    cost, kernel, eps, src_mask, tgt_mask = children
    return cls(
        cost, kernel, eps, src_mask=src_mask, tgt_mask=tgt_mask, **aux_data
    )


def is_affine(fn) -> bool:
  """Test heuristically if a function is affine."""
  x = jnp.arange(10.0)
  out = jax.vmap(jax.grad(fn))(x)
  return jnp.sum(jnp.diff(jnp.abs(out))) == 0.0


def is_linear(fn) -> bool:
  """Test heuristically if a function is linear."""
  return jnp.logical_and(fn(0.0) == 0.0, is_affine(fn))
