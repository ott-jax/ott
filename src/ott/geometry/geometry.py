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
import jax.tree_util as jtu

from ott import utils
from ott.geometry import epsilon_scheduler as eps_scheduler
from ott.math import utils as mu

__all__ = ["Geometry"]


@jtu.register_pytree_node_class
class Geometry:
  r"""Base class to define ground costs/kernels used in optimal transport.

  Optimal transport problems are intrinsically geometric: they compute an
  optimal way to transport mass from one configuration onto another. To define
  what is meant by optimality of transport requires defining a
  :term:`ground cost`, which quantifies how costly it is to move mass from
  one among several source locations, towards one out of multiple
  target locations. These source and target locations can be described as
  points in vectors spaces, grids, or more generally described
  through a (dissimilarity) cost matrix, or almost equivalently, a
  (similarity) kernel matrix. This class describes such a
  geometry and several useful methods to exploit it.

  Args:
    cost_matrix: Cost matrix of shape ``[n, m]``.
    kernel_matrix: Kernel matrix of shape ``[n, m]``.
    epsilon: Regularization parameter or a scheduler:

      - ``epsilon = None`` and ``relative_epsilon = None``, use
        :math:`0.05 * \text{stddev(cost_matrix)}`.
      - if ``epsilon`` is a :class:`float` and ``relative_epsilon = None``,
        it directly corresponds to the regularization strength.
      - otherwise, ``epsilon`` multiplies the :attr:`mean_cost_matrix` or
        :attr:`std_cost_matrix`, depending on the value of ``relative_epsilon``.

      If ``epsilon = None``, the value of
      :obj:`DEFAULT_EPSILON_SCALE = 0.05 <ott.geometry.epsilon_scheduler.DEFAULT_EPSILON_SCALE>`.
      will be used.
    relative_epsilon: Whether ``epsilon`` refers to a fraction of the
      :attr:`mean_cost_matrix` or :attr:`std_cost_matrix`.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean', 'std' and 'max_cost'. Alternatively, a float factor can
      be given to rescale the cost such that ``cost_matrix /= scale_cost``.

  Note:
    When defining a :class:`~ott.geometry.geometry.Geometry` through a
    ``cost_matrix``, it is important to select an ``epsilon`` regularization
    parameter that is meaningful. That parameter can be provided by the user,
    or assigned a default value through a simple rule, using for instance the
    :attr:`mean_cost_matrix` or the :attr:`std_cost_matrix`.
  """  # noqa: E501

  def __init__(
      self,
      cost_matrix: Optional[jnp.ndarray] = None,
      kernel_matrix: Optional[jnp.ndarray] = None,
      epsilon: Optional[Union[float, eps_scheduler.Epsilon]] = None,
      relative_epsilon: Optional[Literal["mean", "std"]] = None,
      scale_cost: Union[float, Literal["mean", "max_cost", "median",
                                       "std"]] = 1.0,
  ):
    self._cost_matrix = cost_matrix
    self._kernel_matrix = kernel_matrix
    self._epsilon_init = epsilon
    self._relative_epsilon = relative_epsilon
    self._scale_cost = scale_cost

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
    return jnp.median(self.cost_matrix)

  @property
  def mean_cost_matrix(self) -> float:
    """Mean of the :attr:`cost_matrix`."""
    n, m = self.shape
    tmp = self.apply_cost(jnp.full((n,), fill_value=1.0 / n))
    return jnp.sum((1.0 / m) * tmp)

  @property
  def std_cost_matrix(self) -> float:
    r"""Standard deviation of all values stored in :attr:`cost_matrix`.

    Uses the :meth:`apply_square_cost` to remain
    applicable to low-rank matrices, through the formula:

    .. math::
        \sigma^2=\frac{1}{nm}\left(\sum_{ij} C_{ij}^2 -
        (\sum_{ij}C_ij)^2\right).

    to output :math:`\sigma`.
    """
    n, m = self.shape
    tmp = self.apply_square_cost(jnp.full((n,), fill_value=1.0 / n))
    tmp = jnp.sum((1.0 / m) * tmp) - (self.mean_cost_matrix ** 2)
    return jnp.sqrt(jax.nn.relu(tmp))

  @property
  def kernel_matrix(self) -> jnp.ndarray:
    """Kernel matrix.

    Either provided by user or recomputed from :attr:`cost_matrix`.
    """
    if self._kernel_matrix is None:
      return jnp.exp(-self._cost_matrix * self.inv_scale_cost / self.epsilon)
    return self._kernel_matrix ** self.inv_scale_cost

  @property
  def epsilon_scheduler(self) -> eps_scheduler.Epsilon:
    """Epsilon scheduler."""
    if isinstance(self._epsilon_init, eps_scheduler.Epsilon):
      return self._epsilon_init
    # no relative epsilon
    if self._relative_epsilon is None:
      if self._epsilon_init is not None:
        return eps_scheduler.Epsilon(self._epsilon_init)
      multiplier = eps_scheduler.DEFAULT_EPSILON_SCALE
      scale = jax.lax.stop_gradient(self.std_cost_matrix)
      return eps_scheduler.Epsilon(target=multiplier * scale)

    if self._relative_epsilon == "std":
      scale = jax.lax.stop_gradient(self.std_cost_matrix)
    elif self._relative_epsilon == "mean":
      scale = jax.lax.stop_gradient(self.mean_cost_matrix)
    else:
      raise ValueError(f"Invalid relative epsilon: {self._relative_epsilon}.")

    multiplier = (
        eps_scheduler.DEFAULT_EPSILON_SCALE
        if self._epsilon_init is None else self._epsilon_init
    )
    return eps_scheduler.Epsilon(target=multiplier * scale)

  @property
  def epsilon(self) -> float:
    """Epsilon regularization value."""
    return self.epsilon_scheduler.target

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
    return self.is_square and jnp.all(mat == mat.T)

  @property
  def is_square(self) -> bool:
    """Whether geometry cost/kernel is a square matrix."""
    n, m = self.shape
    return (n == m)

  @property
  def inv_scale_cost(self) -> jnp.ndarray:
    """Compute and return inverse of scaling factor for cost matrix."""
    if self._scale_cost == "max_cost":
      return 1.0 / jnp.max(self._cost_matrix)
    if self._scale_cost == "mean":
      return 1.0 / jnp.mean(self._cost_matrix)
    if self._scale_cost == "median":
      return 1.0 / jnp.median(self._cost_matrix)
    if utils.is_scalar(self._scale_cost):
      return 1.0 / self._scale_cost
    raise ValueError(f"Scaling {self._scale_cost} not implemented.")

  @property
  def diag_cost(self) -> jnp.ndarray:
    """Diagonal of the cost matrix."""
    assert self.is_square, "Cost matrix must be square to compute diagonal."
    return jnp.diag(self.cost_matrix)

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
    children, aux_data = self.tree_flatten()
    new_geom = type(self).tree_unflatten(aux_data, children)
    new_geom._epsilon_init = other.epsilon_scheduler
    new_geom._relative_epsilon = other._relative_epsilon  # has no effect
    return new_geom

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
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
      vec: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Apply :attr:`kernel_matrix` on positive scaling vector.

    Args:
      vec: jnp.ndarray [num_a or num_b] , scaling of size num_rows or
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

    return jnp.dot(kernel, vec)

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
    eps = self.epsilon_scheduler(iteration)
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
    eps = self.epsilon_scheduler(iteration)
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
      is_linear: bool = False,
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
      is_linear: Whether ``fn`` is linear.

    Returns:
      An array, [num_b, p] if axis=0 or [num_a, p] if axis=1
    """
    if arr.ndim == 1:
      return self._apply_cost_to_vec(arr, axis=axis, fn=fn, is_linear=is_linear)
    app = functools.partial(
        self._apply_cost_to_vec, axis=axis, fn=fn, is_linear=is_linear
    )
    return jax.vmap(app, in_axes=1, out_axes=1)(arr)

  def _apply_cost_to_vec(
      self,
      vec: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      is_linear: bool = False,
  ) -> jnp.ndarray:
    """Apply ``[num_a, num_b]`` fn(cost) (or transpose) to vector.

    Args:
      vec: jnp.ndarray [num_a,] ([num_b,] if axis=1) vector
      axis: axis on which the reduction is done.
      fn: function optionally applied to cost matrix element-wise, before the
        doc product
      is_linear: Whether ``fn`` is linear.

    Returns:
      A jnp.ndarray corresponding to cost x vector
    """
    del is_linear
    matrix = self.cost_matrix.T if axis == 0 else self.cost_matrix
    if fn is not None:
      matrix = fn(matrix)
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

      ci_star = self.subset(row_ixs=i_star).cost_matrix.ravel() ** 2  # (m,)
      cj_star = self.subset(col_ixs=j_star).cost_matrix.ravel() ** 2  # (n,)

      p_row = cj_star + ci_star[j_star] + jnp.mean(ci_star)  # (n,)
      p_row /= jnp.sum(p_row)
      row_ixs = jax.random.choice(rng3, n, shape=(n_subset,), p=p_row)
      # (n_subset, m)
      s = self.subset(row_ixs=row_ixs).cost_matrix
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
      A_trans = self.subset(col_ixs=col_ixs).cost_matrix * inv_scale
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
      self,
      row_ixs: Optional[jnp.ndarray] = None,
      col_ixs: Optional[jnp.ndarray] = None
  ) -> "Geometry":
    """Subset rows or columns of a geometry.

    Args:
      row_ixs: Row indices. If :obj:`None`, use all rows.
      col_ixs: Column indices. If :obj:`None`, use all columns.

    Returns:
      The subsetted geometry.
    """
    (cost, kernel, *rest), aux_data = self.tree_flatten()
    row_ixs = row_ixs if row_ixs is None else jnp.atleast_1d(row_ixs)
    col_ixs = col_ixs if col_ixs is None else jnp.atleast_1d(col_ixs)
    if cost is not None:
      cost = cost if row_ixs is None else cost[row_ixs]
      cost = cost if col_ixs is None else cost[:, col_ixs]
    if kernel is not None:
      kernel = kernel if row_ixs is None else kernel[row_ixs]
      kernel = kernel if col_ixs is None else kernel[:, col_ixs]
    return type(self).tree_unflatten(aux_data, (cost, kernel, *rest))

  @property
  def dtype(self) -> jnp.dtype:
    """The data type."""
    if self._cost_matrix is not None:
      return self._cost_matrix.dtype
    return self._kernel_matrix.dtype

  def tree_flatten(self):  # noqa: D102
    return (
        self._cost_matrix,
        self._kernel_matrix,
        self._epsilon_init,
    ), {
        "scale_cost": self._scale_cost,
        "relative_epsilon": self._relative_epsilon,
    }

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    cost, kernel, epsilon = children
    return cls(cost, kernel_matrix=kernel, epsilon=epsilon, **aux_data)
