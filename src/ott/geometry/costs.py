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
import abc
import functools
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from ott.math import fixed_point_loop, matrix_square_root
from ott.math import utils as mu

__all__ = [
    "PNormP",
    "SqPNorm",
    "Euclidean",
    "SqEuclidean",
    "Cosine",
    "ElasticL1",
    "ElasticSTVS",
    "ElasticSqKOverlap",
    "Bures",
    "UnbalancedBures",
    "SoftDTW",
]


@jax.tree_util.register_pytree_node_class
class CostFn(abc.ABC):
  """Base class for all costs.

  Cost functions evaluate a function on a pair of inputs. For convenience,
  that function is split into two norms -- evaluated on each input separately --
  followed by a pairwise cost that involves both inputs, as in:

  .. math::

    c(x,y) = norm(x) + norm(y) + pairwise(x,y)

  If the :attr:`norm` function is not implemented, that value is handled as
  :math:`0`, and only :func:`pairwise` is used.
  """

  # no norm function created by default.
  norm: Optional[Callable[[jnp.ndarray], Union[float, jnp.ndarray]]] = None

  @abc.abstractmethod
  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost between :math:`x` and :math:`y`.

    Args:
      x: Array.
      y: Array.

    Returns:
      The cost.
    """

  def barycenter(self, weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    """Barycentric operator.

    Args:
      weights: Convex set of weights.
      xs: Points.

    Returns:
      A list, whose first element is the barycenter of `xs` using `weights`
      coefficients, followed by auxiliary information on the convergence of
      the algorithm.
    """
    raise NotImplementedError("Barycenter is not implemented.")

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    """Create a padding vector of adequate dimension, well-suited to a cost.

    Args:
      dim: Dimensionality of the data.

    Returns:
      The padding vector.
    """
    return jnp.zeros((1, dim))

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost between :math:`x` and :math:`y`.

    Args:
      x: Array.
      y: Array.

    Returns:
      The cost, optionally including the :attr:`norms <norm>` of
      :math:`x`/:math:`y`.
    """
    cost = self.pairwise(x, y)
    if self.norm is None:
      return cost
    return cost + self.norm(x) + self.norm(y)

  # TODO(michalk8): unused
  def all_pairs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all pairwise costs, including the :attr:`norms <norm>`.

    Args:
      x: Array of shape ``[n, ...]``.
      y: Array of shape ``[m, ...]``.

    Returns:
      Array of shape ``[n, m]`` of cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self(x_, y_))(y))(x)

  def all_pairs_pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all pairwise costs, excluding the :attr:`norms <norm>`.

    Args:
      x: Array of shape ``[n, ...]``.
      y: Array of shape ``[m, ...]``.

    Returns:
      Array of shape ``[n, m]`` of cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self.pairwise(x_, y_))(y))(x)

  def tree_flatten(self):  # noqa: D102
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class TICost(CostFn):
  """Base class for translation invariant (TI) costs.

  Such costs are defined using a function :math:`h`, mapping vectors to
  real-values, to be used as:

  .. math::

    c(x,y) = h(z), z := x-y.

  If that cost function is used to form an Entropic map using the
  :cite:`brenier:91` theorem, then the user should ensure :math:`h` is
  strictly convex, as well as provide the Legendre transform of :math:`h`,
  whose gradient is necessarily the inverse of the gradient of :math:`h`.
  """

  @abc.abstractmethod
  def h(self, z: jnp.ndarray) -> float:
    """TI function acting on difference of :math:`x-y` to output cost."""

  def h_legendre(self, z: jnp.ndarray) -> float:
    """Legendre transform of :func:`h` when it is convex."""
    raise NotImplementedError("`h_legendre` not implemented.")

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost as evaluation of :func:`h` on :math:`x-y`."""
    return self.h(x - y)


@jax.tree_util.register_pytree_node_class
class SqPNorm(TICost):
  r"""Squared p-norm of the difference of two vectors.

  Args:
    p: Power of the p-norm, :math:`\ge 1`.
  """

  def __init__(self, p: float):
    super().__init__()
    self.p = p
    self.q = 1.0 / (1.0 - (1.0 / p)) if p > 1.0 else jnp.inf

  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    return 0.5 * jnp.linalg.norm(z, self.p) ** 2

  def h_legendre(self, z: jnp.ndarray) -> float:
    """Legendre transform of :func:`h`.

    For details on the derivation, see e.g., :cite:`boyd:04`, p. 93/94.
    """
    return 0.5 * jnp.linalg.norm(z, self.q) ** 2

  def tree_flatten(self):  # noqa: D102
    return (), (self.p,)

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del children
    return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
class PNormP(TICost):
  r"""p-norm to the power p (and divided by p) of the difference of two vectors.

  Args:
    p: Power of the p-norm in :math:`[1, +\infty)`.
      Note that :func:`h_legendre` is not defined for ``p = 1``.
  """

  def __init__(self, p: float):
    super().__init__()
    self.p = p
    self.q = 1.0 / (1.0 - (1.0 / p)) if p > 1.0 else jnp.inf

  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    return jnp.linalg.norm(z, self.p) ** self.p / self.p

  def h_legendre(self, z: jnp.ndarray) -> float:  # noqa: D102
    # not defined for `p=1`
    return jnp.linalg.norm(z, self.q) ** self.q / self.q

  def tree_flatten(self):  # noqa: D102
    return (), (self.p,)

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del children
    return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
class Euclidean(CostFn):
  """Euclidean distance.

  Note that the Euclidean distance is not cast as a
  :class:`~ott.geometry.costs.TICost`, since this would correspond to :math:`h`
  being :func:`jax.numpy.linalg.norm`, whose gradient is not invertible,
  because the function is not strictly convex (it is linear on rays).
  """

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute Euclidean norm."""
    return jnp.linalg.norm(x - y)


@jax.tree_util.register_pytree_node_class
class SqEuclidean(TICost):
  """Squared Euclidean distance."""

  def norm(self, x: jnp.ndarray) -> Union[float, jnp.ndarray]:
    """Compute squared Euclidean norm for vector."""
    return jnp.sum(x ** 2, axis=-1)

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute minus twice the dot-product between vectors."""
    return -2. * jnp.vdot(x, y)

  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    return jnp.sum(z ** 2)

  def h_legendre(self, z: jnp.ndarray) -> float:  # noqa: D102
    return 0.25 * jnp.sum(z ** 2)

  def barycenter(self, weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    """Output barycenter of vectors when using squared-Euclidean distance."""
    return jnp.average(xs, weights=weights, axis=0), None


@jax.tree_util.register_pytree_node_class
class Cosine(CostFn):
  """Cosine distance cost function.

  Args:
    ridge: Ridge regularization.
  """

  def __init__(self, ridge: float = 1e-8):
    super().__init__()
    self._ridge = ridge

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Cosine distance between vectors, denominator regularized with ridge."""
    ridge = self._ridge
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    cosine_similarity = jnp.vdot(x, y) / (x_norm * y_norm + ridge)
    cosine_distance = 1.0 - cosine_similarity
    # similarity is in [-1, 1], clip because of numerical imprecision
    return jnp.clip(cosine_distance, 0., 2.)

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    return jnp.ones((1, dim))


class RegTICost(TICost, abc.ABC):
  r"""Base class for regularized translation-invariant costs.

  .. math::

    \frac{1}{2} \|\cdot\|_2^2 + reg\left(\cdot\right)

  where :func:`reg` is the regularization function.
  """

  @abc.abstractmethod
  def reg(self, z: jnp.ndarray) -> float:
    """Regularization function."""

  def prox_reg(self, z: jnp.ndarray) -> jnp.ndarray:
    """Proximal operator of :func:`reg`."""
    raise NotImplementedError("Proximal operator is not implemented.")

  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    return 0.5 * jnp.linalg.norm(z, ord=2) ** 2 + self.reg(z)

  def h_legendre(self, z: jnp.ndarray) -> float:  # noqa: D102
    q = jax.lax.stop_gradient(self.prox_reg(z))
    return jnp.sum(q * z) - self.h(q)


@jax.tree_util.register_pytree_node_class
class ElasticL1(RegTICost):
  r"""Cost inspired by elastic net :cite:`zou:05` regularization.

  .. math::

    \frac{1}{2} \|\cdot\|_2^2 + \gamma \|\cdot\|_1

  Args:
    gamma: Strength of the :math:`\|\cdot\|_1` regularization, :math:`\ge 0`.
  """

  def __init__(self, gamma: float = 1.0):
    super().__init__()
    self.gamma = gamma

  def reg(self, z: jnp.ndarray) -> float:  # noqa: D102
    return self.gamma * jnp.linalg.norm(z, ord=1)

  def prox_reg(self, z: jnp.ndarray) -> float:  # noqa: D102
    return jnp.sign(z) * jax.nn.relu(jnp.abs(z) - self.gamma)

  def tree_flatten(self):  # noqa: D102
    return (self.gamma,), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class ElasticSTVS(RegTICost):
  r"""Cost with soft thresholding operator with vanishing shrinkage (STVS)
  :cite:`schreck:15` regularization.

  .. math::

    \frac{1}{2} \|\cdot\|_2^2 + \gamma^2\mathbf{1}_d^T\left(\sigma(\cdot) -
    \frac{1}{2} \exp\left(-2\sigma(\cdot)\right) + \frac{1}{2}\right)

  where :math:`\sigma(\cdot) := \text{asinh}\left(\frac{\cdot}{2\gamma}\right)`

  Args:
    gamma: Strength of the STVS regularization, :math:`> 0`.
  """  # noqa

  def __init__(self, gamma: float = 1.0):
    super().__init__()
    self.gamma = gamma

  def reg(self, z: jnp.ndarray) -> float:  # noqa: D102
    u = jnp.arcsinh(jnp.abs(z) / (2 * self.gamma))
    out = u - 0.5 * jnp.exp(-2.0 * u)
    return (self.gamma ** 2) * jnp.sum(out + 0.5)  # make positive

  def prox_reg(self, z: jnp.ndarray) -> float:  # noqa: D102
    return jax.nn.relu(1 - (self.gamma / (jnp.abs(z) + 1e-12)) ** 2) * z

  def tree_flatten(self):  # noqa: D102
    return (self.gamma,), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class ElasticSqKOverlap(RegTICost):
  r"""Cost with squared k-overlap norm regularization :cite:`argyriou:12`.

  .. math::

    \frac{1}{2} \|\cdot\|_2^2 + \frac{1}{2} \gamma \|\cdot\|_{ovk}^2

  where :math:`\|\cdot\|_{ovk}^2` is the squared k-overlap norm,
  see def. 2.1 of :cite:`argyriou:12`.

  Args:
    k: Number of groups. Must be in ``[0, d)`` where :math:`d` is the
      dimensionality of the data.
    gamma: Strength of the squared k-overlap norm regularization, :math:`> 0`.
  """

  def __init__(self, k: int, gamma: float = 1.0):
    super().__init__()
    self.k = k
    self.gamma = gamma

  def reg(self, z: jnp.ndarray) -> float:  # noqa: D102
    # Prop 2.1 in :cite:`argyriou:12`
    k = self.k
    top_w = jax.lax.top_k(jnp.abs(z), k)[0]  # Fetch largest k values
    top_w = jnp.flip(top_w)  # Sort k-largest from smallest to largest
    # sum (dim - k) smallest values
    sum_bottom = jnp.sum(jnp.abs(z)) - jnp.sum(top_w)
    cumsum_top = jnp.cumsum(top_w)
    # Cesaro mean of top_w (each term offset with sum_bottom).
    cesaro = sum_bottom + cumsum_top
    cesaro /= jnp.arange(k) + 1
    # Choose first index satisfying constraint in Prop 2.1
    lower_bound = cesaro - top_w >= 0
    # Last upper bound is always True.
    upper_bound = jnp.concatenate(((top_w[1:] - cesaro[:-1] > 0),
                                   jnp.array((True,))))
    r = jnp.argmax(lower_bound * upper_bound)
    s = jnp.sum(jnp.where(jnp.arange(k) < k - r - 1, jnp.flip(top_w) ** 2, 0))

    return 0.5 * self.gamma * (s + (r + 1) * cesaro[r] ** 2)

  def prox_reg(self, z: jnp.ndarray) -> float:  # noqa: D102

    @functools.partial(jax.vmap, in_axes=[0, None, None])
    def find_indices(r: int, l: jnp.ndarray,
                     z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

      @functools.partial(jax.vmap, in_axes=[None, 0, None])
      def inner(r: int, l: int,
                z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        i = k - r - 1
        res = jnp.sum(z * ((i <= ixs) & (ixs < l)))
        res /= l - k + (beta + 1) * r + beta + 1

        cond1_left = jnp.logical_or(i == 0, (z[i - 1] / beta + 1) > res)
        cond1_right = res >= (z[i] / (beta + 1))
        cond1 = jnp.logical_and(cond1_left, cond1_right)

        cond2_left = z[l - 1] > res
        cond2_right = jnp.logical_or(l == d, res >= z[l])
        cond2 = jnp.logical_and(cond2_left, cond2_right)

        return res, cond1 & cond2

      return inner(r, l, z)

    # Alg. 1 of :cite:`argyriou:12`
    k, d, beta = self.k, z.shape[-1], 1.0 / self.gamma

    ixs = jnp.arange(d)
    z, sgn = jnp.abs(z), jnp.sign(z)
    z_ixs = jnp.argsort(z)[::-1]
    z_sorted = z[z_ixs]

    # (k, d - k + 1)
    T, mask = find_indices(jnp.arange(k), jnp.arange(k, d + 1), z_sorted)
    (r,), (l,) = jnp.where(mask, size=1)  # size=1 for jitting
    T = T[r, l]

    q1 = (beta / (beta + 1)) * z_sorted * (ixs < (k - r - 1))
    q2 = (z_sorted - T) * jnp.logical_and((k - r - 1) <= ixs, ixs < (l + k))
    q = q1 + q2

    # change sign and reorder
    return sgn * q[jnp.argsort(z_ixs.astype(float))]

  def tree_flatten(self):  # noqa: D102
    return (self.gamma,), {"k": self.k}

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(**aux_data, gamma=children[0])


@jax.tree_util.register_pytree_node_class
class Bures(CostFn):
  """Bures distance between a pair of (mean, covariance matrix).

  Args:
    dimension: Dimensionality of the data.
    sqrtm_kw: Dictionary of keyword arguments to control the
      behavior of inner calls to :func:`~ott.math.matrix_square_root.sqrtm`.
    kwargs: keyword arguments to control the behavior of the fixed-point
      iterations used in the inner computation of barycenters of Gaussians.
  """

  def __init__(self, dimension: int, sqrtm_kw: Dict[str, Any] = None, **kwargs):
    super().__init__()
    self._dimension = dimension
    self._sqrtm_kw = sqrtm_kw if sqrtm_kw is not None else {}

  def norm(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute norm of Gaussian, sq. 2-norm of mean + trace of covariance."""
    mean, cov = x_to_means_and_covs(x, self._dimension)
    norm = jnp.sum(mean ** 2, axis=-1)
    norm += jnp.trace(cov, axis1=-2, axis2=-1)
    return norm

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute - 2 x Bures dot-product."""
    mean_x, cov_x = x_to_means_and_covs(x, self._dimension)
    mean_y, cov_y = x_to_means_and_covs(y, self._dimension)
    mean_dot_prod = jnp.vdot(mean_x, mean_y)
    sq_x = matrix_square_root.sqrtm(cov_x, self._dimension, **self._sqrtm_kw)[0]
    sq_x_y_sq_x = jnp.matmul(sq_x, jnp.matmul(cov_y, sq_x))
    sq__sq_x_y_sq_x = matrix_square_root.sqrtm(
        sq_x_y_sq_x, self._dimension, **self._sqrtm_kw
    )[0]
    return -2 * (mean_dot_prod + jnp.trace(sq__sq_x_y_sq_x, axis1=-2, axis2=-1))

  def covariance_fixpoint_iter(
      self,
      covs: jnp.ndarray,
      weights: jnp.ndarray,
      tolerance: float = 1e-4,
      sqrtm_kw: Dict[Any, Any] = None,
      **kwargs
  ) -> jnp.ndarray:
    """Iterate fix-point updates to compute barycenter of Gaussians.

    Args:
      covs: [batch, d^2] covariance matrices
      weights: simplicial weights (non-negative, sum to 1)
      tolerance: tolerance of the fixed-point procedure. That tolerance is
        applied to the Frobenius norm (normalized by total size)
        of two successive iterations of the algorithm
      sqrtm_kw: keyword arguments for :func:`ott.math.matrix_square_root.sqrtm`
      kwargs: keyword arguments for the outer fixed-point iteration

    Returns:
      Weighted Bures average of the covariance matrices.
    """
    sqrtm_kw = {} if sqrtm_kw is None else sqrtm_kw
    min_iterations = kwargs.pop("min_iterations", 1)
    max_iterations = kwargs.pop("max_iterations", 100)
    inner_iterations = kwargs.pop("inner_iterations", 5)
    dtype = covs.dtype

    @functools.partial(jax.vmap, in_axes=[None, 0, 0])
    def scale_covariances(
        cov_sqrt: jnp.ndarray, cov: jnp.ndarray, weight: jnp.ndarray
    ) -> jnp.ndarray:
      """Rescale covariance in barycenter step."""
      return weight * matrix_square_root.sqrtm_only((cov_sqrt @ cov) @ cov_sqrt,
                                                    **sqrtm_kw)

    def cond_fn(iteration: int, constants: Tuple[Any, ...], state) -> bool:
      del constants
      _, diffs = state
      return diffs[iteration // inner_iterations] > tolerance

    def body_fn(
        iteration: int, constants: Tuple[Any, ...],
        state: Tuple[jnp.ndarray, float], compute_error: bool
    ) -> Tuple[jnp.ndarray, float]:
      del constants, compute_error
      cov, diffs = state
      cov_sqrt, cov_inv_sqrt, _ = matrix_square_root.sqrtm(cov, **sqrtm_kw)
      scaled_cov = jnp.linalg.matrix_power(
          jnp.sum(scale_covariances(cov_sqrt, covs, weights), axis=0), 2
      )
      next_cov = (cov_inv_sqrt @ scaled_cov) @ cov_inv_sqrt
      diff = jnp.sum((next_cov - cov) ** 2) / jnp.prod(jnp.array(cov.shape))
      diffs = diffs.at[iteration // inner_iterations].set(diff)
      return next_cov, diffs

    def init_state() -> Tuple[jnp.ndarray, float]:
      cov_init = jnp.eye(self._dimension)
      diffs = -jnp.ones(
          (np.ceil(max_iterations / inner_iterations).astype(int),),
          dtype=dtype
      )
      return cov_init, diffs

    cov, diffs = fixed_point_loop.fixpoint_iter(
        cond_fn=cond_fn,
        body_fn=body_fn,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        inner_iterations=inner_iterations,
        constants=(),
        state=init_state(),
    )
    return cov, diffs

  def barycenter(
      self,
      weights: jnp.ndarray,
      xs: jnp.ndarray,
      tolerance: float = 1e-4,
      sqrtm_kw: Dict[Any, Any] = None,
      **kwargs
  ) -> jnp.ndarray:
    """Compute the Bures barycenter of weighted Gaussian distributions.

    Implements the fixed point approach proposed in :cite:`alvarez-esteban:16`
    for the computation of the mean and the covariance of the barycenter of
    weighted Gaussian distributions.

    Args:
      weights: The barycentric weights.
      xs: The points to be used in the computation of the barycenter, where
        each point is described by a concatenation of the mean and the
        covariance (raveled).
      tolerance: convergence tolerance to control the termination of the
        algorithm.
      sqrtm_kw: Arguments passed on to the
        :func:`ott.math.matrix_square_root.sqrtm` function used within
        :meth:`covariance_fixpoint_iter`. This defines the precision
        (in terms of convergence threshold, and number of iterations) of the
        matrix square root call. That call is used at each outer iteration of
        the computation of Gaussian barycenters. These values are by default, if
        not passed, the same as those used to compute the Bures distance.
      kwargs: Passed on to :meth:`covariance_fixpoint_iter`, to specify the
        number of iterations and tolerance of the fixed-point iteration of the
        barycenter routine, by parameterizing `tolerance` and other relevant
        arguments passed on to :meth:`ott.math.fixed_point_loop.fixpoint_iter`,
        namely `min_iterations`, `max_iterations` and `inner_iterations`.

    Returns:
      A list holding a concatenation of the mean and the raveled covariance
      of the barycenter as its first element, followed by a vector of
      norms of successive differences in iterates.
    """
    # Ensure that barycentric weights sum to 1.
    weights = weights / jnp.sum(weights)
    mus, covs = x_to_means_and_covs(xs, self._dimension)
    mu_bary = jnp.sum(weights[:, None] * mus, axis=0)
    cov_bary, diffs = self.covariance_fixpoint_iter(
        covs=covs,
        weights=weights,
        tolerance=tolerance,
        sqrtm_kw=sqrtm_kw if sqrtm_kw is not None else self._sqrtm_kw,
        **kwargs
    )
    return mean_and_cov_to_x(mu_bary, cov_bary, self._dimension), diffs

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    dimension = int((-1 + math.sqrt(1 + 4 * dim)) / 2)
    padding = mean_and_cov_to_x(
        jnp.zeros((dimension,)), jnp.eye(dimension), dimension
    )
    return padding[jnp.newaxis, :]

  def tree_flatten(self):  # noqa: D102
    return (), (self._dimension, self._sqrtm_kw)

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del children
    return cls(aux_data[0], **aux_data[1])


@jax.tree_util.register_pytree_node_class
class UnbalancedBures(CostFn):
  """Unbalanced Bures distance between two triplets of `(mass, mean, cov)`.

  This cost uses the notation defined in :cite:`janati:20`, eq. 37, 39, 40.

  Args:
    dimension: Dimensionality of the data.
    sigma: Entropic regularization.
    gamma: KL-divergence regularization for the marginals.
    kwargs: Keyword arguments for :func:`~ott.math.matrix_square_root.sqrtm`.
  """

  def __init__(
      self,
      dimension: int,
      *,
      sigma: float = 1.0,
      gamma: float = 1.0,
      **kwargs: Any,
  ):
    super().__init__()
    self._dimension = dimension
    self._sigma = sigma
    self._gamma = gamma
    self._sqrtm_kw = kwargs

  def norm(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute norm of Gaussian for unbalanced Bures.

    Args:
      x: Array of shape ``[n_points + n_points + n_dim ** 2,]``, potentially
        batched, corresponding to the raveled mass, means and the covariance
        matrix.

    Returns:
      The norm, array of shape ``[]`` or ``[batch,]`` in the batched case.
    """
    return self._gamma * x[..., 0]

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute dot-product for unbalanced Bures.

    Args:
      x: Array of shape ``[n_points + n_points + n_dim ** 2,]``
        corresponding to the raveled mass, means and the covariance matrix.
      y: Array of shape ``[n_points + n_points + n_dim ** 2,]``
        corresponding to the raveled mass, means and the covariance matrix.

    Returns:
      The cost.
    """
    # Sets a few constants
    gam = self._gamma
    sig2 = self._sigma ** 2
    lam = sig2 + gam / 2.0
    tau = gam / (2.0 * lam)

    # Extracts mass, mean vector, covariance matrices
    mass_x, mass_y = x[0], y[0]
    mean_x, cov_x = x_to_means_and_covs(x[1:], self._dimension)
    mean_y, cov_y = x_to_means_and_covs(y[1:], self._dimension)

    diff_means = mean_x - mean_y

    # Identity matrix of suitable size
    iden = jnp.eye(self._dimension, dtype=x.dtype)

    # Creates matrices needed in the computation
    tilde_a = 0.5 * gam * (iden - lam * jnp.linalg.inv(cov_x + lam * iden))
    tilde_b = 0.5 * gam * (iden - lam * jnp.linalg.inv(cov_y + lam * iden))

    tilde_a_b = jnp.matmul(tilde_a, tilde_b)
    c_mat = matrix_square_root.sqrtm(
        1 / tau * tilde_a_b + 0.25 * (sig2 ** 2) * iden, **self._sqrtm_kw
    )[0]
    c_mat -= 0.5 * sig2 * iden

    # Computes log determinants (their sign should be >0).
    sldet_c, ldet_c = jnp.linalg.slogdet(c_mat)
    sldet_t_ab, ldet_t_ab = jnp.linalg.slogdet(tilde_a_b)
    sldet_ab, ldet_ab = jnp.linalg.slogdet(jnp.matmul(cov_x, cov_y))
    sldet_c_ab, ldet_c_ab = jnp.linalg.slogdet(c_mat - 2.0 * tilde_a_b / gam)

    # Gathers all these results to compute log total mass of transport
    log_m_pi = (0.5 * self._dimension * sig2 / (gam + sig2)) * jnp.log(sig2)
    log_m_pi += (1.0 / (tau + 1.0)) * (
        jnp.log(mass_x) + jnp.log(mass_y) + ldet_c + 0.5 *
        (tau * ldet_t_ab - ldet_ab)
    )
    log_m_pi += -jnp.sum(
        diff_means * jnp.linalg.solve(cov_x + cov_y + lam * iden, diff_means)
    ) / (2.0 * (tau + 1.0))
    log_m_pi += -0.5 * ldet_c_ab

    # if all logdet signs are 1, output value, nan otherwise
    pos_signs = (sldet_c + sldet_c_ab + sldet_t_ab + sldet_t_ab) == 4

    return jax.lax.cond(
        pos_signs, lambda: 2 * sig2 * mass_x * mass_y - 2 *
        (sig2 + gam) * jnp.exp(log_m_pi), lambda: jnp.nan
    )

  def tree_flatten(self):  # noqa: D102
    return (), (self._dimension, self._sigma, self._gamma, self._sqrtm_kw)

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    del children
    dim, sigma, gamma, kwargs = aux_data
    return cls(dim, sigma=sigma, gamma=gamma, **kwargs)


@jax.tree_util.register_pytree_node_class
class SoftDTW(CostFn):
  """Soft dynamic time warping (DTW) cost :cite:`cuturi:17`.

  Args:
    gamma: Smoothing parameter :math:`> 0` for the soft-min operator.
    ground_cost: Ground cost function. If ``None``,
      use :class:`~ott.geometry.costs.SqEuclidean`.
    debiased: Whether to compute the debiased soft-DTW :cite:`blondel:21`.
  """

  def __init__(
      self,
      gamma: float,
      ground_cost: Optional[CostFn] = None,
      debiased: bool = False
  ):
    self.gamma = gamma
    self.ground_cost = SqEuclidean() if ground_cost is None else ground_cost
    self.debiased = debiased

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:  # noqa: D102
    c_xy = self._soft_dtw(x, y)
    if self.debiased:
      return c_xy - 0.5 * (self._soft_dtw(x, x) + self._soft_dtw(y, y))
    return c_xy

  def _soft_dtw(self, t1: jnp.ndarray, t2: jnp.ndarray) -> float:

    def body(
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        current_antidiagonal: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
      # modified from: https://github.com/khdlr/softdtw_jax
      two_ago, one_ago = carry

      diagonal, right, down = two_ago[:-1], one_ago[:-1], one_ago[1:]
      best = mu.softmin(
          jnp.stack([diagonal, right, down], axis=-1), self.gamma, axis=-1
      )

      next_row = best + current_antidiagonal
      next_row = jnp.pad(next_row, (1, 0), constant_values=jnp.inf)

      return (one_ago, next_row), next_row

    t1 = t1[:, None] if t1.ndim == 1 else t1
    t2 = t2[:, None] if t2.ndim == 1 else t2
    dist = self.ground_cost.all_pairs(t1, t2)

    n, m = dist.shape
    if n < m:
      dist = dist.T
      n, m = m, n

    model_matrix = jnp.full((n + m - 1, n), fill_value=jnp.inf)
    mask = np.tri(n + m - 1, n, k=0, dtype=bool)
    mask = mask & mask[::-1, ::-1]
    model_matrix = model_matrix.T.at[mask.T].set(dist.ravel()).T

    init = (
        jnp.pad(model_matrix[0], (1, 0), constant_values=jnp.inf),
        jnp.pad(
            model_matrix[1] + model_matrix[0, 0], (1, 0),
            constant_values=jnp.inf
        )
    )

    (_, carry), _ = jax.lax.scan(body, init, model_matrix[2:])
    return carry[-1]

  def tree_flatten(self):  # noqa: D102
    return (self.gamma, self.ground_cost), {"debiased": self.debiased}

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)


def x_to_means_and_covs(x: jnp.ndarray,
                        dimension: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Extract means and covariance matrices of Gaussians from raveled vector.

  Args:
    x: [num_gaussians, dimension, (1 + dimension)] array of concatenated means
      and covariances (raveled) dimension: the dimension of the Gaussians.
    dimension: Dimensionality of the Gaussians.

  Returns:
    Means and covariances of shape ``[num_gaussian, dimension]``.
  """
  x = jnp.atleast_2d(x)
  means = x[:, :dimension]
  covariances = jnp.reshape(
      x[:, dimension:dimension + dimension ** 2], (-1, dimension, dimension)
  )
  return jnp.squeeze(means), jnp.squeeze(covariances)


def mean_and_cov_to_x(
    mean: jnp.ndarray, covariance: jnp.ndarray, dimension: int
) -> jnp.ndarray:
  """Ravel a Gaussian's mean and covariance matrix to d(1 + d) vector."""
  return jnp.concatenate(
      (mean, jnp.reshape(covariance, (dimension * dimension)))
  )
