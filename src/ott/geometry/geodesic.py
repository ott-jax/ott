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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import eigsh as _scipy_eigsh
from scipy.special import ive

from ott import utils
from ott.geometry import geometry
from ott.math import utils as mu

__all__ = ["Geodesic"]


@jax.tree_util.register_pytree_node_class
class Geodesic(geometry.Geometry):
  r"""Graph distance approximation using heat kernel :cite:`huguet:2022`.

  Approximates the heat-geodesic kernel using thee Chebyshev polynomials of the
  first kind of max order ``order``, which for small ``t`` approximates the
  geodesic exponential kernel :math:`e^{\frac{-d(x, y)^2}{t}}`.

  Args:
    laplacian: Symmetric graph Laplacian.
    t: Time parameter for heat kernel.
    order: Max order of Chebyshev polynomial.
    kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      laplacian: jnp.ndarray,
      t: float = 1e-3,
      order: int = 100,
      chebyshev_coeffs: Optional[List[float]] = None,
      lap_min_id: Optional[jnp.ndarray
                          ] = None,  # Rescale Laplacian minus identity
      eigval: Optional[jnp.ndarray
                      ] = None,  # (Second)Largest eigenvalue of Laplacian
      **kwargs: Any
  ):
    super().__init__(epsilon=1., **kwargs)
    self.laplacian = laplacian
    self.t = t
    self.order = order
    self.chebyshev_coeffs = chebyshev_coeffs
    self.eigval = eigval
    self.lap_min_id = lap_min_id

  @classmethod
  def from_graph(
      cls,
      G: jnp.ndarray,
      t: Optional[float] = 1e-3,
      order: int = 100,
      directed: bool = False,
      normalize: bool = False,
      eigenval_scipy: bool = False,
      **kwargs: Any
  ) -> "Geodesic":
    r"""Construct a Geodesic geometry from an adjacency matrix.

    Args:
      G: Adjacency matrix.
      t: Time parameter for approximating the geodesic exponential kernel.
        If `None`, it defaults to :math:`\frac{1}{|E|} \sum_{(u, v) \in E}
        \text{weight}(u, v)` :cite:`crane:13`. In this case, the ``graph``
        must be specified and the edge weights are assumed to be positive.
      order: Max order of Chebyshev polynomial.
      directed: Whether the ``graph`` is directed. If not, it's made
        undirected as :math:`G + G^T`. This parameter is ignored when passing
        the Laplacian directly, assumed to be symmetric.
      normalize: Whether to normalize the Laplacian as
        :math:`L^{sym} = \left(D^+\right)^{\frac{1}{2}} L
        \left(D^+\right)^{\frac{1}{2}}`, where :math:`L` is the
        non-normalized Laplacian and :math:`D` is the degree matrix.
      eigenval_scipy: Whether to use the scipy implementation of the
        eigenvalue computation.
      kwargs: Keyword arguments for the Geodesic class.

    Returns:
      The Geodesic geometry.
    """
    assert G.shape[0] == G.shape[1], G.shape

    if directed:
      G = G + G.T

    degree = jnp.sum(G, axis=1)
    laplacian = jnp.diag(degree) - G

    if normalize:
      inv_sqrt_deg = jnp.diag(
          jnp.where(degree > 0.0, 1.0 / jnp.sqrt(degree), 0.0)
      )
      laplacian = inv_sqrt_deg @ laplacian @ inv_sqrt_deg
    if eigenval_scipy:
      eigval = compute_eigenvalue(laplacian)
    else:
      eigval = compute_largest_eigenvalue(laplacian, k=1)
    rescaled_laplacian = rescale_laplacian(laplacian, eigval)
    lap_min_id = define_scaled_laplacian(
        rescaled_laplacian
    )  # TODO: remove if not needed.

    if t is None:
      t = (jnp.sum(G) / jnp.sum(G > 0.)) ** 2

    # Compute the coeffs of the Chebyshev pols approx using Bessel functs.
    chebyshev_coeffs = compute_chebychev_coeff_all(eigval, t, order)

    return cls(
        laplacian=laplacian,
        t=t,
        order=order,
        chebyshev_coeffs=chebyshev_coeffs,
        lap_min_id=lap_min_id,
        eigval=eigval,
        **kwargs
    )

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    # TODO: fix indentation
    # NOTE: GH: We could also input time,
    # since we only need to recompute the coeffs,
    # i.e. we can use the same laplacian, scales laplaciant for different times.
    r"""Apply :attr:`kernel_matrix` on positive scaling vector.

    Args:
      scaling: Scaling to apply the kernel to.
      eps: passed for consistency, not used yet.
      axis: passed for consistency, not used yet.

    Returns:
      Kernel applied to ``scaling``.
    """
    return expm_multiply(
        self.laplacian, scaling, self.chebyshev_coeffs, self.t, self.eigval,
        self.order
    )

  @property
  def kernel_matrix(self) -> jnp.ndarray:  # noqa: D102
    n, _ = self.shape
    kernel = self.apply_kernel(jnp.eye(n))
    # check if the kernel is symmetric
    if jnp.any(kernel != kernel.T):
      kernel = (kernel + kernel.T) / 2.0
    return kernel

  @property
  def cost_matrix(self) -> jnp.ndarray:  # noqa: D102
    # Calculate the cost matrix using the formula (5) from the main reference
    return -4 * self.t * mu.safe_log(self.kernel_matrix)

  @property
  def shape(self) -> Tuple[int, int]:  # noqa: D102
    return self.laplacian.shape

  @property
  def is_symmetric(self) -> bool:  # noqa: D102
    return True

  @property
  def dtype(self) -> jnp.dtype:  # noqa: D102
    return self.laplacian.dtype

  def transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray
  ) -> jnp.ndarray:
    """Not implemented."""
    raise ValueError("Not implemented.")

  def apply_transport_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    """Since applying from potentials is not feasible in grids, use scalings."""
    u, v = self.scaling_from_potential(f), self.scaling_from_potential(g)
    return self.apply_transport_from_scalings(u, v, vec, axis=axis)

  def marginal_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Not implemented."""
    raise ValueError("Not implemented.")

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [self.laplacian, self.t, self.order], {
        "chebyshev_coeffs": self.chebyshev_coeffs,
    }

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "Geodesic":
    return cls(*children, **aux_data)


# TODO:
# Moving some function here for now, idk if we want them in the class
# or in a utils file.


def compute_largest_eigenvalue(laplacian_matrix, k, rng=None):
  # Compute the largest eigenvalue of the Laplacian matrix.
  if rng is None:
    rng = utils.default_prng_key(rng)
  n = laplacian_matrix.shape[0]
  # Generate random initial directions for eigenvalue computation
  initial_dirs = jax.random.normal(rng, (n, k))

  # Create a sparse matrix-vector product function using sparsify
  # This function multiplies the sparse laplacian_matrix with a vector
  lapl_vector_product = jesp.sparsify(lambda v: laplacian_matrix @ v)

  # Compute eigenvalues using the sparse matrix-vector product
  eigvals, _, _ = jesp.linalg.lobpcg_standard(
      lapl_vector_product, initial_dirs, m=k
  )

  return jnp.max(eigvals)


def rescale_laplacian(
    laplacian_matrix: jnp.ndarray, largest_eigenvalue: jnp.ndarray
) -> jnp.ndarray:
  # Rescale the Laplacian matrix.
  if largest_eigenvalue > 2:
    rescaled_laplacian = laplacian_matrix.copy()
    rescaled_laplacian /= largest_eigenvalue
    return 2 * rescaled_laplacian
  return laplacian_matrix


def define_scaled_laplacian(laplacian_matrix: jnp.ndarray) -> jnp.ndarray:
  # Define the scaled Laplacian matrix.
  n = laplacian_matrix.shape[0]
  identity = jnp.eye(n)
  return laplacian_matrix - identity


def _scipy_compute_chebychev_coeff_all(phi, tau, K):
  """Compute the K+1 Chebychev coefficients for our functions."""
  coeff = 2 * ive(np.arange(0, K + 1), -tau * phi)
  if coeff.dtype == np.float64:
    coeff = np.float32(coeff)
  return coeff


def expm_multiply(
    L,
    X,
    coeff,
    phi,
    tau,
    K=None,
):
  # Initialize the accumulator with only the first coeff*polynomial
  T0 = X
  Y = 0.5 * coeff[0] * T0
  # Add the second coeff*polynomial to the accumulator
  T1 = (1 / phi) * L @ X - T0
  Y = Y + coeff[1] * T1
  # Recursively add the next coeff*polynomial
  for j in range(2, K + 1):
    T2 = (2 / phi) * L @ T1 - 2 * T1 - T0
    Y = Y + coeff[j] * T2
    T0 = T1
    T1 = T2
  return Y


def compute_chebychev_coeff_all(phi, tau, K):
  """Jax wrapper to compute the K+1 Chebychev coefficients."""
  if not isinstance(phi, jnp.ndarray):
    phi = jnp.asarray(phi)

  result_shape_dtype = jax.ShapeDtypeStruct(
      shape=(K + 1,),
      dtype=jax.numpy.float32,
  )  # TODO: not sure about the best type here. Maybe the best if to have
  # the same type as the laplacian.
  return jax.pure_callback(
      _scipy_compute_chebychev_coeff_all, result_shape_dtype, phi, tau, K
  )


def compute_eigenvalue(L):
  """Jax wrapper to compute the largest eigenvalue of the Laplacian."""
  result_shape_dtype = jax.ShapeDtypeStruct(
      shape=(1,),
      dtype=jax.numpy.float32,
  )
  eval_only = lambda x: _scipy_eigsh(x, k=1)[0] / 2.0
  return jax.pure_callback(eval_only, result_shape_dtype, L)
