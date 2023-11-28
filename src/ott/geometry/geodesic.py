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
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import numpy as np
from scipy.special import ive

from ott.geometry import geometry
from ott.math import utils as mu
from ott.types import Array_g
from ott.utils import default_prng_key

__all__ = ["Geodesic"]


@jax.tree_util.register_pytree_node_class
class Geodesic(geometry.Geometry):
  r"""Graph distance approximation using heat kernel :cite:`huguet:2022`.

  important::
  This constructor is not meant to be called by the user,
  please use the :meth:`from_graph` method instead.

  Approximates the heat kernel using `Chebyshev polynomials
  <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_ of the first kind of
  max order ``order``, which for small ``t`` approximates the
  geodesic exponential kernel :math:`e^{\frac{-d(x, y)^2}{t}}`.

  Args:
    scaled_laplacian: The Laplacian scaled by the largest eigenvalue.
    eigval: The largest eigenvalue of the Laplacian.
    chebyshev_coeffs: Coefficients of the Chebyshev polynomials.
    t: Time parameter for the heat kernel.
    kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      scaled_laplacian: Array_g,
      eigval: jnp.ndarray,
      chebyshev_coeffs: jnp.ndarray,
      t: float = 1e-3,
      **kwargs: Any
  ):
    super().__init__(epsilon=1., **kwargs)
    self.scaled_laplacian = scaled_laplacian
    self.eigval = eigval
    self.chebyshev_coeffs = chebyshev_coeffs
    self.t = t

  @classmethod
  def from_graph(
      cls,
      G: Array_g,
      t: Optional[float] = 1e-3,
      eigval: Optional[jnp.ndarray] = None,
      order: int = 100,
      directed: bool = False,
      normalize: bool = False,
      **kwargs: Any
  ) -> "Geodesic":
    r"""Construct a Geodesic geometry from an adjacency matrix.

    Args:
      G: Adjacency matrix.
      t: Time parameter for approximating the geodesic exponential kernel.
        If `None`, it defaults to :math:`\frac{1}{|E|} \sum_{(u, v) \in E}
        \text{weight}(u, v)` :cite:`crane:13`. In this case, the ``graph``
        must be specified and the edge weights are assumed to be positive.
      eigval: Largest eigenvalue of the Laplacian. If `None`, it's computed
        at initialization.
      order: Max order of Chebyshev polynomials.
      directed: Whether the ``graph`` is directed. If `True`, it's made
        undirected as :math:`G + G^T`. This parameter is ignored when passing
        the Laplacian directly, assumed to be symmetric.
      normalize: Whether to normalize the Laplacian as
        :math:`L^{sym} = \left(D^+\right)^{\frac{1}{2}} L
        \left(D^+\right)^{\frac{1}{2}}`, where :math:`L` is the
        non-normalized Laplacian and :math:`D` is the degree matrix.
      kwargs: Keyword arguments for :class:`~ott.geometry.geodesic.Geodesic`.

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

    eigval = compute_largest_eigenvalue(laplacian) if eigval is None else eigval

    scaled_laplacian = jax.lax.cond((eigval > 2.0), lambda l: 2.0 * l / eigval,
                                    lambda l: l, laplacian)

    if t is None:
      t = (jnp.sum(G) / jnp.sum(G > 0.)) ** 2.0

    # Compute the coeffs of the Chebyshev pols approx using Bessel functs.
    chebyshev_coeffs = compute_chebychev_coeff_all(
        eigval, t, order, laplacian.dtype
    )

    return cls(
        scaled_laplacian=scaled_laplacian,
        eigval=eigval,
        chebyshev_coeffs=chebyshev_coeffs,
        t=t,
        **kwargs
    )

  def apply_kernel(
      self,
      scaling: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    r"""Apply :attr:`kernel_matrix` on positive scaling vector.

    Args:
      scaling: Scaling to apply the kernel to.
      eps: passed for consistency, not used yet.
      axis: passed for consistency, not used yet.

    Returns:
      Kernel applied to ``scaling``.
    """
    return expm_multiply(
        self.scaled_laplacian, scaling, self.chebyshev_coeffs, self.eigval
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
    return -4.0 * self.t * mu.safe_log(self.kernel_matrix)

  @property
  def shape(self) -> Tuple[int, int]:  # noqa: D102
    return self.scaled_laplacian.shape

  @property
  def is_symmetric(self) -> bool:  # noqa: D102
    return True

  @property
  def dtype(self) -> jnp.dtype:  # noqa: D102
    return self.scaled_laplacian.dtype

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
    return [
        self.scaled_laplacian,
        self.eigval,
        self.chebyshev_coeffs,
        self.t,
    ], {}

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "Geodesic":
    return cls(*children, **aux_data)


def compute_largest_eigenvalue(
    laplacian_matrix: jnp.ndarray, rng: Optional[jax.Array] = None
) -> float:
  # Compute the largest eigenvalue of the Laplacian matrix.
  n = laplacian_matrix.shape[0]
  # Generate random initial directions for eigenvalue computation
  initial_dirs = jax.random.normal(default_prng_key(rng), (n, 1))

  # Create a sparse matrix-vector product function using sparsify
  # This function multiplies the sparse laplacian_matrix with a vector
  lapl_vector_product = jesp.sparsify(lambda v: laplacian_matrix @ v)

  # Compute eigenvalues using the sparse matrix-vector product
  eigvals, _, _ = jesp.linalg.lobpcg_standard(
      lapl_vector_product,
      initial_dirs,
  )
  return jnp.max(eigvals)


def expm_multiply(
    L: jnp.ndarray, X: jnp.ndarray, coeff: jnp.ndarray, eigval: float
) -> jnp.ndarray:

  def body(carry, c):
    T0, T1, Y = carry
    T2 = (2.0 / eigval) * L @ T1 - 2.0 * T1 - T0
    Y = Y + c * T2
    return (T1, T2, Y), None

  T0 = X
  Y = 0.5 * coeff[0] * T0
  T1 = (1.0 / eigval) * L @ X - T0
  Y = Y + coeff[1] * T1

  initial_state = (T0, T1, Y)
  (_, _, Y), _ = jax.lax.scan(body, initial_state, coeff[2:])
  return Y


def compute_chebychev_coeff_all(
    eigval: float, tau: float, K: int, dtype: np.dtype
) -> jnp.ndarray:
  """Jax wrapper to compute the K+1 Chebychev coefficients."""
  result_shape_dtype = jax.ShapeDtypeStruct(
      shape=(K + 1,),
      dtype=dtype,
  )

  chebychev_coeff = lambda eigval, tau, K: (
      2.0 * ive(np.arange(0, K + 1), -tau * eigval)
  ).astype(dtype)

  return jax.pure_callback(chebychev_coeff, result_shape_dtype, eigval, tau, K)
