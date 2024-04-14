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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import numpy as np
from scipy.special import ive

from ott import utils
from ott.geometry import geometry
from ott.math import utils as mu

__all__ = ["Geodesic"]

Array_g = Union[jnp.ndarray, jesp.BCOO]


@jax.tree_util.register_pytree_node_class
class Geodesic(geometry.Geometry):
  r"""Graph distance approximation using heat kernel :cite:`huguet:2023`.

  .. note::
    This constructor is not meant to be called by the user,
    please use the :meth:`from_graph` method instead.

  Approximates the heat kernel using
  `Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_
  of the first kind of max order ``order``, which for small ``t``
  approximates the geodesic exponential kernel :math:`e^{\frac{-d(x, y)^2}{t}}`.

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
    super().__init__(epsilon=1.0, **kwargs)
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
      rng: Optional[jax.Array] = None,
      **kwargs: Any
  ) -> "Geodesic":
    r"""Construct a Geodesic geometry from an adjacency matrix.

    Args:
      G: Adjacency matrix.
      t: Time parameter for approximating the geodesic exponential kernel.
        If `None`, it defaults to :math:`\frac{1}{|E|} \sum_{(u, v) \in E}
        \text{weight}(u, v)` :cite:`crane:13`. In this case, the ``graph``
        must be specified and the edge weights are assumed to be positive.
      eigval: Largest eigenvalue of the Laplacian. If :obj:`None`, it's
        computed using :func:`jax.experimental.sparse.linalg.lobpcg_standard`.
      order: Max order of Chebyshev polynomials.
      directed: Whether the ``graph`` is directed. If :obj:`True`, it's made
        undirected as :math:`G + G^T`. This parameter is ignored when passing
        the Laplacian directly, assumed to be symmetric.
      normalize: Whether to normalize the Laplacian as
        :math:`L^{sym} = \left(D^+\right)^{\frac{1}{2}} L
        \left(D^+\right)^{\frac{1}{2}}`, where :math:`L` is the
        non-normalized Laplacian and :math:`D` is the degree matrix.
      rng: Random key used when computing the largest eigenvalue.
      kwargs: Keyword arguments for :class:`~ott.geometry.geodesic.Geodesic`.

    Returns:
      The Geodesic geometry.
    """
    assert G.shape[0] == G.shape[1], G.shape
    rng = utils.default_prng_key(rng)

    if directed:
      G = G + G.T
    if t is None:
      t = (jnp.sum(G) / jnp.sum(G > 0.0)) ** 2

    if isinstance(G, jesp.BCOO):
      laplacian = compute_sparse_laplacian(G, normalize)
    else:
      laplacian = compute_dense_laplacian(G, normalize)

    if eigval is None:
      eigval = compute_largest_eigenvalue(laplacian, rng)

    scaled_laplacian, eigval = jax.lax.cond((eigval > 2.0), lambda l:
                                            (2.0 * l / eigval, 2.0), lambda l:
                                            (l, eigval), laplacian)

    # compute the coeffs of the Chebyshev pols approx using Bessel funcs
    chebyshev_coeffs = compute_chebychev_coeff_all(
        0.5 * eigval, t, order, laplacian.dtype
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
        self.scaled_laplacian, scaling, self.chebyshev_coeffs, 0.5 * self.eigval
    )

  @property
  def kernel_matrix(self) -> jnp.ndarray:  # noqa: D102
    n, _ = self.shape
    kernel = self.apply_kernel(jnp.eye(n))
    return jax.lax.cond(
        jnp.allclose(kernel, kernel.T, atol=1e-8, rtol=1e-8), lambda x: x,
        lambda x: (x + x.T) / 2.0, kernel
    )

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
    """Not implemented."""
    raise ValueError("Not implemented.")

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


def normalize_laplacian(laplacian: Array_g, degree: jnp.ndarray) -> Array_g:
  inv_sqrt_deg = jnp.where(degree > 0.0, 1.0 / jnp.sqrt(degree), 0.0)
  return inv_sqrt_deg[:, None] * laplacian * inv_sqrt_deg[None, :]


def compute_dense_laplacian(
    G: jnp.ndarray, normalize: bool = False
) -> jnp.ndarray:
  degree = jnp.sum(G, axis=1)
  laplacian = jnp.diag(degree) - G
  if normalize:
    laplacian = normalize_laplacian(laplacian, degree)
  return laplacian


def compute_sparse_laplacian(
    G: jesp.BCOO, normalize: bool = False
) -> jesp.BCOO:
  n, _ = G.shape
  # making sure allocated indices has same dtype
  # on different devices int32 vs int64 can cause issues
  indices_dtype = G.indices.dtype
  data_degree, ixs = G.sum(1).todense(), jnp.arange(n, dtype=indices_dtype)
  degree = jesp.BCOO(
      (data_degree, jnp.c_[ixs, ixs]),
      shape=(n, n),
  )
  laplacian = degree - G
  if normalize:
    laplacian = normalize_laplacian(laplacian, data_degree)
  return laplacian


def compute_largest_eigenvalue(
    laplacian_matrix: jnp.ndarray,
    rng: jax.Array,
) -> float:
  # Compute the largest eigenvalue of the Laplacian matrix.
  n = laplacian_matrix.shape[0]
  # Generate random initial directions for eigenvalue computation
  initial_dirs = jax.random.normal(rng, (n, 1))

  # Create a sparse matrix-vector product function using sparsify
  # This function multiplies the sparse laplacian_matrix with a vector
  lapl_vector_product = jesp.sparsify(lambda v: laplacian_matrix @ v)

  # Compute eigenvalues using the sparse matrix-vector product
  eigvals, _, _ = jesp.linalg.lobpcg_standard(
      lapl_vector_product,
      initial_dirs,
  )
  return eigvals[0]


def expm_multiply(
    L: Array_g, X: jnp.ndarray, coeff: jnp.ndarray, eigval: float
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
