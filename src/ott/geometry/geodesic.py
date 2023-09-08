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
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.linalg import lobpcg_standard
from scipy.special import ive

from ott.geometry import geometry

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
      **kwargs: Any
  ):
    super().__init__(epsilon=1., **kwargs)
    self.laplacian = laplacian
    self.t = t
    self.order = order

  @classmethod
  def from_graph(
      cls,
      G: jnp.ndarray,
      t: Optional[float] = 1e-3,
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
      directed: Whether the ``graph`` is directed. If not, it's made
        undirected as :math:`G + G^T`. This parameter is ignored when passing
        the Laplacian directly, assumed to be symmetric.
      normalize: Whether to normalize the Laplacian as
        :math:`L^{sym} = \left(D^+\right)^{\frac{1}{2}} L
        \left(D^+\right)^{\frac{1}{2}}`, where :math:`L` is the
        non-normalized Laplacian and :math:`D` is the degree matrix.
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

    if t is None:
      t = (jnp.sum(G) / jnp.sum(G > 0.)) ** 2

    return cls(laplacian, t=t, **kwargs)

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

    def compute_laplacian(adjacency_matrix: jnp.ndarray) -> jnp.ndarray:
      """Compute the Laplacian matrix from the adjacency matrix.

      Args:
          adjacency_matrix: An (n, n) array representing the
          adjacency matrix of a graph.

      Returns:
          An (n, n) array representing the Laplacian matrix.
      """
      degree_matrix = jnp.diag(jnp.sum(adjacency_matrix, axis=0))
      return degree_matrix - adjacency_matrix

    def compute_largest_eigenvalue(laplacian_matrix, k):
      """Compute the largest eigenvalue of the Laplacian matrix.

      Args:
          laplacian_matrix: An (n, n) array representing the Laplacian matrix.
          k: Number of eigenvalues/vectors to compute.

      Returns:
          The largest eigenvalue of the Laplacian matrix.
      """
      n, _ = self.shape
      initial_directions = jax.random.normal(jax.random.PRNGKey(0), (n, k))
      # Convert the Laplacian matrix to a dense array
      #laplacian_array = laplacian_matrix.toarray()
      eigvals, _, _ = lobpcg_standard(laplacian_matrix, initial_directions, m=k)

      return np.max(eigvals)

    def rescale_laplacian(laplacian_matrix: jnp.ndarray) -> jnp.ndarray:
      """Rescale the Laplacian matrix.

      Args:
          laplacian_matrix: An (n, n) array representing the Laplacian matrix.

      Returns:
          The rescaled Laplacian matrix.
      """
      largest_eigenvalue = compute_largest_eigenvalue(laplacian_matrix, k=1)
      if largest_eigenvalue > 2:
        rescaled_laplacian = laplacian_matrix.copy()
        rescaled_laplacian /= largest_eigenvalue
      return 2 * rescaled_laplacian

    def define_scaled_laplacian(laplacian_matrix: jnp.ndarray) -> jnp.ndarray:
      """Define the scaled Laplacian matrix.

      Args:
          laplacian_matrix: An (n, n) array representing the Laplacian matrix.

      Returns:
          The scaled Laplacian matrix.
      """
      n = laplacian_matrix.shape[0]
      identity = jnp.eye(n)
      return laplacian_matrix - identity

    def chebyshev_coefficients(t: float, max_order: int) -> List[float]:
      """Compute the coeffs of the Chebyshev pols approx using Bessel functs.

      Args:
          t: Time parameter.
          max_order: Maximum order of the Chebyshev polynomial approximation.

      Returns:
          A list of coefficients.
      """
      return (2 * ive(jnp.arange(0, max_order + 1), -t)).tolist()

    def compute_chebyshev_approximation(
        x: jnp.ndarray, coeffs: List[float]
    ) -> jnp.ndarray:
      """Compute the Chebyshev polynomial approx for the given input and coeffs.

      Args:
          x: Input to evaluate the polynomial at.
          coeffs: List of Chebyshev polynomial coefficients.

      Returns:
          The Chebyshev polynomial approximation evaluated at x.
      """
      return self.apply_kernel(x, coeffs)

    #laplacian_matrix = compute_laplacian(self.adjacency_matrix)
    rescaled_laplacian = rescale_laplacian(self.laplacian)
    scaled_laplacian = define_scaled_laplacian(rescaled_laplacian)
    chebyshev_coeffs = chebyshev_coefficients(self.t, self.order)

    laplacian_times_signal = scaled_laplacian.dot(scaling)  # Apply the kernel

    return compute_chebyshev_approximation(
        laplacian_times_signal, chebyshev_coeffs
    )

  @property
  def kernel_matrix(self) -> jnp.ndarray:  # noqa: D102
    n, _ = self.shape
    return self.apply_kernel(jnp.eye(n))

  @property
  def _scale(self) -> float:
    """Constant used to scale the Laplacian."""
    if self.numerical_scheme == "backward_euler":
      return self.t / (4. * self.order)
    if self.numerical_scheme == "crank_nicolson":
      return self.t / (2. * self.order)
    raise NotImplementedError(
        f"Numerical scheme `{self.numerical_scheme}` is not implemented."
    )

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
    return [self.laplacian, self.t], {
        "order": self.order,
        "numerical_scheme": self.numerical_scheme,
        "tol": self.tol,
    }

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "Geodesic":
    return cls(*children, **aux_data)
