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
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ott.geometry import geometry
from ott.math import fixed_point_loop
from ott.math import utils as mu

__all__ = ["Graph"]


@jax.tree_util.register_pytree_node_class
class Graph(geometry.Geometry):
  r"""Graph distance approximation using heat kernel :cite:`heitz:21,crane:13`.

  Approximates the heat kernel for large ``n_steps``, which for small ``t``
  approximates the geodesic exponential kernel :math:`e^{\frac{-d(x, y)^2}{t}}`.

  Args:
    laplacian: Symmetric graph Laplacian. The check for symmetry is **NOT**
      performed. See also :meth:`from_graph`.
    n_steps: Maximum number of steps used to approximate the heat kernel.
    numerical_scheme: Numerical scheme used to solve the heat diffusion.
    normalize: Whether to normalize the Laplacian as
      :math:`L^{sym} = \left(D^+\right)^{\frac{1}{2}} L
      \left(D^+\right)^{\frac{1}{2}}`, where :math:`L` is the
      non-normalized Laplacian and :math:`D` is the degree matrix.
    tol: Relative tolerance with respect to the Hilbert metric, see
      :cite:`peyre:19`, Remark 4.12. Used when iteratively updating scalings.
      If negative, this option is ignored and only ``n_steps`` is used.
    kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      laplacian: jnp.ndarray,
      t: float = 1e-3,
      n_steps: int = 100,
      numerical_scheme: Literal["backward_euler",
                                "crank_nicolson"] = "backward_euler",
      tol: float = -1.0,
      **kwargs: Any
  ):
    super().__init__(epsilon=1.0, **kwargs)
    self.laplacian = laplacian
    self.t = t
    self.n_steps = n_steps
    self.numerical_scheme = numerical_scheme
    self.tol = tol

  @classmethod
  def from_graph(
      cls,
      G: jnp.ndarray,
      t: Optional[float] = 1e-3,
      directed: bool = False,
      normalize: bool = False,
      **kwargs: Any
  ) -> "Graph":
    r"""Construct :class:`~ott.geometry.graph.Graph` from an adjacency matrix.

    Args:
      G: Adjacency matrix.
      t: Constant used when approximating the geodesic exponential kernel.
        If `None`, use :math:`\frac{1}{|E|} \sum_{(u, v) \in E} weight(u, v)`
        :cite:`crane:13`. In this case, the ``graph`` must be specified
        and the edge weights are all assumed to be positive.
      directed: Whether the ``graph`` is directed. If not, it will be made
        undirected as :math:`G + G^T`. This parameter is ignored when  directly
        passing the Laplacian, which is assumed to be symmetric.
      normalize: Whether to normalize the Laplacian as
        :math:`L^{sym} = \left(D^+\right)^{\frac{1}{2}} L
        \left(D^+\right)^{\frac{1}{2}}`, where :math:`L` is the
        non-normalized Laplacian and :math:`D` is the degree matrix.
      kwargs: Keyword arguments for :class:`~ott.geometry.graph.Graph`.

    Returns:
      The graph geometry.
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
      t = (jnp.sum(G) / jnp.sum(G > 0.0)) ** 2

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

    def conf_fn(
        iteration: int, consts: Tuple[jnp.ndarray, Optional[jnp.ndarray]],
        old_new: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> bool:
      del iteration, consts

      x_old, x_new = old_new
      x_old, x_new = mu.safe_log(x_old), mu.safe_log(x_new)
      # center
      x_old, x_new = x_old - jnp.nanmax(x_old), x_new - jnp.nanmax(x_new)
      # Hilbert metric, see Remark 4.12 in `Computational Optimal Transport`
      f = x_new - x_old
      return (jnp.nanmax(f) - jnp.nanmin(f)) > self.tol

    def body_fn(
        iteration: int, consts: Tuple[jnp.ndarray, Optional[jnp.ndarray]],
        old_new: Tuple[jnp.ndarray, jnp.ndarray], compute_errors: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
      del iteration, compute_errors

      L, scaled_lap = consts
      _, b = old_new

      if self.numerical_scheme == "crank_nicolson":
        # below is a preferred way of specifying the update (albeit more FLOPS),
        # as CSR/CSC/COO matrices don't support adding a diagonal matrix now:
        # b' = (2 * I - M) @ b = (2 * I - (I + c * L)) @ b = (I - c * L) @ b
        b = b - scaled_lap @ b
      return b, jsp.linalg.solve_triangular(L, b, lower=True)

    # eps we cannot use since it would require a re-solve
    # axis we can ignore since the matrix is symmetric
    del eps, axis

    force_scan = self.tol < 0.0
    fixpoint_fn = (
        fixed_point_loop.fixpoint_iter
        if force_scan else fixed_point_loop.fixpoint_iter_backprop
    )

    state = (jnp.full_like(scaling, jnp.nan), scaling)
    L = jsp.linalg.cholesky(self._M, lower=True)
    if self.numerical_scheme == "crank_nicolson":
      constants = L, self._scaled_laplacian
    else:
      constants = L, None

    return fixpoint_fn(
        cond_fn=(lambda *_, **__: True) if force_scan else conf_fn,
        body_fn=body_fn,
        min_iterations=self.n_steps if force_scan else 1,
        max_iterations=self.n_steps,
        inner_iterations=1,
        constants=constants,
        state=state,
    )[1]

  @property
  def kernel_matrix(self) -> jnp.ndarray:  # noqa: D102
    n, _ = self.shape
    kernel = self.apply_kernel(jnp.eye(n))
    # Symmetrize the kernel if needed. Numerical imprecision
    # happens when `numerical_scheme='backward_euler'` and small `t`
    return jax.lax.cond(
        jnp.allclose(kernel, kernel.T, atol=1e-8, rtol=1e-8), lambda x: x,
        lambda x: (x + x.T) / 2.0, kernel
    )

  @property
  def cost_matrix(self) -> jnp.ndarray:  # noqa: D102
    return -self.t * mu.safe_log(self.kernel_matrix)

  @property
  def _scale(self) -> float:
    """Constant used to scale the Laplacian."""
    if self.numerical_scheme == "backward_euler":
      return self.t / (4.0 * self.n_steps)
    if self.numerical_scheme == "crank_nicolson":
      return self.t / (2.0 * self.n_steps)
    raise NotImplementedError(
        f"Numerical scheme `{self.numerical_scheme}` is not implemented."
    )

  @property
  def _scaled_laplacian(self) -> jnp.ndarray:
    """Laplacian scaled by a constant, depending on the numerical scheme."""
    return self._scale * self.laplacian

  @property
  def _M(self) -> jnp.ndarray:
    n, _ = self.shape
    return self._scaled_laplacian + jnp.eye(n)

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
    return [self.laplacian, self.t], {
        "n_steps": self.n_steps,
        "numerical_scheme": self.numerical_scheme,
        "tol": self.tol,
    }

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "Graph":
    return cls(*children, **aux_data)
