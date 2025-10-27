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
import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ott.geometry import costs

try:
  import matplotlib as mpl
  import matplotlib.pyplot as plt
except ImportError:
  mpl = plt = None

__all__ = ["DualPotentials"]

PotentialFn = Callable[[jax.Array], jax.Array]


@jtu.register_static
@dataclasses.dataclass(frozen=True, repr=False)
class DualPotentials:
  r"""The Kantorovich dual potential functions :math:`f` and :math:`g`.

  :math:`f` and :math:`g` are a pair of functions, candidates for the dual
  OT Kantorovich problem, supposedly optimal for a given pair of measures.

  Args:
    f: The first dual potential function.
    g: The second dual potential function.
    cost_fn: The cost function used to solve the OT problem.
  """
  f: Optional[PotentialFn]
  g: Optional[PotentialFn]
  cost_fn: costs.CostFn

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    r"""Transport ``vec`` according to Gangbo-McCann Brenier :cite:`brenier:91`.

    Uses Proposition 1.15 from :cite:`santambrogio:15` to compute an OT map when
    applying the inverse gradient of cost.

    When the cost is a general cost, the operator uses the
    :meth:`~ott.geometry.costs.CostFn.twist_operator` associated of the
    corresponding :class:`~ott.geometry.costs.CostFn`.

    When the cost is a translation invariant :class:`~ott.geometry.costs.TICost`
    cost, :math:`c(x,y)=h(x-y)`, and the twist operator translates to the
    application of the convex conjugate of :math:`h` to the
    gradient of the dual potentials, namely
    :math:`x- (\nabla h^*)\circ \nabla f(x)` for the forward map,
    where :math:`h^*` is the Legendre transform of :math:`h`. For instance,
    in the case :math:`h(\cdot) = \|\cdot\|^2, \nabla h(\cdot) = 2 \cdot\,`,
    one has :math:`h^*(\cdot) = \|.\|^2 / 4`, and therefore
    :math:`\nabla h^*(\cdot) = 0.5 \cdot\,`.

    Args:
      vec: Points to transport, array of shape ``[n, d]``.
      forward: Whether to transport the points from source to the target
        distribution or vice-versa.

    Returns:
      The transported points.
    """
    vec = jnp.atleast_2d(vec)
    twist_op = jax.vmap(self.cost_fn.twist_operator, in_axes=[0, 0, None])
    if forward:
      return twist_op(vec, self._grad_f(vec), False)
    return twist_op(vec, self._grad_g(vec), True)

  def distance(self, src: jnp.ndarray, tgt: jnp.ndarray) -> float:
    r"""Evaluate Wasserstein distance between samples using dual potentials.

    This uses direct estimation of potentials against measures when dual
    functions are provided in usual form. This expression is valid for any
    cost function.

    Args:
      src: Samples from the source distribution, array of shape ``[n, d]``.
      tgt: Samples from the target distribution, array of shape ``[m, d]``.

    Returns:
      Wasserstein distance using specified cost function.
    """
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)
    f, g = jax.vmap(self.f), jax.vmap(self.g)
    return jnp.mean(f(src)) + jnp.mean(g(tgt))

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`f`."""
    assert self.f is not None, "The `f` potential is not computed."
    return jax.vmap(jax.grad(self.f, argnums=0))

  @property
  def _grad_g(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`g`."""
    assert self.g is not None, "The `g` potential is not computed."
    return jax.vmap(jax.grad(self.g, argnums=0))

  def plot_ot_map(
      self,
      source: jnp.ndarray,
      target: jnp.ndarray,
      samples: Optional[jnp.ndarray] = None,
      forward: bool = True,
      ax: Optional["plt.Axes"] = None,
      scatter_kwargs: Optional[Dict[str, Any]] = None,
      legend_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Tuple["plt.Figure", "plt.Axes"]:
    """Plot data and learned optimal transport map.

    Args:
      source: samples from the source measure
      target: samples from the target measure
      samples: extra samples to transport, either ``source`` (if ``forward``) or
        ``target`` (if not ``forward``) by default.
      forward: use the forward map from the potentials if ``True``,
        otherwise use the inverse map.
      ax: axis to add the plot to
      scatter_kwargs: additional kwargs passed into
        :meth:`~matplotlib.axes.Axes.scatter`
      legend_kwargs: additional kwargs passed into
        :meth:`~matplotlib.axes.Axes.legend`

    Returns:
      Figure and axes.
    """
    import matplotlib.pyplot as plt

    if scatter_kwargs is None:
      scatter_kwargs = {"alpha": 0.5}
    if legend_kwargs is None:
      legend_kwargs = {
          "ncol": 3,
          "loc": "upper center",
          "bbox_to_anchor": (0.5, -0.05),
          "edgecolor": "k"
      }

    if ax is None:
      fig = plt.figure(facecolor="white")
      ax = fig.add_subplot(111)
    else:
      fig = ax.get_figure()

    # plot the source and target samples
    if forward:
      label_transport = r"$\nabla f(source)$"
      source_color, target_color = "#1A254B", "#A7BED3"
    else:
      label_transport = r"$\nabla g(target)$"
      source_color, target_color = "#A7BED3", "#1A254B"

    ax.scatter(
        source[:, 0],
        source[:, 1],
        color=source_color,
        label="source",
        **scatter_kwargs,
    )
    ax.scatter(
        target[:, 0],
        target[:, 1],
        color=target_color,
        label="target",
        **scatter_kwargs,
    )

    # plot the transported samples
    samples = (source if forward else target) if samples is None else samples
    transported_samples = self.transport(samples, forward=forward)
    ax.scatter(
        transported_samples[:, 0],
        transported_samples[:, 1],
        color="#F2545B",
        label=label_transport,
        **scatter_kwargs,
    )

    for i in range(samples.shape[0]):
      ax.arrow(
          samples[i, 0],
          samples[i, 1],
          transported_samples[i, 0] - samples[i, 0],
          transported_samples[i, 1] - samples[i, 1],
          color=[0.5, 0.5, 1],
          alpha=0.3,
      )

    ax.legend(**legend_kwargs)
    return fig, ax

  def plot_potential(
      self,
      forward: bool = True,
      quantile: float = 0.05,
      kantorovich: bool = True,
      ax: Optional["mpl.axes.Axes"] = None,
      x_bounds: Tuple[float, float] = (-6, 6),
      y_bounds: Tuple[float, float] = (-6, 6),
      num_grid: int = 50,
      contourf_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Tuple["mpl.figure.Figure", "mpl.axes.Axes"]:
    r"""Plot the potential.

    Args:
      forward: use the forward map from the potentials
        if ``True``, otherwise use the inverse map
      quantile: quantile to filter the potentials with
      kantorovich: whether to plot the Kantorovich potential
      ax: axis to add the plot to
      x_bounds: x-axis bounds of the plot
        :math:`(x_{\text{min}}, x_{\text{max}})`
      y_bounds: y-axis bounds of the plot
        :math:`(y_{\text{min}}, y_{\text{max}})`
      num_grid: number of points to discretize the domain into a grid
        along each dimension
      contourf_kwargs: additional kwargs passed into
        :meth:`~matplotlib.axes.Axes.contourf`

    Returns:
      Figure and axes.
    """
    import matplotlib.pyplot as plt

    if contourf_kwargs is None:
      contourf_kwargs = {}

    ax_specified = ax is not None
    if not ax_specified:
      fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    else:
      fig = ax.get_figure()

    x1 = jnp.linspace(*x_bounds, num=num_grid)
    x2 = jnp.linspace(*y_bounds, num=num_grid)
    X1, X2 = jnp.meshgrid(x1, x2)
    X12flat = jnp.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    Zflat = jax.vmap(self.f if forward else self.g)(X12flat)
    if kantorovich:
      Zflat = 0.5 * (jnp.linalg.norm(X12flat, axis=-1) ** 2) - Zflat
    Zflat = np.asarray(Zflat)
    vmin, vmax = np.quantile(Zflat, [quantile, 1.0 - quantile])
    Zflat = Zflat.clip(vmin, vmax)
    Z = Zflat.reshape(X1.shape)

    CS = ax.contourf(X1, X2, Z, cmap="Blues", **contourf_kwargs)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    fig.colorbar(CS, ax=ax)
    if not ax_specified:
      fig.tight_layout()
    ax.set_title(r"$f$" if forward else r"$g$")
    return fig, ax
