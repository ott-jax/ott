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
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import numpy as np

from ott.geometry import costs
from ott.problems.linear import linear_problem

try:
  import matplotlib as mpl
  import matplotlib.pyplot as plt
except ImportError:
  mpl = plt = None

__all__ = ["DualPotentials", "EntropicPotentials"]
Potential_t = Callable[[jnp.ndarray], float]


@jtu.register_pytree_node_class
class DualPotentials:
  r"""The Kantorovich dual potential functions :math:`f` and :math:`g`.

  :math:`f` and :math:`g` are a pair of functions, candidates for the dual
  OT Kantorovich problem, supposedly optimal for a given pair of measures.

  Args:
    f: The first dual potential function.
    g: The second dual potential function.
    cost_fn: The cost function used to solve the OT problem.
    corr: Whether the duals solve the problem in distance form, or correlation
      form (as used for instance for ICNNs, see, e.g., top right of p.3 in
      :cite:`makkuva:20`)
  """

  def __init__(
      self,
      f: Potential_t,
      g: Potential_t,
      *,
      cost_fn: costs.CostFn,
      corr: bool = False
  ):
    self._f = f
    self._g = g
    assert (
        not corr or type(cost_fn) == costs.SqEuclidean
    ), "Duals in `corr` form can only be used with a squared-Euclidean cost."
    self.cost_fn = cost_fn
    self._corr = corr

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

    Note:
      When the dual potentials are solved in correlation form, and marked
      accordingly by setting ``corr`` to ``True``, the maps are
      :math:`\nabla g` for forward, :math:`\nabla f` for backward map. This can
      only make sense when using the squared-Euclidean
      :class:`~ott.geometry.costs.SqEuclidean` cost.

    Args:
      vec: Points to transport, array of shape ``[n, d]``.
      forward: Whether to transport the points from source to the target
        distribution or vice-versa.

    Returns:
      The transported points.
    """
    from ott.geometry import costs

    vec = jnp.atleast_2d(vec)

    if self._corr and isinstance(self.cost_fn, costs.SqEuclidean):
      return self._grad_f(vec) if forward else self._grad_g(vec)
    twist_op = jax.vmap(self.cost_fn.twist_operator, in_axes=[0, 0, None])
    if forward:
      return twist_op(vec, self._grad_f(vec), False)
    return twist_op(vec, self._grad_g(vec), True)

  def distance(self, src: jnp.ndarray, tgt: jnp.ndarray) -> float:
    r"""Evaluate Wasserstein distance between samples using dual potentials.

    This uses direct estimation of potentials against measures when dual
    functions are provided in usual form. This expression is valid for any
    cost function.

    When potentials are given in correlation form, as specified by the flag
    ``corr``, the dual potentials solve the dual problem corresponding to the
    minimization of the primal OT problem where the ground cost is
    :math:`-2\langle x,y\rangle`. To recover the (squared) 2-Wasserstein
    distance, terms are re-arranged and contributions from squared norms are
    taken into account.

    Args:
      src: Samples from the source distribution, array of shape ``[n, d]``.
      tgt: Samples from the target distribution, array of shape ``[m, d]``.

    Returns:
      Wasserstein distance using specified cost function.
    """
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)
    f = jax.vmap(self.f)
    g = jax.vmap(self.g)
    out = jnp.mean(f(src)) + jnp.mean(g(tgt))
    if self._corr:
      out = -2.0 * out + jnp.mean(jnp.sum(src ** 2, axis=-1))
      out += jnp.mean(jnp.sum(tgt ** 2, axis=-1))
    return out

  @property
  def f(self) -> Potential_t:
    """The first dual potential function."""
    return self._f

  @property
  def g(self) -> Potential_t:
    """The second dual potential function."""
    return self._g

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`f`."""
    return jax.vmap(jax.grad(self.f, argnums=0))

  @property
  def _grad_g(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`g`."""
    return jax.vmap(jax.grad(self.g, argnums=0))

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [], {
        "f": self._f,
        "g": self._g,
        "cost_fn": self.cost_fn,
        "corr": self._corr
    }

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "DualPotentials":
    return cls(*children, **aux_data)

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
    if mpl is None:
      raise RuntimeError("Please install `matplotlib` first.")

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


@jtu.register_pytree_node_class
class EntropicPotentials(DualPotentials):
  """Dual potential functions from finite samples :cite:`pooladian:21`.

  Args:
    f_xy: The first dual potential vector of shape ``[n,]``.
    g_xy: The second dual potential vector of shape ``[m,]``.
    prob: Linear problem with :class:`~ott.geometry.pointcloud.PointCloud`
      geometry that was used to compute the dual potentials using, e.g.,
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.
    f_xx: The first dual potential vector of shape ``[n,]`` used for debiasing
      :cite:`pooladian:22`.
    g_yy: The second dual potential vector of shape ``[m,]`` used for debiasing.
  """

  def __init__(
      self,
      f_xy: jnp.ndarray,
      g_xy: jnp.ndarray,
      prob: linear_problem.LinearProblem,
      f_xx: Optional[jnp.ndarray] = None,
      g_yy: Optional[jnp.ndarray] = None,
  ):
    # we pass directly the arrays and override the properties
    # since only the properties need to be callable
    super().__init__(f_xy, g_xy, cost_fn=prob.geom.cost_fn, corr=False)
    self._prob = prob
    self._f_xx = f_xx
    self._g_yy = g_yy

  @property
  def f(self) -> Potential_t:  # noqa: D102
    return self._potential_fn(kind="f")

  @property
  def g(self) -> Potential_t:  # noqa: D102
    return self._potential_fn(kind="g")

  def _potential_fn(self, *, kind: Literal["f", "g"]) -> Potential_t:
    from ott.geometry import pointcloud

    def callback(
        x: jnp.ndarray,
        *,
        potential: jnp.ndarray,
        y: jnp.ndarray,
        weights: jnp.ndarray,
        epsilon: float,
    ) -> float:
      x = jnp.atleast_2d(x)
      assert x.shape[-1] == y.shape[-1], (x.shape, y.shape)
      geom = pointcloud.PointCloud(x, y, cost_fn=self.cost_fn)
      cost = geom.cost_matrix
      z = (potential - cost) / epsilon
      lse = -epsilon * jsp.special.logsumexp(z, b=weights, axis=-1)
      return jnp.squeeze(lse)

    assert isinstance(
        self._prob.geom, pointcloud.PointCloud
    ), f"Expected point cloud geometry, found `{type(self._prob.geom)}`."
    x, y = self._prob.geom.x, self._prob.geom.y
    a, b = self._prob.a, self._prob.b

    # `f_xx` or `g_yy` can both be `None`, we check for this later
    debiased_potentials = EntropicPotentials(self._f_xx, self._g_yy, self._prob)
    if kind == "f":
      # When seeking to evaluate 1st potential function,
      # the 2nd set of potential values and support should be used,
      # see proof of Prop. 2 in https://arxiv.org/pdf/2109.12004.pdf
      potential, arr, weights = self._g, y, b
      potential_other = None if self._f_xx is None else debiased_potentials.g
    else:
      potential, arr, weights = self._f, x, a
      potential_other = None if self._g_yy is None else debiased_potentials.f

    potential_xy = jax.tree_util.Partial(
        callback,
        potential=potential,
        y=arr,
        weights=weights,
        epsilon=self.epsilon,
    )

    if potential_other is None:
      return potential_xy
    return lambda x: (potential_xy(x) - potential_other(x))

  @property
  def is_debiased(self) -> bool:
    """Whether the :attr:`f` or :attr:`g` is debiased.

    The :attr:`g` potential is **not** debiased when ``static_b = True`` is
    passed in :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.
    """
    return self._f_xx is not None or self._g_yy is not None

  @property
  def epsilon(self) -> float:
    """Entropy regularizer."""
    return self._prob.geom.epsilon

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [self._f, self._g, self._prob, self._f_xx, self._g_yy], {}
