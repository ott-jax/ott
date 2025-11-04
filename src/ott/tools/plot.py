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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.sparse as sp

import matplotlib.colors as mcolors
import matplotlib.patches as ptc
import matplotlib.pyplot as plt
from matplotlib import animation

from ott import math
from ott.experimental import mmsinkhorn
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein

# TODO(michalk8): make sure all outputs conform to a unified transport interface
Transport = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput,
                  gromov_wasserstein.GWOutput]

__all__ = ["Plot", "transport_animation"]


class Plot:
  """Plot an optimal transport map between two point clouds.

  This object can either plot or update a plot, to create animations as a
  :class:`~matplotlib.animation.FuncAnimation`, which can in turned be saved to
  disk at will. There are two design principles here:

  #. we do not rely on saving to/loading from disk to create animations
  #. we try as much as possible to disentangle the transport problem from
     its visualization.

  We use 2D scatter plots by default, relying on PCA visualization for d>3 data.
  This step requires a conversion to a numpy array, in order to compute leading
  singular values. This tool is therefore not designed having performance in
  mind.

  Args:
    fig: Specify figure object. Created by default
    ax: Specify axes objects. Created by default
    threshold: value below which links in transportation matrix won't be
      plotted. This value should be negative when using animations.
    scale: scale used for marker plots.
    show_lines: whether to show OT lines, as described in ``ot.matrix`` argument
    cmap: color map used to plot line colors.
    scale_alpha_by_coupling: use or not the coupling's value as proxy for alpha
    alpha: default alpha value for lines.
    title: title of the plot.
  """

  def __init__(
      self,
      fig: Optional["plt.Figure"] = None,
      ax: Optional["plt.Axes"] = None,
      threshold: float = -1.0,
      scale: int = 200,
      show_lines: bool = True,
      cmap: str = "cool",
      scale_alpha_by_coupling: bool = False,
      alpha: float = 0.7,
      title: Optional[str] = None,
      xlim: Optional[List[float]] = None,
      ylim: Optional[List[float]] = None,
  ):
    if ax is None and fig is None:
      fig, ax = plt.subplots()
    elif fig is None:
      fig = plt.gcf()
    elif ax is None:
      ax = plt.gca()
    self.fig = fig
    self.ax = ax
    self._show_lines = show_lines
    self._lines = []
    self._points_x = None
    self._points_y = None
    self._threshold = threshold
    self._scale = scale
    self._cmap = cmap
    self._scale_alpha_by_coupling = scale_alpha_by_coupling
    self._alpha = alpha
    self._title = title
    self._xlim = xlim
    self._ylim = ylim

  def _scatter(self, ot: Transport):
    """Compute the position and scales of the points on a 2D plot."""
    if not isinstance(ot.geom, pointcloud.PointCloud):
      raise ValueError("So far we only plot PointCloud geometry.")

    x, y = ot.geom.x, ot.geom.y
    a, b = ot.a, ot.b
    x, y = bidimensional(x, y)
    scales_x = a * self._scale * a.shape[0]
    scales_y = b * self._scale * b.shape[0]
    return x, y, scales_x, scales_y

  def _mapping(self, x: jax.Array, y: jax.Array, matrix: jax.Array):
    """Compute the lines representing the mapping between the 2 point clouds."""
    # Only plot the lines with a cost above the threshold.
    u, v = jnp.where(matrix > self._threshold)
    c = matrix[jnp.where(matrix > self._threshold)]
    xy = jnp.concatenate([x[u], y[v]], axis=-1)

    # Check if we want to adjust transparency.
    scale_alpha_by_coupling = self._scale_alpha_by_coupling

    # We can only adjust transparency if max(c) != min(c).
    if scale_alpha_by_coupling:
      min_matrix, max_matrix = jnp.min(c), jnp.max(c)
      scale_alpha_by_coupling = max_matrix != min_matrix

    result = []
    for i in range(xy.shape[0]):
      strength = jnp.max(jnp.array(matrix.shape)) * c[i]
      if scale_alpha_by_coupling:
        normalized_strength = (c[i] - min_matrix) / (max_matrix - min_matrix)
        alpha = self._alpha * float(normalized_strength)
      else:
        alpha = self._alpha

      # Matplotlib's transparency is sensitive to numerical errors.
      alpha = np.clip(alpha, 0.0, 1.0)

      start, end = xy[i, [0, 2]], xy[i, [1, 3]]
      result.append((start, end, strength, alpha))

    return result

  def __call__(self, ot: Transport) -> List[plt.Artist]:
    """Plot couplings in 2-D, using PCA if data is higher dimensional."""
    x, y, sx, sy = self._scatter(ot)
    self._points_x = self.ax.scatter(
        *x.T, s=sx, edgecolors="k", marker="o", label="x"
    )
    self._points_y = self.ax.scatter(
        *y.T, s=sy, edgecolors="k", marker="X", label="y"
    )
    self.ax.legend(fontsize=15)

    if self._title is not None:
      self.ax.set_title(self._title)

    if not self._show_lines:
      return []

    lines = self._mapping(x, y, ot.matrix)
    cmap = plt.get_cmap(self._cmap)
    self._lines = []
    for start, end, strength, alpha in lines:
      line, = self.ax.plot(
          start,
          end,
          linewidth=0.5 + 4 * strength,
          color=cmap(strength),
          zorder=0,
          alpha=alpha
      )
      self._lines.append(line)

    if self._xlim is not None:
      self.ax.set_xlim(self._xlim)
    if self._ylim is not None:
      self.ax.set_ylim(self._ylim)

    return [self._points_x, self._points_y] + self._lines

  def update(self,
             ot: Transport,
             title: Optional[str] = None) -> List[plt.Artist]:
    """Update a plot with a transport instance."""
    x, y, _, _ = self._scatter(ot)
    self._points_x.set_offsets(x)
    self._points_y.set_offsets(y)

    if title is not None:
      self.ax.set_title(title)

    if not self._show_lines:
      return []

    new_lines = self._mapping(x, y, ot.matrix)
    cmap = plt.get_cmap(self._cmap)
    for line, new_line in zip(self._lines, new_lines):
      start, end, strength, alpha = new_line

      line.set_data(start, end)
      line.set_linewidth(0.5 + 4 * strength)
      line.set_color(cmap(strength))
      line.set_alpha(alpha)

    # Maybe add new lines to the plot.
    num_lines = len(self._lines)
    num_to_plot = len(new_lines) if self._show_lines else 0
    for i in range(num_lines, num_to_plot):
      start, end, strength, alpha = new_lines[i]

      line, = self.ax.plot(
          start,
          end,
          linewidth=0.5 + 4 * strength,
          color=cmap(strength),
          zorder=0,
          alpha=alpha
      )
      self._lines.append(line)

    self._lines = self._lines[:num_to_plot]  # Maybe remove some
    return [self._points_x, self._points_y] + self._lines

  def animate(
      self,
      transports: Sequence[Transport],
      titles: Optional[Sequence[str]] = None,
      frame_rate: float = 10.0
  ) -> animation.FuncAnimation:
    """Make an animation from several transports."""
    _ = self(transports[0])
    if titles is None:
      titles = [None for _ in np.arange(0, len(transports))]
    assert len(titles) == len(transports), (
        f"titles/transports lengths differ `{len(titles)}`/`{len(transports)}`."
    )
    return animation.FuncAnimation(
        self.fig,
        lambda i: self.update(transports[i], titles[i]),
        np.arange(0, len(transports)),
        init_func=lambda: self.update(transports[0], titles[0]),
        interval=1000 / frame_rate,
        blit=True
    )


class PlotMM(Plot):
  """Plots an optimal transport map for :class:`~ott.experimental.mmsinkhorn.MMSinkhorn`.

  It enables to either plot or update a plot in a single object, offering the
  possibilities to create animations as a
  :class:`~matplotlib.animation.FuncAnimation`, which can in turned be saved to
  disk at will. There are two design principles here:

  #. we do not rely on saving to/loading from disk to create animations
  #. we try as much as possible to disentangle the transport problem from
       its visualization.

  Args:
    fig: Specify figure object. Created by default
    ax: Specify axes objects. Created by default
    fix_axes_lim: Whether to fix x/y limits to :math:`[0, 1]`.
    cmap: color map used to plot line colors.
    markers: Markers for each marginal.
    alpha: default alpha value for lines.
    title: title of the plot.
  """  # noqa: E501

  def __init__(
      self,
      fig: Optional[plt.Figure] = None,
      ax: Optional[plt.Axes] = None,
      fix_axes_lim: bool = False,
      cmap: Union[str, mcolors.Colormap] = "cividis_r",
      markers: str = "svopxdh",
      alpha: float = 0.6,
      title: Optional[str] = None,
  ):
    if isinstance(cmap, str):
      cmap = plt.colormaps[cmap]
    super().__init__(fig=fig, ax=ax, cmap=cmap, alpha=alpha, title=title)
    self._patches = None
    self._points = None
    self._markers = markers
    self._fix_axes_lim = fix_axes_lim

  def __call__(
      self,
      ot: mmsinkhorn.MMSinkhornOutput,
      top_k: Optional[int] = None
  ) -> List["plt.Artist"]:
    """Plot 2-D couplings. does not support higher dimensional."""
    assert ot.n_marginals <= len(self._markers), "Not enough markers to plot."
    self._points = []
    self._patches = []
    n0 = max(ot.shape)
    top_k = n0 if top_k is None else top_k

    # extract the `top_k` entries in the tensor, and their indices.
    _, idx = jax.lax.top_k(ot.tensor.ravel(), top_k)
    indices = jnp.unravel_index(idx, ot.shape)

    alphas = np.linspace(self._alpha, 0.2, max(0, top_k - n0))
    for j in range(top_k):
      points = [x[indices[i][j], ...] for i, x in enumerate(ot.x_s)]
      # re-order to ensure polygons have maximal area
      points = [points[i] for i in ccworder(jnp.array(points))]
      alpha = self._alpha if j < n0 else alphas[j - n0]

      polygon = ptc.Polygon(
          points,
          fill=True,
          linewidth=2,
          color=self._cmap(float(j >= n0)),
          alpha=alpha,
          zorder=-j,
      )
      self._patches.append(self.ax.add_patch(polygon))

    for i, (x, a) in enumerate(zip(ot.x_s, ot.a_s)):
      points = self.ax.scatter(
          x[:, 0],
          x[:, 1],
          s=200.0 * len(a) * a,
          marker=self._markers[i],
          c="black",
          linewidth=0.0,
          edgecolor=None,
          label=str(i)
      )
      self._points.append(points)

    if self._title is not None:
      self.ax.set_title(self._title)

    return self._points + self._patches

  def update(
      self,
      ot: mmsinkhorn.MMSinkhornOutput,
      title: Optional[str] = None,
      top_k: Optional[int] = None,
  ) -> List[plt.Artist]:
    """Update a plot with a transport instance."""
    n0 = max(ot.shape)
    top_k = n0 if top_k is None else top_k
    # extract the `top_k` entries in the tensor, and their indices.
    _, idx = jax.lax.top_k(ot.tensor.ravel(), top_k)
    indices = jnp.unravel_index(idx, ot.shape)

    alphas = np.linspace(self._alpha, 0.2, max(0, top_k - n0))
    for j, patch in enumerate(self._patches):
      points = [x[indices[i][j], ...] for i, x in enumerate(ot.x_s)]
      # re-order to ensure polygons have maximal area
      points = [points[i] for i in ccworder(jnp.array(points))]
      alpha = self._alpha if j < n0 else alphas[j - n0]
      # update the location of the patches according to the new coordinates
      patch.set_xy(points)
      patch.set_color(self._cmap(float(j >= n0)))
      patch.set_alpha(alpha)

    for points, xs in zip(self._points, ot.x_s):
      points.set_offsets(xs)

    if title is not None:
      self.ax.set_title(title)

    # we keep the axis fixed to 0-1 assuming normalized data
    if self._fix_axes_lim:
      eps = 2.5e-2
      self.ax.set_ylim(-eps, 1.0 + eps)
      self.ax.set_xlim(-eps, 1.0 + eps)

    return self._points + self._patches

  def animate(
      self,
      transports: Sequence[mmsinkhorn.MMSinkhornOutput],
      titles: Optional[Sequence[str]] = None,
      frame_rate: float = 10.0,
      top_k: Optional[int] = None,
  ) -> animation.FuncAnimation:
    """Make an animation from several transports."""
    ot, *_ = transports
    _ = self(ot, top_k=top_k)
    titles = titles if titles is not None else [""] * len(transports)
    return animation.FuncAnimation(
        self.fig,
        lambda i: self.update(ot=transports[i], title=titles[i], top_k=top_k),
        np.arange(0, len(transports)),
        init_func=lambda: self.update(ot, title=titles[0], top_k=top_k),
        interval=1000.0 / frame_rate,
        blit=True,
    )


def get_plotkwargs(
    background: bool,
    *,
    small_alpha: float = 0.2,
    large_alpha: float = 0.7,
    darkmode: bool = False,
    small_size: int = 50,
    mid_size: int = 60,
    size_multiplier: float = 1.2
) -> Dict[str, Any]:
  r"""Generate marker styling specifications for transport visualization.

  This utility function creates a dictionary of matplotlib styling parameters
  for various types of points and arrows used in optimal transport
  visualizations.

  Args:
    background: Whether source and target points should have small alphas to
      de-emphasize them and highlight other elements like dynamic points
      or arrows.
    small_alpha: Transparency value for background points.
    large_alpha: Transparency value for foreground/highlighted points.
    darkmode: Whether to use colors suitable for dark background plots.
      If :obj:`True``, use lighter colors, otherwise use standard colors.
    small_size: Base marker size for regular source/target points.
    mid_size: Marker size for highlighted new source points.
    size_multiplier: Multiplicative factor to enlarge transported points
      relative to their base size.

  Returns:
    A dictionary with the following keys, each containing marker styling
    parameters for matplotlib scatter/quiver plots:

    - ``'x'``: Regular source points :math:`\mu_0`
    - ``'tx'``: Transported source points :math:`\mu_t`
    - ``'xnew'``: New batch of highlighted source points
    - ``'txnew_interm'``: Intermediate positions of new transported points
    - ``'txnew'``: Final positions of new transported points
    - ``'y'``: Target points :math:`\mu_1`
    - ``'ifm'``: Independent flow matching (IFM) interpolated points
    - ``'arrows_grid'``: Velocity field arrows on grid points
    - ``'arrows_dynamic'``: Velocity field arrows for moving points
    - ``'arrows_ifm'``: Velocity field arrows for IFM points
  """
  sourcecolor = "lightcoral" if darkmode else "red"
  newsourcecolor = "salmon" if darkmode else "red"
  targetcolor = "deepskyblue" if darkmode else "blue"
  edgecolor = "white" if darkmode else "black"
  ifmcolor = "palegreen" if darkmode else "green"
  arrowscolor = "white" if darkmode else "black"
  arrows_ifm_color = "palegreen" if darkmode else "green"

  mid_alpha = (large_alpha + small_alpha) / 2
  # Regular points from source
  x = {
      "s": small_size,
      "label": r"$\mu_0$",
      "marker": "o",
      "color": sourcecolor,
      "edgecolor": edgecolor,
      "alpha": small_alpha if background else large_alpha,
  }

  # Points being transported
  tx = {
      "s": small_size * size_multiplier,
      "label": r"$\mu_t$",
      "marker": "o",
      "color": sourcecolor,
      "edgecolor": edgecolor,
      "alpha": large_alpha
  }

  # New batch of source points supposed to be highlighted
  xnew = {
      "s": mid_size,
      "marker": "h",
      "edgecolor": edgecolor,
      "color": newsourcecolor,
      "alpha": large_alpha,
  }

  txnew_interm = {
      "s": mid_size * size_multiplier,
      "marker": "h",
      "edgecolor": edgecolor,
      "color": newsourcecolor,
      "alpha": small_alpha,
  }

  txnew = {
      "s": mid_size * size_multiplier,
      "marker": "h",
      "edgecolor": edgecolor,
      "color": newsourcecolor,
      "alpha": large_alpha,
  }

  # Target Points
  y = {
      "s": small_size,
      "label": r"$\mu_1$",
      "marker": "s",
      "edgecolor": edgecolor,
      "color": targetcolor,
      "alpha": small_alpha if background else large_alpha,
  }

  # IFM Points
  ifm = {
      "s": small_size,
      "label": r"IFM $\mu_t$",
      "marker": "d",
      "edgecolor": edgecolor,
      "color": ifmcolor,
      "alpha": mid_alpha,
  }

  arrows_ifm = {"color": arrows_ifm_color, "alpha": large_alpha}

  arrows_grid = {
      "color": arrowscolor,
      "alpha": mid_alpha if background else large_alpha
  }

  arrows_dynamic = {"color": arrowscolor, "alpha": mid_alpha}

  return {
      "x": x,
      "tx": tx,
      "xnew": xnew,
      "txnew_interm": txnew_interm,
      "txnew": txnew,
      "y": y,
      "ifm": ifm,
      "arrows_grid": arrows_grid,
      "arrows_dynamic": arrows_dynamic,
      "arrows_ifm": arrows_ifm,
  }


def transport_animation(
    n_frames: int,
    brenier_potential: Callable[[jax.Array], jax.Array],
    static_source_points: jax.Array,
    static_target_points: Optional[jax.Array] = None,
    n_grid: int = 0,
    dynamic_points: Optional[jax.Array] = None,
    velocity_field: Optional[Callable[[jax.Array, jax.Array],
                                      jax.Array]] = None,
    plot_dynamic_transport: bool = False,
    plot_monge: bool = False,
    plot_ifm_interpolant: bool = False,
    plot_ifm_arrows: bool = False,
    max_points_ifm_interpolant: int = 256,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    xlimits: Optional[Tuple[float, float]] = None,
    ylimits: Optional[Tuple[float, float]] = None,
    padding: float = 0.1,
    interval: int = 300,
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:  # noqa: F821
  r"""Create animated visualizations of optimal transport and flow matching.

  This function generates animations illustrating various aspects of optimal
  transport, including Monge maps, McCann interpolation, Benamou-Brenier
  velocity fields, and flow matching approaches. It supports multiple
  visualization modes and can display static point clouds, dynamic trajectories,
  and velocity fields on grids.

  Args:
    n_frames: Number of animation frames. Must be at least ``1``. If ``1``,
      creates a static plot instead of an animation.
    brenier_potential: Convex function :math:`\\varphi` whose gradient defines
      the Brenier (optimal) map :math:`\\nabla\\varphi: \\mu_0 \\to \\mu_1`.
      Used to compute target points, velocity fields, and optimal transport
      visualizations.
    static_source_points: Source distribution points of shape ``[n, 2]``,
      representing samples from :math:`\\mu_0`. Always displayed in the plot.
    static_target_points: Target distribution points of shape ``[n, 2]``,
      representing samples from :math:`\\mu_1`. If :obj:`None`, computed as
      :math:`\\nabla\\varphi(\\text{static_source_points})`.
    n_grid: Number of grid points per dimension for displaying velocity fields.
      If ``> 0``, displays velocity field arrows on a uniform :math:`n_{grid}
      \\times n_{grid}` grid.
    dynamic_points: Additional points of shape ``[m, 2]`` to highlight and
      transport dynamically through the animation. Useful for emphasizing
      specific trajectories.
    velocity_field: Optional learned/estimated velocity field function with
      signature ``v(t, x) -> velocity`` where ``t`` is time (shape ``[batch]``)
      and ``x`` is position (shape ``[batch, 2]``). If :obj:`None`, uses
      Benamou-Brenier velocity from ``brenier_potential``.
    plot_dynamic_transport: Whether to show arrows and trajectories for
      ``dynamic_points`` as they are transported over time. Cannot be ``True``
      simultaneously with ``plot_monge``.
    plot_monge: Whether to display the Monge map as static arrows from source
      to target. Shows the complete optimal transport map at once. Cannot be
      ``True`` simultaneously with ``plot_dynamic_transport``.
    plot_ifm_interpolant: Whether to visualize the independent flow matching
      (IFM) interpolant :math:`(1-t)x_0 + tx_1` for random pairs of source and
      target points.
    plot_ifm_arrows: Whether to display velocity arrows for the IFM
      interpolant. Only relevant when ``plot_ifm_interpolant=True``.
    max_points_ifm_interpolant: Maximum number of point pairs to show when
      plotting IFM interpolant. Limits computational cost and visual clutter.
    title: Title for the plot/animation.
    figsize: Figure size as ``(width, height)`` in inches.
    xlimits: X-axis limits as ``(xmin, xmax)``. If :obj:`None`, computed
      automatically from all points with padding.
    ylimits: Y-axis limits as ``(ymin, ymax)``. If :obj:`None`, computed
      automatically from all points with padding.
    padding: Fractional padding to add around automatically computed axis
      limits. For example, ``0.1`` adds 10% padding on each side.
    interval: Used for animations, delay between frames in milliseconds.
    save_path: Path prefix for saving the animation/plot to disk.

  Returns:
    An animation object containing the animation (or static frame if
    ``n_frames=1``).
  """
  assert n_frames >= 1, f"n_frames must be nonnegative, got {n_frames}"
  assert not(plot_monge and plot_dynamic_transport), \
    "Cannot plot both Monge transport and dynamic transport"

  n_src = static_source_points.shape[0]

  plot_transport_arrows = plot_dynamic_transport or plot_monge

  # If we are plotting extra stuff on top of data,
  # data is displayed in low alpha as background
  background = plot_transport_arrows or plot_ifm_arrows
  dict_pk = get_plotkwargs(background=background)

  fig, ax = plt.subplots(figsize=figsize)
  fig.tight_layout(pad=2.0)

  # Time parameterization
  times = jnp.linspace(0.0, 1.0, n_frames)
  delta_times = jnp.diff(times)

  # Make sure we have target points, either froms source and potential, or
  # from data directly (assumed to be paired in that case)
  if static_target_points is not None:
    assert static_target_points.shape == static_source_points.shape
  elif brenier_potential is not None:
    static_target_points = jax.vmap(jax.grad(brenier_potential))(
        static_source_points
    )
  else:
    raise ValueError("Cannot resolve target set of poitnts.")

  if velocity_field is None and brenier_potential is not None:
    vel_brenier = math.velocity_from_brenier_potential(brenier_potential)
  else:
    vel_brenier = None

  ax.scatter(
      static_target_points[:, 0], static_target_points[:, 1], **dict_pk["y"]
  )

  ax.scatter(
      static_source_points[:, 0], static_source_points[:, 1], **dict_pk["x"]
  )
  # scale of arrows for first step (and maybe last-and-only step).
  dt = 1.0 if plot_monge or n_frames == 1 else delta_times[0]
  # Define space of points that will move (all by default if arrows
  # are requested and no dynamic_points are passed)
  if dynamic_points is None:
    dyn_points = static_source_points
  else:
    dyn_points = dynamic_points
  dyn_end_points = None

  if plot_transport_arrows:
    # Where do these arrows come from?
    if velocity_field is None:
      if dynamic_points is None:
        v_points = static_target_points - static_source_points
        dyn_end_points = static_target_points
      else:
        dyn_end_points = jax.vmap(jax.grad(brenier_potential))(dyn_points)
        v_points = dyn_end_points - dyn_points
    else:
      # If velocity field is passed, evaluate at time 0.
      v_points = velocity_field(
          jnp.zeros((dynamic_points.shape[0],)), dynamic_points
      )

    # Plot arrows
    quiver_points = ax.quiver(
        dyn_points[:, 0],
        dyn_points[:, 1],
        dt * v_points[:, 0],
        dt * v_points[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        **dict_pk["arrows_dynamic"],
    )

    # Add dynamic points
    scatter_interm_points_before = ax.scatter(
        dyn_points[:, 0], dyn_points[:, 1], **dict_pk["txnew_interm"]
    )

    # We might want to add another marker right after
    # the arrow, to illustrate the displacement, except when plotting monge
    if not plot_monge:
      scatter_interm_points_after = ax.scatter(
          dyn_points[:, 0] + dt * v_points[:, 0],
          dyn_points[:, 1] + dt * v_points[:, 1], **(dict_pk["txnew"])
      )

  if dynamic_points is not None:
    ax.scatter(dyn_points[:, 0], dyn_points[:, 1], **dict_pk["tx"])

  # Gather all points to set limits adaptively if needed.
  all_points = jnp.concatenate(
      (
          dyn_points,
          dyn_end_points if dyn_end_points is not None else dyn_points,
          static_source_points,
          static_target_points,
      ),
      axis=0,
  )

  if xlimits is None:
    xlimits = jnp.min(all_points[:, 0]), jnp.max(all_points[:, 0])
    xscale = xlimits[1] - xlimits[0]
    xlimits = (xlimits[0] - padding * xscale, xlimits[1] + padding * xscale)

  if ylimits is None:
    ylimits = jnp.min(all_points[:, 1]), jnp.max(all_points[:, 1])
    yscale = ylimits[1] - ylimits[0]
    ylimits = (ylimits[0] - padding * yscale, ylimits[1] + padding * yscale)

  # Display velocities on grids.
  if n_grid > 0:
    assert brenier_potential is not None or velocity_field is not None, \
      "To display field on grid points, provide Brenier potential or velocity."
    x = jnp.linspace(*xlimits, n_grid)
    y = jnp.linspace(*ylimits, n_grid)
    X, Y = jnp.meshgrid(x, y)
    points_grid = jnp.stack([X, Y], axis=-1).reshape(-1, 2)

    zero_time = times[0] * jnp.ones((points_grid.shape[0],))
    if velocity_field is None:
      v_grid = vel_brenier(zero_time, points_grid)
    else:
      v_grid = velocity_field(zero_time, points_grid)

    quiver_grid = ax.quiver(
        points_grid[:, 0],
        points_grid[:, 1],
        dt * v_grid[:, 0],
        dt * v_grid[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        **dict_pk["arrows_grid"],
    )

  if plot_ifm_interpolant:
    max_p = max_points_ifm_interpolant if plot_ifm_arrows else n_src ** 2
    product_points = jnp.stack((
        jnp.repeat(static_source_points, axis=0, repeats=n_src),
        jnp.tile(static_target_points, reps=(n_src, 1)),
    ))

    if max_p < n_src ** 2:
      product_points = product_points[:,
                                      jr.choice(
                                          jr.key(2),
                                          n_src * n_src, (max_p,),
                                          replace=False
                                      ), :]

    product_points_at_t = product_points[0, :, :]
    prod_scatter = ax.scatter(
        product_points_at_t[:, 0], product_points_at_t[:, 1], **dict_pk["ifm"]
    )
    if plot_ifm_arrows:
      prod_quiver = ax.quiver(
          product_points_at_t[:, 0],
          product_points_at_t[:, 1],
          dt * (product_points[1, :, 0] - product_points[0, :, 0]),
          dt * (product_points[1, :, 1] - product_points[0, :, 1]),
          angles="xy",
          scale_units="xy",
          scale=1,
          **dict_pk["arrows_ifm"]
      )

  # End of static frame

  ax.set_title(title)
  ax.legend()
  ax.grid(True)
  ax.set_aspect("equal")
  ax.set_xlim(*xlimits)
  ax.set_ylim(*ylimits)

  # Initialize dynamic points at t=0
  dyn_points_t = dyn_points

  def update_frame(frame) -> None:
    nonlocal dyn_points_t, v_points
    t = times[frame]

    # Update grid arrows (locations stay fixed)
    if n_grid > 0 and t < 1.0:
      dt = delta_times[frame - 1] if frame > 0 else delta_times[0]

      times_t = jnp.ones((points_grid.shape[0],)) * t

      if velocity_field is None:
        v_grid = vel_brenier(times_t, points_grid)
      else:
        v_grid = velocity_field(times_t, points_grid)

      quiver_grid.set_UVC(dt * v_grid[:, 0], dt * v_grid[:, 1])

    # Update moving point arrows (locations move with time)
    if plot_transport_arrows and not plot_monge:
      dt = delta_times[frame - 1] if frame > 0 else delta_times[0]
      if t >= 1.0:
        # Stop displaying arrows at t=1.0
        v_points = np.zeros_like(v_points)
        if velocity_field is None:
          dyn_points_t = dyn_end_points
      else:
        if velocity_field is not None:
          v_points = velocity_field(
              jnp.ones((dyn_points_t.shape[0],)) * t, dyn_points_t
          )
        else:
          # velocity field is constant, path can be reconstructed.
          dyn_points_t = (1 - t) * dyn_points + t * dyn_end_points

      quiver_points.set_offsets(dyn_points_t)
      quiver_points.set_UVC(dt * v_points[:, 0], dt * v_points[:, 1])
      scatter_interm_points_after.set_offsets(dyn_points_t + v_points * dt)
      scatter_interm_points_before.set_offsets(dyn_points_t)
      # Make move for next iteration if integrating along path.
      if velocity_field is not None:
        dyn_points_t = dyn_points_t + dt * v_points

    if (n_grid > 0 or plot_transport_arrows) and not plot_monge:
      ax.set_title(f"{title} at time {t:.2f}")

    if plot_ifm_interpolant:
      product_points_at_t = (1 - t) * product_points[
          0, :, :] + t * product_points[1, :, :]
      prod_scatter.set_offsets(product_points_at_t)

      if plot_ifm_arrows:
        prod_quiver.set_offsets(product_points_at_t)
      ax.set_title(f"{title} at time {t:.2f}")

  ani = animation.FuncAnimation(
      fig,
      update_frame,
      frames=n_frames,
      blit=False,
      interval=interval,
      repeat=True
  )

  if save_path is not None:
    if n_frames == 1:
      plt.savefig(save_path)
    else:
      ani.save(save_path, bitrate=2000)
  if n_frames >= 1:
    plt.close()
  return ani


@jax.jit
def ccworder(A: jax.Array) -> jax.Array:
  """Order points in counter-clockwise direction for polygon plotting.

  This helper function reorders a set of 2D points so that they can be used to
  draw a polygon with maximal area. It centers the points at the origin and
  then sorts them by their angular position.

  Args:
    A: Array of shape ``[n, 2]`` containing 2D point coordinates.

  Returns:
    Array of indices that reorder the input points in counter-clockwise order
    starting from the angle 0 (positive x-axis).

  Note:
    Based on: https://stackoverflow.com/questions/5040412/how-to-draw-the-largest-polygon-from-a-set-of-points
  """
  A = A - jnp.mean(A, 0, keepdims=True)
  return jnp.argsort(jnp.arctan2(A[:, 1], A[:, 0]))


def bidimensional(x: jax.Array, y: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Apply PCA to reduce to bi-dimensional data."""
  if x.shape[1] < 3:
    return x, y

  u, s, _ = sp.linalg.svds(np.array(jnp.concatenate([x, y], axis=0)), k=2)
  proj = u * s
  k = x.shape[0]
  return proj[:k], proj[k:]
