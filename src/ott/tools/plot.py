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
from typing import List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import scipy

from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein

try:
  import matplotlib.pyplot as plt
  from matplotlib import animation
except ImportError:
  plt = animation = None

# TODO(michalk8): make sure all outputs conform to a unified transport interface
Transport = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput,
                  gromov_wasserstein.GWOutput]


def bidimensional(x: jnp.ndarray,
                  y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Apply PCA to reduce to bi-dimensional data."""
  if x.shape[1] < 3:
    return x, y

  u, s, _ = scipy.sparse.linalg.svds(
      np.array(jnp.concatenate([x, y], axis=0)), k=2
  )
  proj = u * s
  k = x.shape[0]
  return proj[:k], proj[k:]


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
      title: Optional[str] = None
  ):
    if plt is None:
      raise RuntimeError("Please install `matplotlib` first.")

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

  def _mapping(self, x: jnp.ndarray, y: jnp.ndarray, matrix: jnp.ndarray):
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

  def __call__(self, ot: Transport) -> List["plt.Artist"]:
    """Plot couplings in 2-D, using PCA if data is higher dimensional."""
    x, y, sx, sy = self._scatter(ot)
    self._points_x = self.ax.scatter(
        *x.T, s=sx, edgecolors="k", marker="o", label="x"
    )
    self._points_y = self.ax.scatter(
        *y.T, s=sy, edgecolors="k", marker="X", label="y"
    )
    self.ax.legend(fontsize=15)
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
    if self._title is not None:
      self.ax.set_title(self._title)
    return [self._points_x, self._points_y] + self._lines

  def update(self,
             ot: Transport,
             title: Optional[str] = None) -> List["plt.Artist"]:
    """Update a plot with a transport instance."""
    x, y, _, _ = self._scatter(ot)
    self._points_x.set_offsets(x)
    self._points_y.set_offsets(y)
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
    if title is not None:
      self.ax.set_title(title)
    return [self._points_x, self._points_y] + self._lines

  def animate(
      self,
      transports: Sequence[Transport],
      titles: Optional[Sequence[str]] = None,
      frame_rate: float = 10.0
  ) -> "animation.FuncAnimation":
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
