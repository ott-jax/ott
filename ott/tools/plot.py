# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plotting utils."""

from typing import List, Optional, Sequence, Union

import jax.numpy as jnp
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from ott.geometry import pointcloud
from ott.tools import transport
import scipy


def bidimensional(x: jnp.ndarray, y: jnp.ndarray):
  """Applies PCA to reduce to bimensional data."""
  if x.shape[1] < 3:
    return x, y

  u, s, _ = scipy.sparse.linalg.svds(jnp.concatenate([x, y], axis=0), k=2)
  proj = u * s
  k = x.shape[0]
  return proj[:k], proj[k:]


class Plot:
  """Plots an optimal transport map between two point clouds.

  It enables to either plot or update a plot in a single object, offering the
  possibilities to create animations as matplotlib.animation.FuncAnimation,
  which can in turned be saved to disk at will. There are two design principles
  here: 1) we do not rely on saving to/loading from disk to create animations
  2) we try as much as possible to disentangle the transport problem(s) from the
  its visualization(s), leveraging the transport.Transport interface.
  """

  def __init__(self,
               fig: Optional[plt.Figure] = None,
               ax: Optional[plt.Axes] = None,
               cost_threshold: float = 0.0,
               scale: int = 200,
               show_lines: bool = True,
               cmap: str = 'cool'):
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
    self._threshold = cost_threshold
    self._scale = scale
    self._cmap = cmap

  def _scatter(self, ot: transport.Transport):
    """Computes the position and scales of the points on a 2D plot."""
    if not isinstance(ot.geom, pointcloud.PointCloud):
      raise ValueError('So far we only plot PointCloud geometry.')

    x, y = ot.geom.x, ot.geom.y
    a, b = ot.a, ot.b
    x, y = bidimensional(x, y)
    scales_x = a * self._scale * a.shape[0]
    scales_y = b * self._scale * b.shape[0]
    return x, y, scales_x, scales_y

  def _mapping(self, x: jnp.ndarray, y: jnp.ndarray, matrix: jnp.ndarray):
    """Computes the lines representing the mapping between the 2 point clouds."""
    u, v = jnp.where(matrix > self._threshold)
    c = matrix[jnp.where(matrix > self._threshold)]
    xy = jnp.concatenate([x[u], y[v]], axis=-1)
    result = []
    for i in range(xy.shape[0]):
      strength = jnp.max(jnp.array(matrix.shape)) * c[i]
      result.append((xy[i, [0, 2]], xy[i, [1, 3]], strength))
    return result

  def __call__(self, ot: transport.Transport) -> List[plt.Artist]:
    """Plots 2-D couplings. Projects via PCA if data is higher dimensional."""
    x, y, sx, sy = self._scatter(ot)
    self._points_x = self.ax.scatter(
        *x.T, s=sx, edgecolors='k', marker='o', label='x')
    self._points_y = self.ax.scatter(
        *y.T, s=sy, edgecolors='k', marker='X', label='y')
    self.ax.legend(fontsize=15)
    if not self._show_lines:
      return []

    lines = self._mapping(x, y, ot.matrix)
    cmap = plt.get_cmap(self._cmap)
    self._lines = []
    for start, end, strength in lines:
      line, = self.ax.plot(start, end,
                           linewidth=0.5 + 4 * strength,
                           color=cmap(strength),
                           zorder=0, alpha=0.7)
      self._lines.append(line)
    return [self._points_x, self._points_y] + self._lines

  def update(self, ot: transport.Transport) -> List[plt.Artist]:
    """Updates a plot with a transport.Transport instance."""
    x, y, _, _ = self._scatter(ot)
    self._points_x.set_offsets(x)
    self._points_y.set_offsets(y)
    if not self._show_lines:
      return []

    new_lines = self._mapping(x, y, ot.matrix)
    cmap = plt.get_cmap(self._cmap)
    for line, new_line in zip(self._lines, new_lines):
      start, end, strength = new_line
      line.set_data(start, end)
      line.set_linewidth(0.5 + 4 * strength)
      line.set_color(cmap(strength))

    # Maybe add new lines to the plot.
    num_lines = len(self._lines)
    num_to_plot = len(new_lines) if self._show_lines else 0
    for i in range(num_lines, num_to_plot):
      start, end, strength = new_lines[i]
      line, = self.ax.plot(start, end,
                           linewidth=0.5 + 4 * strength,
                           color=cmap(strength),
                           zorder=0, alpha=0.7)
      self._lines.append(line)

    self._lines = self._lines[:num_to_plot]  # Maybe remove some
    return [self._points_x, self._points_y] + self._lines

  def animate(self,
              transports: Sequence[transport.Transport],
              frame_rate: float = 10.0) -> animation.FuncAnimation:
    """Makes an animation from several transport.Transport."""
    self(transports[0])
    return animation.FuncAnimation(
        self.fig,
        lambda i: self.update(transports[i]),
        np.arange(1, len(transports)),
        init_func=lambda: self.update(transports[0]),
        interval=1000 / frame_rate,
        blit=True)


def _barycenters(ax: plt.Axes,
                 y: jnp.ndarray,
                 a: jnp.ndarray,
                 b: jnp.ndarray,
                 matrix: jnp.ndarray,
                 scale: int = 200):
  """Plots 2-D sinkhorn barycenters."""
  sa, sb = jnp.min(a) / scale, jnp.min(b) / scale
  ax.scatter(*y.T, s=b / sb, edgecolors='k', marker='X', label='y')
  tx = 1 / a[:, None] * jnp.matmul(matrix, y)
  ax.scatter(*tx.T, s=a / sa, edgecolors='k', marker='X', label='T(x)')
  ax.legend(fontsize=15)


def barycentric_projections(arg: Union[transport.Transport, jnp.ndarray],
                            a: jnp.ndarray = None,
                            b: jnp.ndarray = None,
                            matrix: jnp.ndarray = None,
                            ax: Optional[plt.Axes] = None,
                            **kwargs):
  """Plots the barycenters, from the Transport object or from arguments."""
  if ax is None:
    _, ax = plt.subplots(1, 1, figsize=(8, 5))

  if isinstance(arg, transport.Transport):
    ot = arg
    return _barycenters(ax, ot.geom.y, ot.a, ot.b, ot.matrix, **kwargs)

  if matrix is None:
    raise ValueError('The `matrix` argument cannot be None.')

  a = jnp.ones(matrix.shape[0]) / matrix.shape[0] if a is None else a
  b = jnp.ones(matrix.shape[1]) / matrix.shape[1] if b is None else b
  return _barycenters(ax, arg, a, b, matrix, **kwargs)

