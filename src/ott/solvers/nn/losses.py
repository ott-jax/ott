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

from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.solvers.linear import sinkhorn

__all__ = ["monge_gap", "monge_gap_from_samples"]


def monge_gap(
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    reference_points: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    relative_epsilon: Optional[bool] = None,
    scale_cost: Union[bool, int, float, Literal["mean", "max_cost",
                                                "median"]] = 1.0,
    return_output: bool = False,
    **kwargs: Any
) -> Union[float, Tuple[float, sinkhorn.SinkhornOutput]]:
  r"""Monge gap regularizer :cite:`uscidda:23`.

  For a cost function :math:`c` and empirical reference measure
  :math:`\hat{\rho}_n=\frac{1}{n}\sum_{i=1}^n \delta_{x_i}`, the
  (entropic) Monge gap of a map function
  :math:`T:\mathbb{R}^d\rightarrow\mathbb{R}^d` is defined as:

  .. math::
    \mathcal{M}^c_{\hat{\rho}_n, \varepsilon} (T)
    = \frac{1}{n} \sum_{i=1}^n c(x_i, T(x_i)) -
    W_{c, \varepsilon}(\hat{\rho}_n, T \sharp \hat{\rho}_n)

  See :cite:`uscidda:23` Eq. (8). This function is a thin wrapper that calls
  :func:`~ott.solvers.linear.nn.lossses.monge_gap_from_samples`.

  Args:
    map_fn: Callable corresponding to map :math:`T` in definition above. The
      callable should be vectorized (e.g. using :func:`jax.vmap`), i.e,
      able to process a *batch* of vectors of size `d`, namely
      ``map_fn`` applied to an array returns an array of the same shape.
    reference_points: Array of `[n,d]` points, :math:`\hat\rho_n` in paper
    cost_fn: An object of class :class:`~ott.geometry.costs.CostFn`.
    epsilon: Regularization parameter. See
      :class:`~ott.geometry.pointcloud.PointCloud`
    relative_epsilon: when `False`, the parameter ``epsilon`` specifies the
      value of the entropic regularization parameter. When `True`, ``epsilon``
      refers to a fraction of the
      :attr:`~ott.geometry.pointcloud.PointCloud.mean_cost_matrix`, which is
      computed adaptively using ``source`` and ``target`` points.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean' and 'max_cost'. Alternatively, a float factor can be
      given to rescale the cost such that ``cost_matrix /= scale_cost``.
      If `True`, use 'mean'.
    return_output: boolean to also return the
      :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
    kwargs: holds the kwargs to instantiate the or
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver to
      compute the regularized OT cost.

  Returns:
    The Monge gap value and optionally the
    :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
  """
  target = map_fn(reference_points)
  return monge_gap_from_samples(
      source=reference_points,
      target=target,
      cost_fn=cost_fn,
      epsilon=epsilon,
      relative_epsilon=relative_epsilon,
      scale_cost=scale_cost,
      **kwargs
  )


def monge_gap_from_samples(
    source: jnp.ndarray,
    target: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    relative_epsilon: Optional[bool] = None,
    scale_cost: Union[bool, int, float, Literal["mean", "max_cost",
                                                "median"]] = 1.0,
    return_output: bool = False,
    **kwargs: Any
) -> Union[float, Tuple[float, sinkhorn.SinkhornOutput]]:
  r"""Monge gap, instantiated in terms of samples before / after applying map.

  .. math::

    \frac{1}{n} \sum_{i=1}^n c(x_i, y_i)) -
    W_{c, \varepsilon}(\frac{1}{n}\sum_i \delta_{x_i},
    \frac{1}{n}\sum_i \delta_{y_i})

  where :math:`W_{c, \varepsilon}` is an entropy-regularized optimal transport
  cost, :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.ent_reg_cost`

  Args:
    source: samples from first measure, array of shape ``[n, d]``.
    target: samples from second measure, array of shape ``[n, d]``.
    cost_fn: a cost function between two points in dimension :math:`d`.
      If :obj:`None`, :class:`~ott.geometry.costs.SqEuclidean` is used.
    epsilon: Regularization parameter. See
      :class:`~ott.geometry.pointcloud.PointCloud`
    relative_epsilon: when `False`, the parameter ``epsilon`` specifies the
      value of the entropic regularization parameter. When `True`, ``epsilon``
      refers to a fraction of the
      :attr:`~ott.geometry.pointcloud.PointCloud.mean_cost_matrix`, which is
      computed adaptively using ``source`` and ``target`` points.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean' and 'max_cost'. Alternatively, a float factor can be
      given to rescale the cost such that ``cost_matrix /= scale_cost``.
      If `True`, use 'mean'.
    return_output: boolean to also return the
      :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
    kwargs: holds the kwargs to instantiate the or
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver to
      compute the regularized OT cost.

  Returns:
    The Monge gap value and optionally the
    :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
  """
  cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
  geom = pointcloud.PointCloud(
      x=source,
      y=target,
      cost_fn=cost_fn,
      epsilon=epsilon,
      relative_epsilon=relative_epsilon,
      scale_cost=scale_cost,
  )
  gt_displacement_cost = jnp.mean(jax.vmap(cost_fn)(source, target))
  out = linear.solve(geom=geom, **kwargs)
  loss = gt_displacement_cost - out.ent_reg_cost
  return (loss, out) if return_output else loss
