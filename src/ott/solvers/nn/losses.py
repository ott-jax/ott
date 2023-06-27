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

from typing import Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.solvers.linear import sinkhorn

__all__ = ["monge_gap"]


def monge_gap(
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
  r"""Monge gap regularizer :cite:`uscidda:23`.

  For a cost function :math:`c` and an empirical reference :math:`\rho`
  defined by samples :math:`(x_i)_{i=1,...,n}`, the (entropic) Monge gap
  of a vector field :math:`T` is defined as:
  :math:`\mathcal{M}^c_{\rho_x, \epsilon}
  = \frac{1}{n} \sum_{i=1}^n c(x_i, T(x_i)) - W_\epsilon(\rho, T \sharp \rho)`.
  See :cite:`uscidda:23` Eq. (8).

  Args:
    source: samples from the reference measure :math:`\rho`.
    target: samples from the mapped reference measure :math:`T \sharp \rho`.
      mapped with :math:`T`, i.e. samples from :math:`T \sharp \rho`.
    cost_fn: a cost function between two points in dimension :math:`d`.
    epsilon: Regularization parameter. If ``scale_epsilon = None`` and either
      ``relative_epsilon = True`` or ``relative_epsilon = None`` and
      ``epsilon = None`` in :class:`~ott.geometry.epsilon_scheduler.Epsilon`
      is used, ``scale_epsilon`` the is :attr:`mean_cost_matrix`. If
      ``epsilon = None``, use :math:`0.05`.
    relative_epsilon: when `False`, the parameter ``epsilon`` specifies the
      value of the entropic regularization parameter. When `True`, ``epsilon``
      refers to a fraction of the :attr:`mean_cost_matrix`, which is computed
      adaptively from data.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean' and 'max_cost'. Alternatively, a float factor can be
      given to rescale the cost such that ``cost_matrix /= scale_cost``.
      If `True`, use 'mean'.
    return_output: boolean to also return Sinkhorn output.
    kwargs: holds the kwargs to instantiate the or
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver to
      compute the regularized OT cost.

  Returns:
    The Monge gap value and optionally the Sinkhorn output.
  """
  geom = pointcloud.PointCloud(
      x=source,
      y=target,
      cost_fn=cost_fn,
      epsilon=epsilon,
      relative_epsilon=relative_epsilon,
      scale_cost=scale_cost,
  )
  gt_displacement_cost = jnp.mean(jax.vmap(cost_fn)(source, target))
  out = sinkhorn.solve(geom=geom, **kwargs)
  loss = gt_displacement_cost - out.ent_reg_cost
  return (loss, out) if return_output else loss
