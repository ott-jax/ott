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

from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Union

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.solvers.linear import sinkhorn


def monge_gap(
    samples: jnp.ndarray,
    mapped_samples: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    relative_epsilon: Optional[bool] = None,
    scale_cost: Union[bool, int, float, Literal["mean", "max_cost",
                                                "median"]] = 1.0,
    sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any
) -> float:
  r"""Monge gap regularizer :cite:`uscidda:23`.

  For a cost function :math:`c` and an empirical reference :math:`\rho`
  defined by samples :math:`(x_i)_{i=1,...,n}`, the (entropic) Monge gap
  of a vector field :math:`T` is defined as:
  :math:`\mathcal{M}^c_{\rho_x, \epsilon}
  = \frac{1}{n} \sum_{i=1}^n c(x_i, T(x_i)) - W_\epsilon(\rho, T \# \rho)`.
  See :cite:`uscidda:23` Eq. (8).

  Args:
  samples: samples from the reference measure :math:`rho`.
  mapped_samples: samples from the reference measure :math:`rho`
  mapped with :math:`T`, i.e. samples from :math:T\sharp\rho.
  cost_fn: a CostFn function between two points in dimension d.
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
  sinkhorn_kwargs: holds the kwargs to instanciate the or
  :class:`ott.solvers.linear.sinkhorn.Sinkhorn` solver to
  compute the regularized OT cost.
  kwargs: additional kwargs to pass to the
  :class:`ott.geometry.pointcloud.PointCloud`, for instantiating
  the geometry between ``samples`` and ``mapped samples``.

  Returns:
  The Monge gap value.
  """
  n = len(samples)
  geom = pointcloud.PointCloud(
      x=samples,
      y=mapped_samples,
      cost_fn=cost_fn,
      epsilon=epsilon,
      relative_epsilon=relative_epsilon,
      scale_cost=scale_cost,
      **kwargs
  )
  gt_displacement_cost = (1 / n) * jnp.sum(
      jax.vmap(cost_fn)(samples, mapped_samples)
  )
  reg_opt_displacement_cost = sinkhorn.solve(
      geom=geom, **sinkhorn_kwargs
  ).reg_ot_cost

  # to ensures the Monge gap positivity,
  # we use as entropic regularizer the negative Shannon entropy
  # :math:`H(P)` instead of the kl-divergence between
  # the plan and the product of the marginals :math:`KL(P|a\otimes b)`.
  # since :math:`KL(P|a\otimes b) = - H(P) + H(a) + H(b)`, we just need to
  # remove the the entropy of the marginals here, which are both uniforms.
  reg_opt_displacement_cost -= 2 * geom.epsilon * jnp.log(n)

  return gt_displacement_cost - reg_opt_displacement_cost
