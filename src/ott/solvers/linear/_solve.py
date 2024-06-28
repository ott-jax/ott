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
from typing import Any, Optional, Union

import jax.numpy as jnp

from ott.geometry import geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr, univariate

__all__ = ["solve", "solve_univariate"]


def solve(
    geom: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    rank: int = -1,
    **kwargs: Any
) -> Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput]:
  """Solve linear regularized OT problem using Sinkhorn iterations.

  Args:
    geom: The ground geometry of the linear problem.
    a: The first marginal. If :obj:`None`, it will be uniform.
    b: The second marginal. If :obj:`None`, it will be uniform.
    tau_a: If :math:`< 1`, defines how much unbalanced the problem is
      on the first marginal.
    tau_b: If :math:`< 1`, defines how much unbalanced the problem is
      on the second marginal.
    rank:
      Rank constraint on the coupling to minimize the linear OT problem
      :cite:`scetbon:21`. If :math:`-1`, no rank constraint is used.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
      :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`,
      depending on the ``rank``.

  Returns:
    The Sinkhorn output.
  """
  prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau_a, tau_b=tau_b)
  if rank > 0:
    solver = sinkhorn_lr.LRSinkhorn(rank=rank, **kwargs)
  else:
    solver = sinkhorn.Sinkhorn(**kwargs)
  return solver(prob)


def solve_univariate(
    geom: pointcloud.PointCloud,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    *,
    return_transport: bool = False,
    return_dual_variables: bool = False,
) -> univariate.UnivariateOutput:
  """Solve 1D OT problems between two :math:`d`-dimensional point clouds.

  This function selects the underlying solver based on the following criteria:

  - :func:`~ott.solvers.linear.univariate.north_west_solver` - if
    ``return_dual_variables = True``.
  - :func:`~ott.solvers.linear.univariate.uniform_solver` - if ``a`` and
    ``b`` are both uniform and have the same size.
  - :func:`~ott.solvers.linear.univariate.quantile_solver` - otherwise.

  Args:
    geom: Geometry containing two :math:`d`-dimensional point clouds and
      a ground :class:`translation-invariant cost <ott.geometry.costs.TICost>`.
    a: The first marginal. If :obj:`None`, it will be uniform.
    b: The second marginal. If :obj:`None`, it will be uniform.
    return_transport: Whether to also return the mapped pairs used to compute
      the :attr:`~ott.solvers.linear.univariate.UnivariateOutput.transport_matrices`.
    return_dual_variables: Whether to also return the dual variables.

  Returns:
    The univariate output.
  """  # noqa: E501
  prob = linear_problem.LinearProblem(geom, a=a, b=b)
  if return_dual_variables:
    return univariate.north_west_solver(prob)
  if prob.is_uniform and prob.is_equal_size:
    return univariate.uniform_solver(prob, return_transport=return_transport)
  return univariate.quantile_solver(prob, return_transport=return_transport)
