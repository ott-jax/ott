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
from typing import Any, Literal, Optional, Union

import jax.numpy as jnp

from ott.geometry import geometry
from ott.problems.quadratic import quadratic_costs, quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein as gw
from ott.solvers.quadratic import gromov_wasserstein_lr as lrgw

__all__ = ["solve"]


def solve(
    geom_xx: geometry.Geometry,
    geom_yy: geometry.Geometry,
    geom_xy: Optional[geometry.Geometry] = None,
    fused_penalty: float = 1.0,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    loss: Union[Literal["sqeucl", "kl"], quadratic_costs.GWLoss] = "sqeucl",
    gw_unbalanced_correction: bool = True,
    rank: int = -1,
    **kwargs: Any,
) -> Union[gw.GWOutput, lrgw.LRGWOutput]:
  """Solve quadratic regularized OT problem using a Gromov-Wasserstein solver.

  Args:
    geom_xx: Ground geometry of the first space.
    geom_yy: Ground geometry of the second space.
    geom_xy: Geometry defining the linear penalty term for
      fused Gromov-Wasserstein :cite:`vayer:19`. If :obj:`None`, the problem
      reduces to a plain Gromov-Wasserstein problem :cite:`peyre:16`.
    fused_penalty: Multiplier of the linear term in fused Gromov-Wasserstein,
      i.e. ``problem = purely quadratic + fused_penalty * linear problem``.
    a: The first marginal. If :obj:`None`, it will be uniform.
    b: The second marginal. If :obj:`None`, it will be uniform.
    tau_a: If :math:`< 1`, defines how much unbalanced the problem is
      on the first marginal.
    tau_b: If :math:`< 1`, defines how much unbalanced the problem is
      on the second marginal.
    loss: Gromov-Wasserstein loss function, see
      :class:`~ott.problems.quadratic.quadratic_costs.GWLoss` for more
      information. If ``rank > 0``, ``'sqeucl'`` is always used.
    gw_unbalanced_correction: Whether the unbalanced version of
      :cite:`sejourne:21` is used. Otherwise, ``tau_a`` and ``tau_b``
      only affect the resolution of the linearization of the GW problem
      in the inner loop. Only used when ``rank = -1``.
    rank: Rank constraint on the coupling to minimize the quadratic OT problem
      :cite:`scetbon:22`. If :math:`-1`, no rank constraint is used.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein` or
      :class:`~ott.solvers.quadratic.gromov_wasserstein_lr.LRGromovWasserstein`,
      depending on the ``rank``

  Returns:
    The Gromov-Wasserstein output.
  """
  prob = quadratic_problem.QuadraticProblem(
      geom_xx=geom_xx,
      geom_yy=geom_yy,
      geom_xy=geom_xy,
      fused_penalty=fused_penalty,
      a=a,
      b=b,
      tau_a=tau_a,
      tau_b=tau_b,
      loss=loss,
      gw_unbalanced_correction=gw_unbalanced_correction
  )

  if rank > 0:
    solver = lrgw.LRGromovWasserstein(rank=rank, **kwargs)
  else:
    solver = gw.GromovWasserstein(rank=rank, **kwargs)

  return solver(prob)
