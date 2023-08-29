from typing import Any, Optional, Union

import jax.numpy as jnp

from ott.geometry import geometry
from ott.problems.linear import linear_problem

#if TYPE_CHECKING:
from ott.solvers.linear import sinkhorn, sinkhorn_lr

__all__ = ["solve"]


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
    geom: The ground geometry cost of the linear problem.
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
      depending on ``rank``.

  Returns:
    The Sinkhorn output.
  """
  prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau_a, tau_b=tau_b)
  if rank > 0:
    solver = sinkhorn_lr.LRSinkhorn(rank=rank, **kwargs)
  else:
    solver = sinkhorn.Sinkhorn(**kwargs)
  return solver(prob)
