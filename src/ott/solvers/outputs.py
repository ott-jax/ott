from typing import Optional, Tuple

import jax.numpy as jnp
from flax import struct

from ott.geometry import geometry
from ott.problems.linear import linear_problem


@struct.dataclass
class _BaseTransportOutput:
  """Base class for Transport output.

  Args:
    ot_prob: OT problem.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
    shape: Shape of the transport plan.
    costs: Holds the sequence of regularized GW costs seen through the loop
      of the solver.
    converged: Boolean value wether or not the problem has converged.
    reg_ot_cost: Value of the regularised OT loss.
  """

  # TODO(michalk8): must be called `errors`, because of `store_inner_errors`
  # in future, enforce via class hierarchy
  ot_prob: linear_problem.LinearProblem
  errors: jnp.ndarray
  shape: Tuple[int, int] = struct.field(pytree_node=False)
  # TODO(michalk8): Optional is an artifact of the current impl., refactor
  converged: bool = False
  costs: Optional[jnp.ndarray] = None
  reg_ot_cost: Optional[float] = None

  @property
  def linear(self) -> bool:  # noqa: D102
    return isinstance(self.ot_prob, linear_problem.LinearProblem)

  @property
  def linear_output(self) -> bool:  # noqa: D102
    return True

  @property
  def geom(self) -> geometry.Geometry:  # noqa: D102
    return self.ot_prob.geom

  @property
  def a(self) -> jnp.ndarray:  # noqa: D102
    return self.ot_prob.a

  @property
  def b(self) -> jnp.ndarray:  # noqa: D102
    return self.ot_prob.b

  @property
  def primal_cost(self) -> jnp.ndarray:
    """Return (by recomputing it) transport cost of current solution."""
    return self.transport_cost_at_geom(other_geom=self.geom)

  @property
  def transport_mass(self) -> float:
    """Sum of transport matrix."""
    return self.marginal(0).sum()

  def marginal(self, axis: int) -> jnp.ndarray:  # noqa: D102
    return self.apply(jnp.ones(self.shape[axis],), axis=axis)
