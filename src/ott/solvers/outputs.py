from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
from flax import struct

from ott.geometry import geometry
from ott.problems.linear import linear_problem


@struct.dataclass
class BaseTransportOutput:
    """Implement the problems.Transport interface, for a LR Sinkhorn solution."""

    # TODO(michalk8): must be called `errors`, because of `store_inner_errors`
    # in future, enforce via class hierarchy
    errors: jnp.ndarray
    shape: (int, int) = struct.field(pytree_node=False)
    ot_prob: linear_problem.LinearProblem
    # TODO(michalk8): Optional is an artifact of the current impl., refactor
    costs: Optional[jnp.ndarray] = None
    reg_ot_cost: Optional[float] = None

    @property
    def linear(self) -> bool:
        return isinstance(self.ot_prob, linear_problem.LinearProblem)

    @property
    def linear_output(self) -> bool:
        return True

    @property
    def geom(self) -> geometry.Geometry:
        return self.ot_prob.geom

    @property
    def a(self) -> jnp.ndarray:
        return self.ot_prob.a

    @property
    def b(self) -> jnp.ndarray:
        return self.ot_prob.b

    @property
    def primal_cost(self) -> jnp.ndarray:
        """Return (by recomputing it) transport cost of current solution."""
        return self.transport_cost_at_geom(other_geom=self.geom)

    @property
    def transport_mass(self) -> float:
        """Sum of transport matrix."""
        return self.marginal(0).sum()

    def marginal(self, axis: int) -> jnp.ndarray:
        return self.apply(jnp.ones(self.shape[axis],), axis=axis)
