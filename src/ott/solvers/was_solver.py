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
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ott.solvers.linear import sinkhorn, sinkhorn_lr

if TYPE_CHECKING:
  from ott.solvers.linear import continuous_barycenter

__all__ = ["WassersteinSolver"]

State = Union[sinkhorn.SinkhornState, sinkhorn_lr.LRSinkhornState,
              "continuous_barycenter.FreeBarycenterState"]


@jax.tree_util.register_pytree_node_class
class WassersteinSolver:
  """A generic solver for problems that use a linear problem in inner loop."""

  def __init__(
      self,
      linear_solver: Union["sinkhorn.Sinkhorn", "sinkhorn_lr.LRSinkhorn"],
      threshold: float = 1e-3,
      min_iterations: int = 5,
      max_iterations: int = 50,
      store_inner_errors: bool = False,
  ):
    self.linear_solver = linear_solver
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.threshold = threshold
    self.store_inner_errors = store_inner_errors

  @property
  def rank(self) -> int:
    """Rank of the linear OT solver."""
    return self.linear_solver.rank if self.is_low_rank else -1

  @property
  def is_low_rank(self) -> bool:
    """Whether the solver is low-rank."""
    return isinstance(self.linear_solver, sinkhorn_lr.LRSinkhorn)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return ([self.linear_solver, self.threshold], {
        "min_iterations": self.min_iterations,
        "max_iterations": self.max_iterations,
        "store_inner_errors": self.store_inner_errors,
    })

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "WassersteinSolver":
    return cls(*children, **aux_data)

  def _converged(self, state: State, iteration: int) -> bool:
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_and(
        i >= 2, jnp.isclose(costs[i - 2], costs[i - 1], rtol=tol)
    )

  def _diverged(self, state: State, iteration: int) -> bool:
    return jnp.logical_not(jnp.isfinite(state.costs[iteration - 1]))

  def _continue(self, state: State, iteration: int) -> bool:
    """Continue while not(converged) and not(diverged)."""
    return jnp.logical_or(
        iteration <= 2,
        jnp.logical_and(
            jnp.logical_not(self._diverged(state, iteration)),
            jnp.logical_not(self._converged(state, iteration))
        )
    )
