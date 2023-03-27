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
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from ott.solvers.linear import continuous_barycenter, sinkhorn, sinkhorn_lr

__all__ = ["WassersteinSolver"]

State = Union["sinkhorn.SinkhornState", "sinkhorn_lr.LRSinkhornState",
              "continuous_barycenter.FreeBarycenterState"]


# TODO(michalk8): refactor to have generic nested solver API
@jax.tree_util.register_pytree_node_class
class WassersteinSolver:
  """A generic solver for problems that use a linear problem in inner loop."""

  def __init__(
      self,
      epsilon: Optional[float] = None,
      rank: int = -1,
      linear_ot_solver: Optional[Union["sinkhorn.Sinkhorn",
                                       "sinkhorn_lr.LRSinkhorn"]] = None,
      min_iterations: int = 5,
      max_iterations: int = 50,
      threshold: float = 1e-3,
      store_inner_errors: bool = False,
      **kwargs: Any,
  ):
    from ott.solvers.linear import sinkhorn, sinkhorn_lr
    default_epsilon = 1.0
    # Set epsilon value to default if needed, but keep track of whether None was
    # passed to handle the case where a linear_ot_solver is passed or not.
    self.epsilon = epsilon if epsilon is not None else default_epsilon
    self.rank = rank
    self.linear_ot_solver = linear_ot_solver
    if self.linear_ot_solver is None:
      # Detect if user requests low-rank solver. In that case the
      # default_epsilon makes little sense, since it was designed for GW.
      if self.is_low_rank:
        if epsilon is None:
          # Use default entropic regularization in LRSinkhorn if None was passed
          self.linear_ot_solver = sinkhorn_lr.LRSinkhorn(
              rank=self.rank, **kwargs
          )
        else:
          # If epsilon is passed, use it to replace the default LRSinkhorn value
          self.linear_ot_solver = sinkhorn_lr.LRSinkhorn(
              rank=self.rank, epsilon=self.epsilon, **kwargs
          )
      else:
        # When using Entropic GW, epsilon is not handled inside Sinkhorn,
        # but rather added back to the Geometry object re-instantiated
        # when linearizing the problem. Therefore, no need to pass it to solver.
        self.linear_ot_solver = sinkhorn.Sinkhorn(**kwargs)

    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.threshold = threshold
    self.store_inner_errors = store_inner_errors
    self._kwargs = kwargs

  @property
  def is_low_rank(self) -> bool:
    """Whether the solver is low-rank."""
    return self.rank > 0

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return ([self.epsilon, self.linear_ot_solver, self.threshold], {
        "min_iterations": self.min_iterations,
        "max_iterations": self.max_iterations,
        "rank": self.rank,
        "store_inner_errors": self.store_inner_errors,
        **self._kwargs
    })

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "WassersteinSolver":
    epsilon, linear_ot_solver, threshold = children
    return cls(
        epsilon=epsilon,
        linear_ot_solver=linear_ot_solver,
        threshold=threshold,
        **aux_data
    )

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
