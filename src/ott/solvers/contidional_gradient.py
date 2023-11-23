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
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import host_callback
from jaxopt import BacktrackingLineSearch

from ott.geometry.geometry import Geometry
from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn

__all__ = ["ConditionalGradient", "ConditionalGradientState"]

LinearOutput = sinkhorn.SinkhornOutput

ProgressCallbackFn_t = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "ConditionalGradient"]], None]


class ConditionalGradientState(NamedTuple):
  """State of the Generalized Conditional Gradient (ConditionalGradient) Solver.

  Attributes:
    costs: Holds the sequence of costs seen through the outer
    loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
    inner linear solvers.
    linear_pb: last inner linear_problem
    linear_state: solution to the linear_pb
    sol_matrix: current solution matrix for ConditionalGradient
    errors: sequence of vectors of errors of the Sinkhorn algorithm
    at each iteration.
  """

  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_pb: linear_problem.LinearProblem
  linear_state: LinearOutput
  sol_matrix: jnp.ndarray
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "ConditionalGradientState":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(  # noqa: D102
      self,
      iteration: int,
      cost: float,
      sol_matrix: jnp.ndarray,
      linear_pb: linear_problem.LinearProblem,
      linear_sol: LinearOutput,
      store_errors: bool,
  ) -> "ConditionalGradientState":
    costs = self.costs.at[iteration].set(cost)
    errors = None
    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :].set(linear_sol.errors)
    linear_convergence = self.linear_convergence.at[iteration].set(
        linear_sol.converged
    )
    return self.set(
        linear_pb=linear_pb,
        linear_state=linear_sol,
        sol_matrix=sol_matrix,
        costs=costs,
        linear_convergence=linear_convergence,
        errors=errors,
    )


@jax.tree_util.register_pytree_node_class
class ConditionalGradient(was_solver.WassersteinSolver):
  """Implements generatlized conditional gradient solver for regularized OT.

  Args:
    args: positional arguments for
    :class:`~ott.solvers.was_solver.WassersteinSolver`.
    kwargs: key arguments for
    :class:`~ott.solvers.was_solver.WassersteinSolver`.
  """

  def __init__(
      self,
      *args: Any,
      **kwargs: Any,
  ):
    super().__init__(*args, **kwargs)

  def tree_flatten(self,) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    children, aux_data = super().tree_flatten()
    aux_data["progress_fn"] = self.progress_fn
    aux_data["kwargs_init"] = self.kwargs_init
    return children, aux_data

  def init_state(
      self,
      init_linear_pb: linear_problem.LinearProblem,
  ) -> ConditionalGradientState:
    """Initialize the state of the ConditionalGradient.

    Args:
      init_linear_pb: initialization for linear OT problem

    Returns:
      initial ConditionalGradientState .
    """
    linear_state = self.linear_ot_solver(init_linear_pb)
    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones((num_iter, self.linear_ot_solver.outer_iterations))
    else:
      errors = None

    return ConditionalGradientState(
        costs=-jnp.ones((num_iter,)),
        linear_convergence=-jnp.ones((num_iter,)),
        linear_pb=init_linear_pb,
        linear_state=linear_state,
        sol_matrix=linear_state.matrix,
        errors=errors,
    )

  def __call__(
      self,
      cost_mat,
      loss,
      epsilon,
      reg,
      linesearch_maxiter: int = 40,
      init_linear_pb: Optional[linear_problem.LinearProblem] = None,
      init_state: Optional[ConditionalGradientState] = None,
      progress_fn: Optional[ProgressCallbackFn_t] = None,
  ) -> ConditionalGradientState:
    """Run ConditionalGradient.

    Args:
      cost_mat : cost matrix related to the transport problem.
      init_linear_pb: the initial linear problem to be solved
      loss: regulatization function
      epsilon: entropic regularization constant
      reg: weight for non-entropic regularization
      linesearch_maxiter: maximum iterations for stepsize linesearch
      init_linear_pb: first linear to be solve
      init_state: allows the user to directly define the initial state.
      If provided init_prob and cost_mat are ignored
      progress_fn: callback function which gets called during the
      inner iterations, so the user can display the error at each
      iteration, e.g., using a progress bar.
      See :func:`~ott.utils.default_progress_fn` for a basic implementation.

    Returns:
      Last solver state.
    """
    grad_loss = jax.grad(loss)

    def cost_fun(x):
      return (
          jnp.sum(x * cost_mat) - epsilon * (jnp.sum(x * jnp.log(x)) + 1) +
          reg * loss(x)
      )

    line_search = BacktrackingLineSearch(
        fun=cost_fun, maxiter=linesearch_maxiter
    )

    def next_linearization(
        state: ConditionalGradientState
    ) -> linear_problem.LinearProblem:
      current_sol = state.sol_matrix
      new_cost_matrix = cost_mat + reg * grad_loss(current_sol)
      geom = Geometry(new_cost_matrix, epsilon)
      return linear_problem.LinearProblem(geom)

    def update_state_fn(
        state: ConditionalGradientState,
        new_linear_sol: "LinearOutput",
    ) -> Tuple[jnp.array, float]:
      # Constructing the cost function
      current_sol = state.sol_matrix
      new_sol = new_linear_sol.matrix
      delta = new_sol - current_sol
      step_size, _ = line_search.run(
          init_stepsize=1.0, params=current_sol, descent_direction=delta
      )

      new_sol = current_sol + step_size * delta
      return new_sol, loss(new_sol)

    if (init_linear_pb is None) and init_state is None:
      raise ValueError(
          "If initial solver state is None, init_linear_pb and \
            init_prob_state should be provided"
      )

    if init_state is None:
      init_state = self.init_state(init_linear_pb)

    return iterations(
        self, init_state, next_linearization, update_state_fn, progress_fn
    )


def iterations(
    solver: ConditionalGradient,
    init_state: ConditionalGradientState,
    next_linear_pb: Callable[[ConditionalGradientState],
                             linear_problem.LinearProblem],
    new_pstate_cost: Callable[
        [
            ConditionalGradientState,
            "LinearOutput",
        ],
        Tuple[jnp.array, float],
    ],
    progress_fn: Optional[ProgressCallbackFn_t] = None
) -> ConditionalGradientState:
  """Jittable ConditionalGradient outer loop."""

  def cond_fn(
      iteration: int, solver: ConditionalGradient,
      state: ConditionalGradientState
  ) -> bool:
    return solver._continue(state, iteration)

  def body_fn(
      iteration: int,
      solver: ConditionalGradient,
      state: ConditionalGradientState,
      compute_error: bool,
  ) -> ConditionalGradientState:
    del compute_error  # always assumed true for the outer loop

    linear_pb = next_linear_pb(state)
    # solving the lienar_pb
    init = (None, None)
    linear_pb_sol = solver.linear_ot_solver(linear_pb, init=init)

    # Updating the state
    new_sol_matrix, new_cost = new_pstate_cost(state, linear_pb_sol)

    new_state = state.update(
        iteration,
        new_cost,
        new_sol_matrix,
        linear_pb,
        linear_pb_sol,
        solver.store_inner_errors,
    )

    # Inner iterations is currently fixed to 1.
    inner_iterations = 1
    if progress_fn is not None:
      host_callback.id_tap(
          solver.progress_fn,
          (iteration, inner_iterations, solver.max_iterations, new_state),
      )

    return new_state

  return fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=solver,
      state=init_state,
  )
