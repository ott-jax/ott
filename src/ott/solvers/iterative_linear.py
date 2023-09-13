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
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import host_callback

from ott import utils
from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn, sinkhorn_lr

__all__ = ["IterativeLinearSolver", "IterLinState"]

LinearOutput = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput]

ProgressCallbackFn_t = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "IterLinState"]], None]


class IterLinState(NamedTuple):
  """State of the IterativeLinearSolver.

  Attributes:
  costs: Holds the sequence of costs seen through the outer
  loop of the solver.
  linear_convergence: Holds the sequence of bool convergence flags of the
  inner linear solvers.
  linear_pb: last inner linear_problem
  linear_state: solution to the linear_pb
  prob_state: a problem specific state, provided and managed by
    user-defined functions
  rngs: random keys passed to the user defined functions at each iteration.
  errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
  at each iteration.
  """

  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_pb: linear_problem.LinearProblem
  linear_state: LinearOutput
  prob_state: NamedTuple = None
  rngs: Optional[jax.random.PRNGKeyArray] = None
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "IterLinState":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(  # noqa: D102
      self,
      iteration: int,
      cost: float,
      linear_pb: linear_problem.LinearProblem,
      linear_sol: LinearOutput,
      prob_state: NamedTuple,
      store_errors: bool,
  ) -> "IterLinState":
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
        prob_state=prob_state,
        costs=costs,
        linear_convergence=linear_convergence,
        errors=errors,
    )


@jax.tree_util.register_pytree_node_class
class IterativeLinearSolver(was_solver.WassersteinSolver):
  """Implements probems requiring iterative calls to (low_rank) sinkhorn solver.

  Args:
  args: Positional arguments for
  :class:`~ott.solvers.was_solver.WassersteinSolver`.
  warm_start: Whether to initialize (low-rank) Sinkhorn calls using values
  from the previous iteration. If `None`, warm starts are not used for
  standard Sinkhorn, but used for low-rank Sinkhorn.
  progress_fn: callback function which gets called during the
  inner iterations, so the user can display the error at each
  iteration, e.g., using a progress bar.
  See :func:`~ott.utils.default_progress_fn` for a basic implementation.
  kwargs_init: auxilary key arguments
  """

  def __init__(
      self,
      *args: Any,
      warm_start: Optional[bool] = None,
      progress_fn: Optional[ProgressCallbackFn_t] = None,
      kwargs_init: Optional[Mapping[str, Any]] = None,
      **kwargs: Any,
  ):
    super().__init__(*args, **kwargs)
    self._warm_start = warm_start
    self.progress_fn = progress_fn
    self.kwargs_init = {} if kwargs_init is None else kwargs_init

  def tree_flatten(self,) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    children, aux_data = super().tree_flatten()
    aux_data["warm_start"] = self._warm_start
    aux_data["progress_fn"] = self.progress_fn
    aux_data["kwargs_init"] = self.kwargs_init
    return children, aux_data

  def init_state(
      self,
      init_linear_pb: linear_problem.LinearProblem,
      init_prob_state: NamedTuple,
      rng: jax.random.PRNGKeyArray,
  ) -> IterLinState:
    """Initialize the state of the IterativeLinearSolver.

    Args:
    init_linear_pb: initialization for linear OT problem
    init_prob_state: initalization of prob_state
    rng: Random key for low-rank initializers. Only used when
    :attr:`warm_start` is `False`.

    Returns:
    The initial IterLinState .
    """
    linear_state = self.linear_ot_solver(init_linear_pb)
    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones((num_iter, self.linear_ot_solver.outer_iterations))
    else:
      errors = None

    return IterLinState(
        costs=-jnp.ones((num_iter,)),
        linear_convergence=-jnp.ones((num_iter,)),
        linear_pb=init_linear_pb,
        linear_state=linear_state,
        prob_state=init_prob_state,
        rngs=jax.random.split(rng, num_iter),
        errors=errors,
    )

  def __call__(
      self,
      next_linear_pb: Callable[
          [IterLinState, jax.random.PRNGKeyArray],
          linear_problem.LinearProblem,
      ],
      new_pstate_cost: Callable[
          [
              IterLinState,
              linear_problem.LinearProblem,
              "LinearOutput",
              jax.random.PRNGKeyArray,
          ],
          Tuple[NamedTuple, float],
      ],
      init_linear_pb: Optional[linear_problem.LinearProblem] = None,
      init_prob_state: Optional[NamedTuple] = None,
      init_state: Optional[IterLinState] = None,
      rng: Optional[jax.random.PRNGKeyArray] = None,
  ) -> IterLinState:
    """Run the generic solver.

    Args:
    next_linear_pb: a user-defined function that takes the current state a
      random key and outputs a new linear problem
    new_pstate_cost: a user-defined function that takes the current state,
      the new linear problem, new linear solution, a random key and and
      outputs a new state
    init_linear_pb: the initial linear problem to be solved
    init_prob_state: initialization for the problem state
    init_state: allows the user to directly define the initial state.
      If provided init_prob and init-aux state are ignored
    rng: Random number key.

    Returns:
    Last solver state.
    """
    if rng is None:
      rng = utils.default_prng_key(rng)

    if (
        init_linear_pb is None or init_prob_state is None
    ) and init_state is None:
      raise ValueError(
          "If initial solver state is None, init_linear_pb and \
            init_prob_state should be provided"
      )
    if init_state is None:
      init_state = self.init_state(init_linear_pb, init_prob_state, rng)

    return iterations(self, init_state, next_linear_pb, new_pstate_cost)

  @property
  def warm_start(self) -> bool:
    """Whether to initialize (low-rank) Sinkhorn using previous solutions."""
    return (self.is_low_rank if self._warm_start is None else self._warm_start)


def iterations(
    solver: IterativeLinearSolver,
    init_state: IterLinState,
    next_linear_pb: Callable[[IterLinState, jax.random.PRNGKeyArray],
                             linear_problem.LinearProblem],
    new_pstate_cost: Callable[
        [
            IterLinState,
            linear_problem.LinearProblem,
            "LinearOutput",
            jax.random.PRNGKeyArray,
        ],
        Tuple[NamedTuple, float],
    ],
) -> IterLinState:
  """Jittable IterativeLinearSolver outer loop."""

  def cond_fn(
      iteration: int, solver: IterativeLinearSolver, state: IterLinState
  ) -> bool:
    return solver._continue(state, iteration)

  def body_fn(
      iteration: int,
      solver: IterativeLinearSolver,
      state: IterLinState,
      compute_error: bool,
  ) -> IterLinState:
    del compute_error  # always assumed true for the outer loop
    rng = state.rngs[iteration]
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    lin_state = state.linear_state
    linear_pb = next_linear_pb(state, rng1)
    # solving the lienar_pb
    if solver.is_low_rank:
      init = ((lin_state.q, lin_state.r, lin_state.g) if solver.warm_start else
              (None, None, None))
      linear_pb_sol = solver.linear_ot_solver(linear_pb, init=init, rng=rng2)
    else:
      init = ((lin_state.f, lin_state.g) if solver.warm_start else (None, None))
      linear_pb_sol = solver.linear_ot_solver(linear_pb, init=init, rng=rng2)

    # Updating the state
    new_prob_state, new_cost = new_pstate_cost(
        state, linear_pb, linear_pb_sol, rng3
    )

    new_state = state.update(
        iteration,
        new_cost,
        linear_pb,
        linear_pb_sol,
        new_prob_state,
        solver.store_inner_errors,
    )

    # Inner iterations is currently fixed to 1.
    inner_iterations = 1
    if solver.progress_fn is not None:
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
