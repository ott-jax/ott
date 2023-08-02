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
from typing import Any, Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp

from ott import utils
from ott.initializers.quadratic import initializers
from ott.math import fixed_point_loop
from ott.problems.quadratic import quadratic_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn_lr

__all__ = ["LRGromovWasserstein"]


class LRGWState(NamedTuple):
  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_state: sinkhorn_lr.LRSinkhornOutput
  errors: Optional[jnp.ndarray] = None


@jax.tree_util.register_pytree_node_class
class LRGromovWasserstein(was_solver.WassersteinSolver):

  def __init__(
      self,
      *args: Any,
      relative_epsilon: Optional[bool] = None,
      quad_initializer: initializers.LRQuadraticInitializer = None,
      progress_fn: Optional["ProgressCallbackFn_t"] = None,
      kwargs_init: Optional[Mapping[str, Any]] = None,
      **kwargs: Any
  ):
    super().__init__(*args, **kwargs)
    self.relative_epsilon = relative_epsilon
    self.quad_initializer = quad_initializer
    self.progress_fn = progress_fn
    self.kwargs_init = {} if kwargs_init is None else kwargs_init

  def __call__(
      self,
      prob: quadratic_problem.QuadraticProblem,
      init: Optional[sinkhorn_lr.LRSinkhornOutput] = None,
      rng: Optional[jax.random.PRNGKeyArray] = None,
      **kwargs: Any,
  ) -> "GWOutput":
    if prob._is_low_rank_convertible:
      prob = prob.to_low_rank()
    rng = utils.default_prng_key(rng)

    if init is None:
      init = self.quad_initializer(
          prob,
          epsilon=self.epsilon,
          rng=rng,
          relative_epsilon=self.relative_epsilon,
          **kwargs
      )

    return self._iterations(prob, init)

  def init_state(
      self,
      prob: quadratic_problem.QuadraticProblem,
      init: sinkhorn_lr.LRSinkhornOutput,
  ) -> LRGWState:
    """Initialize the state of the low-rank Gromov-Wasserstein iterations.

    Args:
      prob: Quadratic OT problem.
      init: Initial linearization of the quadratic problem.

    Returns:
      The initial low-rank Gromov-Wasserstein state.
    """
    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones((num_iter, self.linear_ot_solver.outer_iterations))
    else:
      errors = None

    return LRGWState(
        costs=-jnp.ones((num_iter,)),
        linear_convergence=-jnp.ones((num_iter,)),
        linear_state=init,
        errors=errors,
    )

  def output_from_state(
      self,
      state: LRGWState,
  ) -> "GWOutput":
    return state

  def _iterations(
      self,
      prob: quadratic_problem.QuadraticProblem,
      init: sinkhorn_lr.LRSinkhornOutput,
  ):

    def cond_fn(iteration: int, constants: Any, state: LRGWState) -> bool:
      del constants
      return self._continue(state, iteration)

    def body_fn(
        iteration: int, constants: Any, state: LRGWState, compute_error: bool
    ) -> LRGWState:
      del constants
      return state

    state = fixed_point_loop.fixpoint_iter(
        cond_fn=cond_fn,
        body_fn=body_fn,
        min_iterations=self.min_iterations,
        max_iterations=self.max_iterations,
        inner_iterations=1,
        constants=None,
        state=self.init_state(prob, init),
    )

    return self.output_from_state(state)
