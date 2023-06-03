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
import functools
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import pointcloud
from ott.math import fixed_point_loop
from ott.math import utils as mu
from ott.problems.linear import barycenter_problem, linear_problem
from ott.solvers import was_solver

__all__ = ["FreeBarycenterState", "FreeWassersteinBarycenter"]


class FreeBarycenterState(NamedTuple):
  """Holds the state of the Wasserstein barycenter solver.

  Args:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
    x: barycenter points.
    a: barycenter weights.
  """

  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  errors: Optional[jnp.ndarray] = None
  x: Optional[jnp.ndarray] = None
  a: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "FreeBarycenterState":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(
      self, iteration: int, bar_prob: barycenter_problem.FreeBarycenterProblem,
      linear_ot_solver: Any, store_errors: bool
  ) -> "FreeBarycenterState":
    """Update the state of the solver.

    Args:
      iteration: the current iteration of the outer loop.
      bar_prob: the barycenter problem.
      linear_ot_solver: the linear OT solver to use.
      store_errors: whether to store the errors of the inner loop.

    Returns:
      The updated state.
    """
    seg_y, seg_b = bar_prob.segmented_y_b

    @functools.partial(jax.vmap, in_axes=[None, None, 0, 0])
    def solve_linear_ot(
        a: Optional[jnp.ndarray], x: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray
    ):
      out = linear_ot_solver(
          linear_problem.LinearProblem(
              pointcloud.PointCloud(
                  x,
                  y,
                  src_mask=a > 0.,
                  tgt_mask=b > 0.,
                  cost_fn=bar_prob.cost_fn,
                  epsilon=bar_prob.epsilon
              ), a, b
          )
      )
      return (
          out.reg_ot_cost, out.converged, out.matrix,
          out.errors if store_errors else None
      )

    if bar_prob.debiased:
      raise NotImplementedError(
          "Debiased version of continuous Wasserstein barycenter "
          "not yet implemented."
      )

    reg_ot_costs, convergeds, matrices, errors = solve_linear_ot(
        self.a, self.x, seg_b, seg_y
    )

    cost = jnp.sum(reg_ot_costs * bar_prob.weights)
    updated_costs = self.costs.at[iteration].set(cost)
    converged = jnp.all(convergeds)
    linear_convergence = self.linear_convergence.at[iteration].set(converged)

    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :, :].set(errors)
    else:
      errors = None

    # Approximation of barycenter as barycenter of barycenters per measure.

    barycenters_per_measure = mu.barycentric_projection(
        matrices, seg_y, bar_prob.cost_fn
    )

    x_new = jax.vmap(
        lambda w, y: bar_prob.cost_fn.barycenter(w, y)[0], in_axes=[None, 1]
    )(bar_prob.weights, barycenters_per_measure)

    return self.set(
        costs=updated_costs,
        linear_convergence=linear_convergence,
        errors=errors,
        x=x_new
    )


@jax.tree_util.register_pytree_node_class
class FreeWassersteinBarycenter(was_solver.WassersteinSolver):
  """Continuous Wasserstein barycenter solver :cite:`cuturi:14`."""

  def __call__(  # noqa: D102
      self,
      bar_prob: barycenter_problem.FreeBarycenterProblem,
      bar_size: int = 100,
      x_init: Optional[jnp.ndarray] = None,
      rng: Optional[jax.random.PRNGKeyArray] = None,
  ) -> FreeBarycenterState:
    # TODO(michalk8): no reason for iterations to be outside this class
    rng = utils.default_prng_key(rng)
    return iterations(self, bar_size, bar_prob, x_init, rng)

  def init_state(
      self,
      bar_prob: barycenter_problem.FreeBarycenterProblem,
      bar_size: int,
      x_init: Optional[jnp.ndarray] = None,
      rng: Optional[jax.random.PRNGKeyArray] = None,
  ) -> FreeBarycenterState:
    """Initialize the state of the Wasserstein barycenter iterations.

    Args:
      bar_prob: The barycenter problem.
      bar_size: Size of the barycenter.
      x_init: Initial barycenter estimate of shape ``[bar_size, ndim]``.
        If `None`, ``bar_size`` points will be sampled from the input
        measures according to their weights
        :attr:`~ott.problems.linear.barycenter_problem.FreeBarycenterProblem.flattened_y`.
      rng: Random key for seeding.

    Returns:
      The initial barycenter state.
    """
    if x_init is not None:
      assert bar_size == x_init.shape[0]
      x = x_init
    else:
      # sample randomly points in the support of the y measures
      rng = utils.default_prng_key(rng)
      indices_subset = jax.random.choice(
          rng,
          a=bar_prob.flattened_y.shape[0],
          shape=(bar_size,),
          replace=False,
          p=bar_prob.flattened_b
      )
      x = bar_prob.flattened_y[indices_subset, :]

    # TODO(cuturi) expand to non-uniform weights for barycenter.
    a = jnp.ones((bar_size,)) / bar_size
    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones((
          num_iter, bar_prob.num_measures,
          self.linear_ot_solver.outer_iterations
      ))
    else:
      errors = None
    return FreeBarycenterState(
        -jnp.ones((num_iter,)), -jnp.ones((num_iter,)), errors, x, a
    )

  def output_from_state(  # noqa: D102
      self, state: FreeBarycenterState
  ) -> FreeBarycenterState:
    # TODO(michalk8): create an output variable to match rest of the framework
    return state


def iterations(
    solver: FreeWassersteinBarycenter, bar_size: int,
    bar_prob: barycenter_problem.FreeBarycenterProblem, x_init: jnp.ndarray,
    rng: jax.random.PRNGKeyArray
) -> FreeBarycenterState:
  """Jittable Wasserstein barycenter outer loop."""

  def cond_fn(
      iteration: int,
      constants: Tuple[FreeWassersteinBarycenter,
                       barycenter_problem.FreeBarycenterProblem],
      state: FreeBarycenterState
  ) -> bool:
    solver, _ = constants
    return solver._continue(state, iteration)

  def body_fn(
      iteration, constants: Tuple[FreeWassersteinBarycenter,
                                  barycenter_problem.FreeBarycenterProblem],
      state: FreeBarycenterState, compute_error: bool
  ) -> FreeBarycenterState:
    del compute_error  # Always assumed True
    solver, bar_prob = constants
    return state.update(
        iteration, bar_prob, solver.linear_ot_solver, solver.store_inner_errors
    )

  state = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=(solver, bar_prob),
      state=solver.init_state(bar_prob, bar_size, x_init, rng)
  )

  return solver.output_from_state(state)
