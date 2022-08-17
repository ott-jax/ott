# Copyright 2022 Apple.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A Jax version of the W barycenter algorithm (Cuturi Doucet 2014)."""
import functools
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from ott.core import bar_problems, fixed_point_loop, linear_problems, was_solver
from ott.geometry import pointcloud

__all__ = ["BarycenterState", "WassersteinBarycenter"]


class BarycenterState(NamedTuple):
  """Holds the state of the Wasserstein barycenter solver.

  Args:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
    linear_states: State used to solve and store solutions to the OT problems
      from the barycenter to the measures.
    x: barycenter points.
    a: barycenter weights.
  """

  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  errors: Optional[jnp.ndarray] = None
  x: Optional[jnp.ndarray] = None
  a: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> 'BarycenterState':
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(
      self, iteration: int, bar_prob: bar_problems.BarycenterProblem,
      linear_ot_solver: Any, store_errors: bool
  ) -> 'BarycenterState':
    seg_y, seg_b = bar_prob.segmented_y_b

    @functools.partial(jax.vmap, in_axes=[None, None, 0, 0])
    def solve_linear_ot(
        a: Optional[jnp.ndarray], x: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray
    ):
      out = linear_ot_solver(
          linear_problems.LinearProblem(
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

    barycenters_per_measure = bar_problems.barycentric_projection(
        matrices, seg_y, bar_prob.cost_fn
    )

    x_new = jax.vmap(
        bar_prob.cost_fn.barycenter, in_axes=[None, 1]
    )(bar_prob.weights, barycenters_per_measure)

    return self.set(
        costs=updated_costs,
        linear_convergence=linear_convergence,
        errors=errors,
        x=x_new
    )


@jax.tree_util.register_pytree_node_class
class WassersteinBarycenter(was_solver.WassersteinSolver):
  """A Continuous Wasserstein barycenter solver, built on generic template."""

  def __call__(
      self,
      bar_prob: bar_problems.BarycenterProblem,
      bar_size: int = 100,
      x_init: Optional[jnp.ndarray] = None,
      rng: int = 0
  ) -> BarycenterState:
    bar_fn = jax.jit(iterations, static_argnums=1) if self.jit else iterations
    out = bar_fn(self, bar_size, bar_prob, x_init, rng)
    return out

  def init_state(
      self,
      bar_prob: bar_problems.BarycenterProblem,
      bar_size: int,
      x_init: Optional[jnp.ndarray] = None,
      # TODO(michalk8): change the API to pass the PRNG key directly
      rng: int = 0,
  ) -> BarycenterState:
    """Initialize the state of the Wasserstein barycenter iterations.

    Args:
      bar_prob: The barycenter problem.
      bar_size: Size of the barycenter.
      x_init: Initial barycenter estimate of shape ``[bar_size, ndim]``.
        If `None`, ``bar_size`` points will be sampled from the input
        measures according to their weights
        :attr:`~ott.core.bar_problems.BarycenterProblem.flattened_y`.
      rng: Seed for :func:`jax.random.PRNGKey`.

    Returns:
      The initial barycenter state.
    """
    if x_init is not None:
      assert bar_size == x_init.shape[0]
      x = x_init
    else:
      # sample randomly points in the support of the y measures
      indices_subset = jax.random.choice(
          jax.random.PRNGKey(rng),
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
    return BarycenterState(
        -jnp.ones((num_iter,)), -jnp.ones((num_iter,)), errors, x, a
    )

  def output_from_state(self, state: BarycenterState) -> BarycenterState:
    return state


def iterations(
    solver: WassersteinBarycenter, bar_size: int,
    bar_prob: bar_problems.BarycenterProblem, x_init: jnp.ndarray, rng: int
) -> BarycenterState:
  """Jittable Wasserstein barycenter outer loop."""

  def cond_fn(
      iteration: int, constants: Tuple[WassersteinBarycenter,
                                       bar_problems.BarycenterProblem],
      state: BarycenterState
  ) -> bool:
    solver, _ = constants
    return solver._continue(state, iteration)

  def body_fn(
      iteration, constants: Tuple[WassersteinBarycenter,
                                  bar_problems.BarycenterProblem],
      state: BarycenterState, compute_error: bool
  ) -> BarycenterState:
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
