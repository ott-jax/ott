# coding=utf-8
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
from typing import Any, Dict, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from ott.core import fixed_point_loop
from ott.core import problems
from ott.core import bar_problems
from ott.core import was_solver
from ott.geometry import pointcloud


class BarycenterOutput(NamedTuple):
  """Holds the output of a Wasserstein Barycenter solver.

  The goal is to approximate the W barycenter of a set of N measures using
  a discrete measure described by k locations x. To do so the OT between
  each of the input N measures to the barycenter is recomputed and x_bar
  adjusted following that result.

  Attributes:
    costs: Holds the sequence of weighted sum of N regularized W costs seen
      (possibly debiased) through the outer loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner N Sinkhorn iterations.
    convergence: Bool convergence flag for the outer Barycenter iterations.
    errors: Holds sequence of matrices of N x max_iterations errors of the
      N Sinkhorn algorithms run at each inner iteration.
    x : barycenter locations, k x dimension
    a : weights of the barycenter
    transports: final N transport objects mapping barycenter to input measures.
    reg_gw_cost: Total regularized optimal transport cost upon convergence
  """
  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  convergence: bool = False
  errors: Optional[jnp.ndarray] = None
  x = None
  a = None
  transports = None
  reg_gw_cost = None
  
  def set(self, **kwargs) -> 'BarycenterOutput':
    """Returns a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)


class BarycenterState(NamedTuple):
  """Holds the state of the Wasserstein barycenter solver.

  Attributes:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
    linear_states: State used to solve and store solutions to the OT problems
      from the barycenter to the measures.
    x: barycenter points
    a: barycenter weights
  """
  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  errors: Optional[jnp.ndarray] = None
  x: Optional[jnp.ndarray] = None
  a: Optional[jnp.ndarray] = None

  def set(self, **kwargs) -> 'BarycenterState':
    """Returns a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)
  
  def update(self,
            iteration: int, 
            bar_prob: bar_problems.BarycenterProblem,
            linear_ot_solver: Any, 
            store_errors: bool):
    segmented_y, segmented_b = bar_prob.segmented_y_b

    @functools.partial(jax.vmap, in_axes=[None, None, 0, 0])
    def solve_linear_ot(a, x, b, y):
      out = linear_ot_solver(
        problems.LinearProblem(pointcloud.PointCloud(
          x, y, cost_fn = bar_prob.cost_fn, epsilon= bar_prob.epsilon),
        a, b))
      return (out.reg_ot_cost, out.converged, out.matrix,
            out.errors if store_errors else None)
    
    if bar_prob.debiased:
      # Check max size (used to pad) is bigger than barycenter size
      n, dim = self.x.shape
      max_size = bar_prob.max_measure_size      
      segmented_y = segmented_y.at[-1,:n,:].set(self.x)
      segmented_b = segmented_b.at[-1,:n].set(self.a)

    reg_ot_costs, convergeds, matrices, errors = solve_linear_ot(
      self.a, self.x, segmented_b, segmented_y)    
    
    cost = jnp.sum(reg_ot_costs * bar_prob.weights)
    updated_costs = self.costs.at[iteration].set(cost)
    converged = jnp.all(convergeds)
    linear_convergence = self.linear_convergence.at[iteration].set(converged)
    
    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :, :].set(errors)
    else:
      errors = None
    
    divide_a = jnp.where(self.a > 0, 1.0 / self.a, 1.0)
    convex_weights = matrices * divide_a[None, :, None]
    x_new = jnp.sum(
      barycentric_projection(convex_weights, segmented_y, bar_prob.cost_fn)
      * bar_prob.weights[:, None, None], axis=0)
    return self.set(costs=updated_costs,
                    linear_convergence=linear_convergence,
                    errors=errors,
                    x=x_new)

@functools.partial(jax.vmap, in_axes=[0, 0, None])
def barycentric_projection(matrix, y, cost_fn):
  return jax.vmap(cost_fn.barycenter, in_axes=[0, None])(matrix, y)

@jax.tree_util.register_pytree_node_class
class WassersteinBarycenter(was_solver.WassersteinSolver):
  """A Continuous Wasserstein barycenter solver, built on generic template."""

  def __call__(
      self,
      bar_prob: bar_problems.BarycenterProblem,
      bar_size: int = 100,
      x_init: jnp.ndarray = None,      
      rng: int = 0
      ) -> BarycenterState:    
    bar_fn = jax.jit(iterations, static_argnums=1) if self.jit else iterations
    out = bar_fn(self, bar_size, bar_prob, x_init, rng)
    return out

  def init_state(self, bar_prob, bar_size, x_init, rng
  ) -> BarycenterState:
    """Initializes the state of the Wasserstein barycenter iterations."""
    if x_init is not None:
      assert bar_size == x_init.shape[0]
      x = x_init
    else:
      # sample randomly points in the support of the y measures
      indices_subset = jax.random.choice(jax.random.PRNGKey(rng), 
        a=bar_prob.flattened_y.shape[0],
        shape=(bar_size,),
        replace=False,
        p=bar_prob.flattened_b)
      x = bar_prob.flattened_y[indices_subset,:]
    
    # TODO(cuturi) expand to non-uniform weights for barycenter.
    a = jnp.ones((bar_size,))/ bar_size
    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones(
        (num_iter, bar_prob.num_segments,
        self.linear_ot_solver.outer_iterations))
    else:
      errors = None
    return BarycenterState(-jnp.ones((num_iter,)), -jnp.ones((num_iter,)),
                   errors, x, a)

  def output_from_state(self, state):    
    return state

def iterations(solver: WassersteinBarycenter,
               bar_size, bar_prob, x_init, rng) -> BarycenterState:
  """A jittable Wasserstein barycenter outer loop."""  
  def cond_fn(iteration, constants, state):
    solver, _ = constants
    return solver._continue(state, iteration)

  def body_fn(iteration, constants, state, compute_error):
    del compute_error  # Always assumed True
    solver, bar_prob = constants    
    return state.update(
        iteration,
        bar_prob,
        solver.linear_ot_solver,
        solver.store_inner_errors)

  state = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=(solver, bar_prob),
      state=solver.init_state(bar_prob, bar_size, x_init, rng))
  
  return solver.output_from_state(state)
