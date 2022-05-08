# coding=utf-8
# Copyright 2022 Google LLC.
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
"""jheek@ backprop-friendly implementation of fixed point loop."""
from typing import Callable, Any

import jax
from jax import numpy as jnp
import numpy as np


def fixpoint_iter(cond_fn: Callable[[int, Any, Any], bool],
                  body_fn: Callable[[Any, Any, Any, Any], Any],
                  min_iterations: int,
                  max_iterations: int,
                  inner_iterations: int,
                  constants: Any,
                  state: Any):
  """Implementation of a fixed point loop.

  This fixed point loop iterator applies body_fn to a tuple
  (iteration, constants, state, compute_error) to output a new state, using
  context provided in iteration and constants.

  body_fn is iterated (inner_iterations -1) times, and one last time with the
  compute_error flag indicating that additional computational effort can be
  spent on recalculating the latest error (errors are stored as the first
  element of the state tuple).

  upon termination of these inner_iterations, the loop is continued if iteration
  is smaller than min_iterations, stopped if equal/larger than max_iterations,
  and interrupted if cond_fn returns False.

  Args:
    cond_fn : termination condition function
    body_fn : body loop instructions
    min_iterations : lower bound on the total amount of fixed point iterations
    max_iterations : upper bound on the total amount of fixed point iterations
    inner_iterations : number of iterations body_fn will be executed
      successively before calling cond_fn.
    constants : constant (during loop) parameters passed on to body
    state : state variable

  Returns:
    outputs state returned by body_fn upon termination.
  """
  # If number of minimal iterations matches maximal number, force a scan instead
  # of a while loop.

  force_scan = (min_iterations == max_iterations)

  compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1

  def max_cond_fn(iteration_state):
    iteration, state = iteration_state
    return jnp.logical_and(iteration < max_iterations,
                           jnp.logical_or(iteration < min_iterations,
                                          cond_fn(iteration, constants, state)))
  def unrolled_body_fn(iteration_state):
    def one_iteration(iteration_state, compute_error):
      iteration, state = iteration_state
      state = body_fn(iteration, constants, state, compute_error)
      iteration += 1
      return (iteration, state), None
    iteration_state, _ = jax.lax.scan(one_iteration, iteration_state,
                                      compute_error_flags)
    return (iteration_state, None) if force_scan else iteration_state

  if force_scan:
    (_, state), _ = jax.lax.scan(
        lambda carry, x: unrolled_body_fn(carry),
        (0, state), None,
        length=max_iterations // inner_iterations)
  else:
    _, state = jax.lax.while_loop(max_cond_fn, unrolled_body_fn, (0, state))
  return state


def fixpoint_iter_fwd(cond_fn, body_fn, min_iterations, max_iterations,
                      inner_iterations, constants, state):
  """Forward iteration of fixed point iteration to handle backpropagation.

  The main difference with fixpoint_iter is the checkpointing, in variable
  states, of the state variables as they are recorded through iterations, every
  inner_iterations. This sequence of states will be used in the backward loop.

  Args:
    cond_fn : termination condition function
    body_fn : body loop instructions
    min_iterations : lower bound on the total amount of fixed point iterations
    max_iterations : upper bound on the total amount of fixed point iterations
    inner_iterations : number of iterations body_fn will be executed
      successively before calling cond_fn.
    constants : constant (during loop) parameters passed on to body
    state : state variable

  Returns:
    outputs state returned by body_fn upon termination.
  """
  force_scan = (min_iterations == max_iterations)
  compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1
  states = jax.tree_map(lambda x: jnp.zeros(
      (max_iterations // inner_iterations + 1,) + x.shape,
      dtype=x.dtype), state)
  def max_cond_fn(iteration_states_state):
    iteration, _, state = iteration_states_state
    return jnp.logical_and(iteration < max_iterations,
                           jnp.logical_or(iteration < min_iterations,
                                          cond_fn(iteration, constants, state)))
  def unrolled_body_fn(iteration_states_state):
    iteration, states, state = iteration_states_state
    states = jax.tree_util.tree_map(
        lambda states, state: jax.lax.dynamic_update_index_in_dim(
            states, state, iteration // inner_iterations, 0), states, state)

    def one_iteration(iteration_state, compute_error):
      iteration, state = iteration_state
      state = body_fn(iteration, constants, state, compute_error)
      iteration += 1
      return (iteration, state), None

    iteration_state, _ = jax.lax.scan(one_iteration, (iteration, state),
                                      compute_error_flags)
    iteration, state = iteration_state
    out = (iteration, states, state)
    return (out, None) if force_scan else out

  if force_scan:
    (iteration, states, state), _ = jax.lax.scan(
        lambda carry, x: unrolled_body_fn(carry),
        (0, states, state), None,
        length=max_iterations // inner_iterations)
  else:
    iteration, states, state = jax.lax.while_loop(
        max_cond_fn, unrolled_body_fn, (0, states, state))

  return state, (constants, iteration, states)


def fixpoint_iter_bwd(
    cond_fn, body_fn, min_iterations, max_iterations, inner_iterations,
    res, g):
  """Backward iteration of fixed point iteration, using checkpointed states."""
  del cond_fn
  force_scan = (min_iterations == max_iterations)
  constants, iteration, states = res
  # The tree may contain some python floats
  g_constants = jax.tree_map(
      lambda x: jnp.zeros_like(x, dtype=x.dtype)
      if isinstance(x, (np.ndarray, jnp.ndarray)) else 0, constants)

  def bwd_cond_fn(iteration_g_gconst):
    iteration, _, _ = iteration_g_gconst
    return iteration >= 0

  def unrolled_body_fn_no_errors(iteration, constants, state):
    compute_error_flags = jnp.zeros((inner_iterations,), dtype=bool)
    def one_iteration(iteration_state, compute_error):
      iteration, state = iteration_state
      state = body_fn(iteration, constants, state, compute_error)
      iteration += 1
      return (iteration, state), None

    iteration_state, _ = jax.lax.scan(one_iteration, (iteration, state),
                                      compute_error_flags)
    _, state = iteration_state
    return state

  def unrolled_body_fn(iteration_g_gconst):
    iteration, g, g_constants = iteration_g_gconst
    state = jax.tree_map(lambda x: x[iteration // inner_iterations], states)
    _, pullback = jax.vjp(unrolled_body_fn_no_errors, iteration, constants,
                          state)
    _, gi_constants, g_state = pullback(g)
    g_constants = jax.tree_util.tree_map(lambda x, y: x + y, g_constants,
                                         gi_constants)
    out = (iteration - inner_iterations, g_state, g_constants)
    return (out, None) if force_scan else out

  if force_scan:
    (_, g_state, g_constants), _ = jax.lax.scan(
        lambda carry, x: unrolled_body_fn(carry),
        (0, g, g_constants), None,
        length=max_iterations // inner_iterations)
  else:
    _, g_state, g_constants = jax.lax.while_loop(
        bwd_cond_fn, unrolled_body_fn,
        (iteration - inner_iterations, g, g_constants))

  return g_constants, g_state

# definition of backprop friendly variant of fixpoint_iter.
fixpoint_iter_backprop = jax.custom_vjp(fixpoint_iter,
                                        nondiff_argnums=(0, 1, 2, 3, 4))

fixpoint_iter_backprop.defvjp(fixpoint_iter_fwd, fixpoint_iter_bwd)
