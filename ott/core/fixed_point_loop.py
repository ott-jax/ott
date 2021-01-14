# coding=utf-8
# Copyright 2021 Google LLC.
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

"""jheek@ backprop-friendly implementation of fixed point loop."""
import functools

import jax
from jax import numpy as np


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def fixpoint_iter(cond_fn, body_fn, max_iterations, inner_iterations, constants,
                  state):
  """Implementation of a backprop friendly fixed point loop.

  Args:
    cond_fn : termination condition function
    body_fn : body loop instructions
    max_iterations : upper bound on outer loop fixed point iterations
    inner_iterations : default number of iterations in inner loop
    constants : constant (during loop) parameters passed on to body
    state : state variable
  Returns:
    outputs state returned by body_fn upon termination conditioned on cond_fn
  """
  def max_cond_fn(iteration_state):
    iteration, state = iteration_state
    return np.logical_and(iteration < max_iterations,
                          cond_fn(iteration, constants, state))
  def unrolled_body_fn(iteration_state):
    iteration, state = iteration_state
    for j in range(inner_iterations):
      state = body_fn(iteration, constants, state, j + 1 == inner_iterations)
      iteration += 1
    return iteration, state

  _, state = jax.lax.while_loop(max_cond_fn, unrolled_body_fn, (0, state))
  return state


def fixpoint_iter_fwd(cond_fn, body_fn, max_iterations, inner_iterations,
                      constants, state):
  """Forward iteration of fixed point iteration."""
  states = jax.tree_map(lambda x: np.zeros((max_iterations,) + x.shape), state)
  def max_cond_fn(iteration_states_state):
    iteration, _, state = iteration_states_state
    return np.logical_and(iteration < max_iterations,
                          cond_fn(iteration, constants, state))
  def unrolled_body_fn(iteration_states_state):
    iteration, states, state = iteration_states_state
    states = jax.tree_multimap(
        lambda states, state: jax.lax.dynamic_update_index_in_dim(
            states, state, iteration // inner_iterations, 0), states, state)
    for j in range(inner_iterations):
      state = body_fn(iteration, constants, state, j + 1 == inner_iterations)
      iteration += 1
    return iteration, states, state

  iteration, states, state = jax.lax.while_loop(max_cond_fn, unrolled_body_fn,
                                                (0, states, state))
  return state, (constants, iteration, states)


def fixpoint_iter_bwd(
    cond_fn, body_fn, max_iterations, inner_iterations, res, g):
  """Backward iteration of fixed point iteration."""
  del cond_fn, max_iterations
  constants, iteration, states = res
  g_constants = jax.tree_map(np.zeros_like, constants)
  def bwd_cond_fn(iteration_g_gconst):
    iteration, _, _ = iteration_g_gconst
    return iteration >= 0

  def f(iteration, constants, state):
    for j in range(inner_iterations):
      state = body_fn(iteration, constants, state, j + 1 == inner_iterations)
      iteration += 1
    return state

  def unrolled_body_fn(iteration_g_gconst):
    iteration, g, g_constants = iteration_g_gconst
    state = jax.tree_map(lambda x: x[iteration // inner_iterations], states)
    _, pullback = jax.vjp(f, iteration, constants, state)
    _, gi_constants, g_state = pullback(g)
    g_constants = jax.tree_multimap(lambda x, y: x + y, g_constants,
                                    gi_constants)
    return iteration - inner_iterations, g_state, g_constants

  iteration, g_state, g_constants = jax.lax.while_loop(
      bwd_cond_fn, unrolled_body_fn,
      (iteration - inner_iterations, g, g_constants))
  return g_constants, g_state

fixpoint_iter.defvjp(fixpoint_iter_fwd, fixpoint_iter_bwd)
