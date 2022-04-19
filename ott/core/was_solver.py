# coding=utf-8
# Copyright 2022 Apple Inc.
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
"""A Jax version of the regularised GW Solver (Peyre et al. 2016)."""
import functools
from typing import Any, Dict, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from ott.core import sinkhorn
from ott.core import sinkhorn_lr

@jax.tree_util.register_pytree_node_class
class WassersteinSolver:
  """A generic solver for problems that use a linear reg-OT pb in inner loop."""
  def __init__(self,
               epsilon: Optional[float] = None,
               rank: int = -1,
               linear_ot_solver: Any = None,
               min_iterations: int = 5,
               max_iterations: int = 50,
               threshold: float = 1e-3,
               jit: bool = True,
               store_inner_errors: bool = False,
               **kwargs):
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
            rank=self.rank, jit=False, **kwargs)
        else:
          # If epsilon is passed, use it to replace the default LRSinkhorn value
          self.linear_ot_solver = sinkhorn_lr.LRSinkhorn(
            rank=self.rank,
            epsilon=self.epsilon, **kwargs)
      else:
        # When using Entropic GW, epsilon is not handled inside Sinkhorn, 
        # but rather added back to the Geometry object reinstantiated 
        # when linearizing the problem. Therefore no need to pass it to solver.
        self.linear_ot_solver = sinkhorn.Sinkhorn(**kwargs)

    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.threshold = threshold
    self.jit = jit
    self.store_inner_errors = store_inner_errors
    self._kwargs = kwargs

  @property
  def is_low_rank(self):
    return self.rank > 0

  def tree_flatten(self):
    return ([self.epsilon, self.rank,
             self.linear_ot_solver, self.threshold],
            dict(
                min_iterations=self.min_iterations,
                max_iterations=self.max_iterations,
                jit=self.jit,
                store_inner_errors=self.store_inner_errors,
                **self._kwargs))

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(
        epsilon=children[0],
        rank=children[1],
        linear_ot_solver=children[2],
        threshold=children[3],
        **aux_data)

  def _converged(self, state, iteration):
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_and(
        i >= 2,
        jnp.isclose(costs[i - 2], costs[i - 1], rtol=tol))

  def _diverged(self, state, iteration):
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_not(jnp.isfinite(costs[i - 1]))

  def _continue(self, state, iteration):
    """ continue while not(converged) and not(diverged)"""
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_or(
        i <= 2,
        jnp.logical_and(
            jnp.logical_not(self._diverged(state, iteration)),
            jnp.logical_not(self._converged(state, iteration))))
