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

# Lint as: python3
"""A Jax backprop friendly version of Matrix square root."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from ott.core import fixed_point_loop


def sqrtm(x: jnp.ndarray,
          threshold: float = 1e-3,
          min_iterations: int = 0,
          inner_iterations: int = 10,
          max_iterations: int = 1000,
          regularization: float = 1e-3):
  """Implements Higham algorithm to compute matrix square root of p.d. matrix.

  See reference below, eq. 2.6.b
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf

  Args:
    x: a (batch of) square p.s.d. matrices of the same size.
    threshold: convergence tolerance threshold for Newton-Schulz iterations.
    min_iterations: min number of iterations after which error is computed.
    inner_iterations: error is re-evaluated every inner_iterations iterations.
    max_iterations: max number of iterations.
    regularization: small regularizer added to norm of x, before normalization.

  Returns:
    sqrt matrix of x (or x's if batch), its inverse, errors along iterates.
  """
  dimension = x.shape[-1]
  norm_x = jnp.linalg.norm(x, axis=(-2, -1)) * (1 + regularization)

  if jnp.ndim(x) == 3:
    norm_x = norm_x[:, jnp.newaxis, jnp.newaxis]

  def cond_fn(iteration, const, state):  # pylint: disable=unused-argument
    """Stopping criterion. Checking decrease of objective is needed here."""
    _, threshold = const
    errors, _, _ = state
    err = errors[iteration // inner_iterations-1]

    return jnp.logical_or(
        iteration == 0,
        jnp.logical_and(
            jnp.logical_and(jnp.isfinite(err), err > threshold),
            jnp.all(jnp.diff(errors) <= 0)))  # check decreasing obj, else stop

  def body_fn(iteration, const, state, compute_error):
    """Carries out matrix updates on y and z, stores error if requested.

    Args:
      iteration: iteration number
      const: tuple of constant parameters that do not change throughout the
        loop.
      state: state variables currently updated in the loop.
      compute_error: flag to indicate this iteration computes/stores an error

    Returns:
      state variables.
    """
    x, _ = const
    errors, y, z = state
    w = 0.5 * jnp.matmul(z, y)
    y = 1.5 * y - jnp.matmul(y, w)
    z = 1.5 * z - jnp.matmul(w, z)

    err = jnp.where(compute_error, new_err(x, norm_x, y), np.inf)

    errors = jax.ops.index_update(
        errors, jax.ops.index[iteration // inner_iterations], err)

    return errors, y, z

  def new_err(x, norm_x, y):
    res = x - norm_x * jnp.matmul(y, y)
    norm_fn = functools.partial(jnp.linalg.norm, axis=(-2, -1))
    return jnp.max(norm_fn(res) / norm_fn(x))

  y = x / norm_x
  z = jnp.eye(dimension)
  if jnp.ndim(x) == 3:
    z = jnp.tile(z, (x.shape[0], 1, 1))
  errors = -jnp.ones((np.ceil(max_iterations / inner_iterations).astype(int),))
  state = (errors, y, z)
  const = (x, threshold)
  errors, y, z = fixed_point_loop.fixpoint_iter_backprop(
      cond_fn, body_fn, min_iterations, max_iterations, inner_iterations, const,
      state)
  sqrt_x = jnp.sqrt(norm_x) * y
  inv_sqrt_x = z / jnp.sqrt(norm_x)

  return sqrt_x, inv_sqrt_x, errors
