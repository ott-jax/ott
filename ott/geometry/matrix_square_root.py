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
"""A Jax backprop friendly version of Matrix square root."""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ott.core import fixed_point_loop


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def sqrtm(
    x: jnp.ndarray,
    threshold: float = 1e-3,
    min_iterations: int = 0,
    inner_iterations: int = 10,
    max_iterations: int = 1000,
    regularization: float = 1e-3
):
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

  if jnp.ndim(x) > 2:
    norm_x = norm_x[..., jnp.newaxis, jnp.newaxis]

  def cond_fn(iteration, const, state):
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

    errors = errors.at[iteration // inner_iterations].set(err)

    return errors, y, z

  def new_err(x, norm_x, y):
    res = x - norm_x * jnp.matmul(y, y)
    norm_fn = functools.partial(jnp.linalg.norm, axis=(-2, -1))
    return jnp.max(norm_fn(res) / norm_fn(x))

  dtype = x.dtype
  y = x / norm_x
  z = jnp.eye(dimension, dtype=dtype)
  if jnp.ndim(x) > 2:
    z = jnp.tile(z, list(x.shape[:-2]) + [1, 1])
  errors = -jnp.ones((np.ceil(max_iterations / inner_iterations).astype(int),),
                     dtype=dtype)
  state = (errors, y, z)
  const = (x, threshold)
  errors, y, z = fixed_point_loop.fixpoint_iter_backprop(
      cond_fn, body_fn, min_iterations, max_iterations, inner_iterations, const,
      state)
  sqrt_x = jnp.sqrt(norm_x) * y
  inv_sqrt_x = z / jnp.sqrt(norm_x)

  return sqrt_x, inv_sqrt_x, errors


def solve_sylvester_bartels_stewart(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
  """Solve the real Sylvester equation AX - XB = C using Bartels-Stewart."""
  # See https://nhigham.com/2020/09/01/what-is-the-sylvester-equation/ for
  # discussion of the algorithm (but note that in the derivation, the sign on
  # the right hand side is flipped in the equation in which the columns are set
  # to be equal).
  m = a.shape[-1]
  n = b.shape[-1]
  # Cast a and b to complex to ensure we get the complex Schur decomposition
  # (the real Schur decomposition may not give an upper triangular solution).
  # For the decomposition below, a = u r u* and b = v s v*
  r, u = jax.lax.linalg.schur(a + 0j)
  s, v = jax.lax.linalg.schur(b + 0j)
  d = jnp.matmul(jnp.conjugate(jnp.swapaxes(u, axis1=-2, axis2=-1)),
                 jnp.matmul(c, v))
  # The solution in the transformed space will in general be complex, too.
  y = jnp.zeros(a.shape[:-2] + (m, n)) + 0j
  idx = jnp.arange(m, dtype=jnp.int32)
  for j in range(n):
    lhs = r.at[..., idx, idx].add(-s[..., j:j+1, j])
    rhs = d[..., j] + jnp.matmul(y[..., :j], s[..., :j, j:j+1])[..., 0]
    y = y.at[..., j].set(jax.scipy.linalg.solve_triangular(lhs, rhs))

  x = jnp.matmul(
      u, jnp.matmul(y, jnp.conjugate(jnp.swapaxes(v, axis1=-2, axis2=-1))))
  # The end result should be real; remove the imaginary part of the solution.
  return jnp.real(x)


def sqrtm_fwd(
    x: jnp.ndarray,
    threshold: float,
    min_iterations: int,
    inner_iterations: int,
    max_iterations: int,
    regularization: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
           Tuple[jnp.ndarray, jnp.ndarray]]:
  """Forward pass of custom VJP."""
  sqrt_x, inv_sqrt_x, errors = sqrtm(
      x=x,
      threshold=threshold,
      min_iterations=min_iterations,
      inner_iterations=inner_iterations,
      max_iterations=max_iterations,
      regularization=regularization,
      )
  return ((sqrt_x, inv_sqrt_x, errors), (sqrt_x, inv_sqrt_x))


def sqrtm_bwd(
    threshold: float,
    min_iterations: int,
    inner_iterations: int,
    max_iterations: int,
    regularization: float,
    residual: Tuple[jnp.ndarray, jnp.ndarray],
    cotangent: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray,]:
  """Compute the derivative by solving a Sylvester equation."""
  del threshold, min_iterations, inner_iterations, max_iterations, regularization
  sqrt_x, inv_sqrt_x = residual
  # ignores cotangent associated with errors
  cot_sqrt, cot_inv_sqrt, _ = cotangent

  # Solve for d(X^{1/2}):
  # Start with X^{1/2} X^{1/2} = X
  # Differentiate to obtain
  # d(X^{1/2}) X^{1/2} + X^{1/2} d(X^{1/2}) = dX
  # The above is a Sylvester equation that we can solve using Bartels-Stewart.
  # Below think of cot_sqrt as (dX)^T and vjp_cot_sqrt as d(X^{1/2})^T.
  # See https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
  vjp_cot_sqrt = jnp.swapaxes(
      solve_sylvester_bartels_stewart(
          a=sqrt_x, b=-sqrt_x,
          c=jnp.swapaxes(cot_sqrt, axis1=-1, axis2=-2)),
      axis1=-1, axis2=-2)

  # Now solve for d(X^{-1/2}):
  # Start with X^{-1/2} X^{-1/2} = X^{-1}
  # Use the product rule and the fact that d(X^{-1}) = -X^{-1} dX X^{-1}
  # to obtain
  # (See The Matrix Cookbook section on derivatives of an inverse
  # https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf )
  # d(X^{-1/2}) X^{-1/2} + X^{-1/2} d(X^{-1/2}) = -X^{-1} dX X^{-1}
  # Again we have a Sylvester equation that we solve as above, and again we
  # think of cot_inv_sqrt as (dX)^T and vjp_cot_inv_sqrt as d(X^{-1/2})^T
  inv_x = jnp.matmul(inv_sqrt_x, inv_sqrt_x)
  vjp_cot_inv_sqrt = jnp.swapaxes(
      solve_sylvester_bartels_stewart(
          a=inv_sqrt_x, b=-inv_sqrt_x,
          c=-jnp.matmul(
              inv_x,
              jnp.matmul(jnp.swapaxes(cot_inv_sqrt, axis1=-2, axis2=-1),
                         inv_x))),
      axis1=-1, axis2=-2)
  return (vjp_cot_sqrt + vjp_cot_inv_sqrt,)


sqrtm.defvjp(sqrtm_fwd, sqrtm_bwd)


# Specialized versions of sqrtm that compute only the square root or inverse.
# These functions have lower complexity gradients than sqrtm.


@jax.custom_vjp
def sqrtm_only(
    x: jnp.ndarray
) -> jnp.ndarray:
  return sqrtm(x)[0]


def sqrtm_only_fwd(
    x: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  sqrt_x = sqrtm(x)[0]
  return sqrt_x, sqrt_x


def sqrtm_only_bwd(
    sqrt_x: jnp.ndarray,
    cotangent: jnp.ndarray
) -> Tuple[jnp.ndarray]:
  vjp = jnp.swapaxes(
      solve_sylvester_bartels_stewart(
          a=sqrt_x, b=-sqrt_x,
          c=jnp.swapaxes(cotangent, axis1=-2, axis2=-1)),
      axis1=-2, axis2=-1)
  return (vjp,)


sqrtm_only.defvjp(sqrtm_only_fwd, sqrtm_only_bwd)


@jax.custom_vjp
def inv_sqrtm_only(
    x: jnp.ndarray,
) -> jnp.ndarray:
  return sqrtm(x)[1]


def inv_sqrtm_only_fwd(
    x: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  inv_sqrt_x = sqrtm(x)[1]
  return inv_sqrt_x, inv_sqrt_x


def inv_sqrtm_only_bwd(
    residual: jnp.ndarray,
    cotangent: jnp.ndarray) -> jnp.ndarray:
  inv_sqrt_x = residual
  inv_x = jnp.matmul(inv_sqrt_x, inv_sqrt_x)
  vjp = jnp.swapaxes(
      solve_sylvester_bartels_stewart(
          a=inv_sqrt_x, b=-inv_sqrt_x,
          c=-jnp.matmul(
              inv_x,
              jnp.matmul(jnp.swapaxes(cotangent, axis1=-2, axis2=-1),
                         inv_x))),
      axis1=-1, axis2=-2)
  return (vjp,)


inv_sqrtm_only.defvjp(inv_sqrtm_only_fwd, inv_sqrtm_only_bwd)
