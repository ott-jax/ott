# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

__all__ = ["solve_lineax"]


def _cg(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    maxiter: Optional[int] = None,
) -> jnp.ndarray:
  """Conjugate gradient solver using jax.lax.while_loop."""
  if maxiter is None:
    maxiter = 10 * b.shape[0]

  b_norm = jnp.linalg.norm(b)
  tol = jnp.maximum(atol, rtol * b_norm)

  x0 = jnp.zeros_like(b)
  r0 = b
  p0 = r0
  rtr0 = jnp.vdot(r0, r0)

  def cond_fun(state):
    _, _, _, rtr, k = state
    return (jnp.sqrt(rtr) > tol) & (k < maxiter)

  def body_fun(state):
    x, r, p, rtr, k = state
    Ap = matvec(p)
    alpha = rtr / jnp.vdot(p, Ap)
    x_new = x + alpha * p
    r_new = r - alpha * Ap
    rtr_new = jnp.vdot(r_new, r_new)
    beta = rtr_new / rtr
    p_new = r_new + beta * p
    return x_new, r_new, p_new, rtr_new, k + 1

  x, _, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, r0, p0, rtr0, 0))
  return x


def solve_lineax(
    lin: Callable,
    b: jnp.ndarray,
    lin_t: Optional[Callable] = None,
    symmetric: bool = False,
    nonsym_solver: Optional[Any] = None,
    ridge_identity: float = 0.0,
    ridge_kernel: float = 0.0,
    **kwargs: Any
) -> jnp.ndarray:
  """Solve a linear system using conjugate gradients.

  This implementation uses a JAX-native CG solver that works correctly inside
  JAX transformations (VJP backward pass), avoiding equinox closure conversion
  issues that affect lineax on certain JAX versions.

  Args:
    lin: Linear operator
    b: vector. Returned `x` is such that `lin(x)=b`
    lin_t: Linear operator, corresponding to transpose of `lin`.
    symmetric: whether `lin` is symmetric.
    nonsym_solver: unused, kept for API compatibility.
    ridge_kernel: promotes zero-sum solutions. Only use if `tau_a = tau_b = 1.0`
    ridge_identity: handles rank deficient transport matrices (this happens
      typically when rows/cols in cost/kernel matrices are collinear, or,
      equivalently when two points from either measure are close).
    kwargs: arguments passed to the CG solver (rtol, atol, maxiter).
  """
  kwargs.setdefault("rtol", 1e-6)
  kwargs.setdefault("atol", 1e-6)

  if ridge_kernel > 0.0 or ridge_identity > 0.0:
    lin_reg = lambda x: lin(x) + ridge_kernel * jnp.sum(x) + ridge_identity * x
    lin_t_reg = lambda x: lin_t(x) + ridge_kernel * jnp.sum(
        x
    ) + ridge_identity * x
  else:
    lin_reg, lin_t_reg = lin, lin_t

  if symmetric:
    return _cg(lin_reg, b, **kwargs)
  # Non-symmetric: solve normal equations A^T A x = A^T b
  normal_matvec = lambda x: lin_t_reg(lin_reg(x))
  normal_b = lin_t_reg(b)
  return _cg(normal_matvec, normal_b, **kwargs)
