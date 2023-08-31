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
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem

__all__ = ["unbalanced_dykstra_lse", "unbalanced_dykstra_kernel"]


class State(NamedTuple):  # noqa: D101
  v1: jnp.ndarray
  v2: jnp.ndarray
  u1: jnp.ndarray
  u2: jnp.ndarray
  g: jnp.ndarray
  err: float


class Constants(NamedTuple):  # noqa: D101
  a: jnp.ndarray
  b: jnp.ndarray
  rho_a: float
  rho_b: float
  supp_a: Optional[jnp.ndarray] = None
  supp_b: Optional[jnp.ndarray] = None


def unbalanced_dykstra_lse(
    c_q: jnp.ndarray,
    c_r: jnp.ndarray,
    c_g: jnp.ndarray,
    gamma: float,
    ot_prob: linear_problem.LinearProblem,
    translation_invariant: bool = True,
    tolerance: float = 1e-3,
    min_iter: int = 0,
    inner_iter: int = 10,
    max_iter: int = 10000
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Dykstra's algorithm for the unbalanced
  :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn` in LSE mode.

  Args:
    c_q: Cost associated with :math:`Q`.
    c_r: Cost associated with :math:`R`.
    c_g: Cost associated with :math:`g`.
    gamma: The (inverse of) the gradient step.
    ot_prob: Unbalanced OT problem.
    translation_invariant: Whether to use the translation invariant objective,
      see :cite:`scetbon:23`, alg. 3.
    tolerance: Convergence threshold.
    min_iter: Minimum number of iterations.
    inner_iter: Compute error every ``inner_iter``.
    max_iter: Maximum number of iterations.

  Returns:
    The :math:`Q`, :math:`R` and :math:`g` factors.
  """  # noqa: D205

  def _softm(
      v: jnp.ndarray,
      c: jnp.ndarray,
      axis: int,
  ) -> jnp.ndarray:
    v = jnp.expand_dims(v, axis=1 - axis)
    return jsp.special.logsumexp(v + c, axis=axis)

  def _error(
      gamma: float,
      new_state: State,
      old_state: State,
  ) -> float:
    u1_err = jnp.linalg.norm(new_state.u1 - old_state.u1, ord=jnp.inf)
    u2_err = jnp.linalg.norm(new_state.u2 - old_state.u2, ord=jnp.inf)
    v1_err = jnp.linalg.norm(new_state.v1 - old_state.v1, ord=jnp.inf)
    v2_err = jnp.linalg.norm(new_state.v2 - old_state.v2, ord=jnp.inf)
    return (1.0 / gamma) * jnp.max(jnp.array([u1_err, u2_err, v1_err, v2_err]))

  def cond_fn(
      iteration: int,
      const: Constants,
      state: State,
  ) -> bool:
    del iteration, const
    return tolerance < state.err

  def body_fn(
      iteration: int, const: Constants, state: State, compute_error: bool
  ) -> State:
    log_a, log_b = jnp.log(const.a), jnp.log(const.b)
    rho_a, rho_b = const.rho_a, const.rho_b

    c_a = _get_ratio(const.rho_a, gamma)
    c_b = _get_ratio(const.rho_b, gamma)

    if translation_invariant:
      lam_a, lam_b = compute_lambdas(const, state, gamma, g=c_g, lse_mode=True)

      u1 = c_a * (log_a - _softm(state.v1, c_q, axis=1))
      u1 = u1 - lam_a / ((1.0 / gamma) + rho_a)
      u2 = c_b * (log_b - _softm(state.v2, c_r, axis=1))
      u2 = u2 - lam_b / ((1.0 / gamma) + rho_b)

      state_lam = State(
          v1=state.v1, v2=state.v2, u1=u1, u2=u2, g=state.g, err=state.err
      )
      lam_a, lam_b = compute_lambdas(
          const, state_lam, gamma, g=c_g, lse_mode=True
      )

      v1_trans = _softm(u1, c_q, axis=0)
      v2_trans = _softm(u2, c_r, axis=0)

      g_trans = gamma * (lam_a + lam_b) + c_g
    else:
      u1 = c_a * (log_a - _softm(state.v1, c_q, axis=1))
      u2 = c_b * (log_b - _softm(state.v2, c_r, axis=1))

      v1_trans = _softm(u1, c_q, axis=0)
      v2_trans = _softm(u2, c_r, axis=0)
      g_trans = c_g

    g = (1.0 / 3.0) * (g_trans + v1_trans + v2_trans)
    v1 = g - v1_trans
    v2 = g - v2_trans

    new_state = State(v1=v1, v2=v2, u1=u1, u2=u2, g=g, err=jnp.inf)
    err = jax.lax.cond(
        jnp.logical_and(compute_error, iteration >= min_iter),
        _error,
        lambda *_: state.err,
        gamma,
        new_state,
        state,
    )
    return State(v1=v1, v2=v2, u1=u1, u2=u2, g=g, err=err)

  n, m, r = c_q.shape[0], c_r.shape[0], c_g.shape[0]
  constants = Constants(
      a=ot_prob.a,
      b=ot_prob.b,
      rho_a=_rho(ot_prob.tau_a),
      rho_b=_rho(ot_prob.tau_b),
      supp_a=ot_prob.a > 0,
      supp_b=ot_prob.b > 0,
  )
  init_state = State(
      v1=jnp.zeros(r),
      v2=jnp.zeros(r),
      u1=jnp.zeros(n),
      u2=jnp.zeros(m),
      g=c_g,
      err=jnp.inf,
  )

  state: State = fixed_point_loop.fixpoint_iter_backprop(
      cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, init_state
  )

  q = jnp.exp(state.u1[:, None] + c_q + state.v1[None, :])
  r = jnp.exp(state.u2[:, None] + c_r + state.v2[None, :])
  g = jnp.exp(state.g)

  return q, r, g


def unbalanced_dykstra_kernel(
    k_q: jnp.ndarray,
    k_r: jnp.ndarray,
    k_g: jnp.ndarray,
    gamma: float,
    ot_prob: linear_problem.LinearProblem,
    translation_invariant: bool = True,
    tolerance: float = 1e-3,
    min_iter: int = 0,
    inner_iter: int = 10,
    max_iter: int = 10000
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Dykstra's algorithm for the unbalanced
  :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn` in kernel mode.

  Args:
    k_q: Kernel associated with :math:`Q`.
    k_r: Kernel associated with :math:`R`.
    k_g: Kernel associated with :math:`g`.
    gamma: The (inverse of) the gradient step.
    ot_prob: Unbalanced OT problem.
    translation_invariant: Whether to use the translation invariant objective,
      see :cite:`scetbon:23`, alg. 3.
    tolerance: Convergence threshold.
    min_iter: Minimum number of iterations.
    inner_iter: Compute error every ``inner_iter``.
    max_iter: Maximum number of iterations.

  Returns:
    The :math:`Q`, :math:`R` and :math:`g` factors.
  """  # noqa: D205

  def _error(
      gamma: float,
      new_state: State,
      old_state: State,
  ) -> float:
    u1_err = jnp.linalg.norm(
        jnp.log(new_state.u1) - jnp.log(old_state.u1), ord=jnp.inf
    )
    u2_err = jnp.linalg.norm(
        jnp.log(new_state.u2) - jnp.log(old_state.u2), ord=jnp.inf
    )
    v1_err = jnp.linalg.norm(
        jnp.log(new_state.v1) - jnp.log(old_state.v1), ord=jnp.inf
    )
    v2_err = jnp.linalg.norm(
        jnp.log(new_state.v2) - jnp.log(old_state.v2), ord=jnp.inf
    )
    return (1.0 / gamma) * jnp.max(jnp.array([u1_err, u2_err, v1_err, v2_err]))

  def cond_fn(
      iteration: int,
      const: Constants,
      state: State,
  ) -> bool:
    del iteration, const
    return tolerance < state.err

  def body_fn(
      iteration: int, const: Constants, state: State, compute_error: bool
  ) -> State:
    c_a = _get_ratio(const.rho_a, gamma)
    c_b = _get_ratio(const.rho_b, gamma)

    if translation_invariant:
      lam_a, lam_b = compute_lambdas(const, state, gamma, g=k_g, lse_mode=False)

      u1 = jnp.where(const.supp_a, (const.a / (k_q @ state.v1)) ** c_a, 0.0)
      u1 = u1 * jnp.exp(-lam_a / ((1.0 / gamma) + const.rho_a))
      u2 = jnp.where(const.supp_b, (const.b / (k_r @ state.v2)) ** c_b, 0.0)
      u2 = u2 * jnp.exp(-lam_b / ((1.0 / gamma) + const.rho_b))

      state_lam = State(
          v1=state.v1, v2=state.v2, u1=u1, u2=u2, g=state.g, err=state.err
      )
      lam_a, lam_b = compute_lambdas(
          const, state_lam, gamma, g=k_g, lse_mode=False
      )

      v1_trans = k_q.T @ u1
      v2_trans = k_r.T @ u2

      k_trans = jnp.exp(gamma * (lam_a + lam_b)) * k_g
      g = (k_trans * v1_trans * v2_trans) ** (1.0 / 3.0)
    else:
      u1 = jnp.where(const.supp_a, (const.a / (k_q @ state.v1)) ** c_a, 0.0)
      u2 = jnp.where(const.supp_b, (const.b / (k_r @ state.v2)) ** c_b, 0.0)

      v1_trans = k_q.T @ u1
      v2_trans = k_r.T @ u2

      g = (k_g * v1_trans * v2_trans) ** (1.0 / 3.0)

    v1 = g / v1_trans
    v2 = g / v2_trans

    new_state = State(v1=v1, v2=v2, u1=u1, u2=u2, g=g, err=jnp.inf)
    err = jax.lax.cond(
        jnp.logical_and(compute_error, iteration >= min_iter),
        _error,
        lambda *_: state.err,
        gamma,
        new_state,
        state,
    )
    return State(v1=v1, v2=v2, u1=u1, u2=u2, g=g, err=err)

  n, m, r = k_q.shape[0], k_r.shape[0], k_g.shape[0]
  constants = Constants(
      a=ot_prob.a,
      b=ot_prob.b,
      rho_a=_rho(ot_prob.tau_a),
      rho_b=_rho(ot_prob.tau_b),
      supp_a=ot_prob.a > 0.0,
      supp_b=ot_prob.b > 0.0,
  )
  init_state = State(
      v1=jnp.ones(r),
      v2=jnp.ones(r),
      u1=jnp.ones(n),
      u2=jnp.ones(m),
      g=k_g,
      err=jnp.inf
  )

  state: State = fixed_point_loop.fixpoint_iter_backprop(
      cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, init_state
  )

  q = state.u1[:, None] * k_q * state.v1[None, :]
  r = state.u2[:, None] * k_r * state.v2[None, :]

  return q, r, state.g


def compute_lambdas(
    const: Constants, state: State, gamma: float, g: jnp.ndarray, *,
    lse_mode: bool
) -> Tuple[float, float]:
  """TODO."""
  gamma_inv = 1.0 / gamma
  rho_a = const.rho_a
  rho_b = const.rho_b

  if lse_mode:
    num_1 = jsp.special.logsumexp((-gamma_inv / rho_a) * state.u1, b=const.a)
    num_2 = jsp.special.logsumexp((-gamma_inv / rho_b) * state.u2, b=const.b)
    den = jsp.special.logsumexp(g - (state.v1 + state.v2))
    const_1 = num_1 - den
    const_2 = num_2 - den

    ratio_1 = _get_ratio(rho_a, gamma)
    ratio_2 = _get_ratio(rho_b, gamma)
    harmonic = 1.0 / (1.0 - (ratio_1 * ratio_2))
    lam_1 = harmonic * gamma_inv * ratio_1 * (const_1 - ratio_2 * const_2)
    lam_2 = harmonic * gamma_inv * ratio_2 * (const_2 - ratio_1 * const_1)
    return lam_1, lam_2

  num_1 = jnp.sum(
      jnp.where(
          const.supp_a, ((state.u1 ** (-gamma_inv / rho_a)) * const.a), 0.0
      )
  )
  num_2 = jnp.sum(
      jnp.where(
          const.supp_b, ((state.u2 ** (-gamma_inv / rho_b)) * const.b), 0.0
      )
  )
  den = jnp.sum(g / (state.v1 * state.v2))
  const_1 = jnp.log(num_1 / den)
  const_2 = jnp.log(num_2 / den)

  ratio_1 = _get_ratio(rho_a, gamma)
  ratio_2 = _get_ratio(rho_b, gamma)
  harmonic = 1.0 / (1.0 - (ratio_1 * ratio_2))
  lam_1 = harmonic * gamma_inv * ratio_1 * (const_1 - ratio_2 * const_2)
  lam_2 = harmonic * gamma_inv * ratio_2 * (const_2 - ratio_1 * const_1)
  return lam_1, lam_2


def _rho(tau: float) -> float:
  tau = jnp.asarray(tau)  # avoid division by 0 in Python, get NaN instead
  return tau / (1.0 - tau)


def _get_ratio(rho: float, gamma: float) -> float:
  gamma_inv = 1.0 / gamma
  return jnp.where(jnp.isfinite(rho), rho / (rho + gamma_inv), 1.0)
