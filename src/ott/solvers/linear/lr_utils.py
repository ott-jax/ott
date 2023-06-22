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
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem


def ibp_step(
    c_q: jnp.ndarray,
    c_r: jnp.ndarray,
    h: jnp.ndarray,
    gamma: float,
    ot_prob: linear_problem.LinearProblem,
    tolerance: float = 1e-3,
    min_iter: int = 0,
    inner_iter: int = 10,
    max_iter: int = 10000
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """TODO."""

  class State(NamedTuple):
    v1: jnp.ndarray
    v2: jnp.ndarray
    u1: jnp.ndarray
    u2: jnp.ndarray
    g: jnp.ndarray
    err: float

  class Constants(NamedTuple):
    log_a: jnp.ndarray
    log_b: jnp.ndarray
    tau_a: float
    tau_b: float

  def _softm(
      v: jnp.ndarray,
      c: jnp.ndarray,
      axis: int,
  ) -> jnp.ndarray:
    v = jnp.expand_dims(v, axis=1 - axis)
    return jsp.special.logsumexp(gamma * (v - c), axis=axis)

  def _error(
      v1: jnp.ndarray, v2: jnp.ndarray, u1: jnp.ndarray, u2: jnp.ndarray
  ) -> float:
    u1_trans = jnp.exp(_softm(v1, c_q, axis=1))
    err_1 = jnp.sum(jnp.abs(jnp.exp(gamma * u1) * u1_trans - ot_prob.a))
    u2_trans = jnp.exp(_softm(v2, c_r, axis=1))
    err_2 = jnp.sum(jnp.abs(jnp.exp(gamma * u2) * u2_trans - ot_prob.b))
    return err_1 + err_2

  def cond_fn(
      iteration: int,
      const: Constants,
      state: State,
  ) -> bool:
    del iteration, const
    return state.err > tolerance

  def body_fn(
      iteration: int, const: Constants, state: State, compute_error: bool
  ) -> State:
    # TODO(michalk8): handle 0s in a/b
    u1 = const.tau_a * (const.log_a - _softm(state.v1, c_q, axis=1)) / gamma
    v1_trans = _softm(u1, c_q, axis=0) / gamma

    u2 = const.tau_b * (const.log_b - _softm(state.v2, c_r, axis=1)) / gamma
    v2_trans = _softm(u2, c_r, axis=0) / gamma

    g = (1. / 3.) * (-h + v1_trans + v2_trans)
    v1 = g - v1_trans
    v2 = g - v2_trans

    err = jax.lax.cond(
        jnp.logical_and(compute_error, iteration >= min_iter), _error,
        lambda *_: state.err, v1, v2, u1, u2
    )

    return State(v1=v1, v2=v2, u1=u1, u2=u2, g=g, err=err)

  n, m, r = c_q.shape[0], c_r.shape[0], h.shape[0]
  constants = Constants(
      jnp.log(ot_prob.a), jnp.log(ot_prob.b), ot_prob.tau_a, ot_prob.tau_b
  )
  init_state = State(
      jnp.zeros(r), jnp.zeros(r), jnp.zeros(n), jnp.zeros(m), g=h, err=jnp.inf
  )

  state: State = fixed_point_loop.fixpoint_iter_backprop(
      cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, init_state
  )

  q = jnp.exp(gamma * (state.u1[:, None] + state.v1[None, :] - c_q))
  r = jnp.exp(gamma * (state.u2[:, None] + state.v2[None, :] - c_r))
  g = jnp.exp(gamma * state.g)

  return q, r, g
