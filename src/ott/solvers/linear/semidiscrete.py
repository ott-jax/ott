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
import dataclasses
import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu

import optax

from ott.math import fixed_point_loop
from ott.math import utils as math_utils
from ott.problems.linear import linear_problem, semidiscrete_linear_problem
from ott.solvers.linear import sinkhorn

__all__ = ["SemidiscreteState", "SemidiscreteOutput", "SemidiscreteSolver"]


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteState:
  """TODO."""
  it: jax.Array
  g: jax.Array
  g_ema: jax.Array
  opt_state: Any
  losses: jax.Array
  errors: jax.Array


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteOutput:
  """TODO."""
  it: int
  g: jax.Array
  prob: semidiscrete_linear_problem.SemidiscreteLinearProblem
  losses: jax.Array
  errors: jax.Array
  converged: bool

  def matrix(
      self,
      rng: jax.Array,
      num_samples: int,
  ) -> jax.Array:
    """TODO."""
    if self.prob.geom.is_entropy_regularized:
      out = self.materialize(rng, num_samples)
      return out.matrix

    geom = self.prob.materialize(rng, num_samples).geom
    z = self.g[None, :] - geom.cost_matrix
    return _hardmax(z)

  def marginal_chi2_error(
      self,
      rng: jax.Array,
      *,
      num_iters: int,
      batch_size: int,
  ) -> jax.Array:
    """TODO."""
    return _marginal_chi2_error(
        rng,
        self.g,
        self.prob,
        num_iters=num_iters,
        batch_size=batch_size,
    )

  def materialize(
      self, rng: jax.Array, num_samples: int
  ) -> sinkhorn.SinkhornOutput:
    """TODO."""
    epsilon = self.prob.geom.epsilon
    prob = self.prob.sample(rng, num_samples)

    f, _ = _soft_c_transform(self.g, prob)
    # SinkhornOutput's potentials must contain
    # probability weight normalization
    f_tilde = f + epsilon * jnp.log(1.0 / num_samples)
    g_tilde = self.g + epsilon * jnp.log(self.prob.b)

    return sinkhorn.SinkhornOutput(
        potentials=(f_tilde, g_tilde),
        ot_prob=prob,
    )

  @property
  def dual_loss(self) -> jax.Array:
    """TODO."""
    return self.losses[self.it]


@jtu.register_static
@dataclasses.dataclass
class SemidiscreteSolver:
  """TODO."""
  batch_size: int
  min_iterations: int
  max_iterations: int
  optimizer: optax.GradientTransformation
  inner_iterations: int = 1000
  error_iterations: int = 1000
  error_batch_size: Optional[int] = None
  threshold: float = 1e-3
  potential_ema: float = 0.99
  callback: Optional[Callable[[SemidiscreteState], None]] = None

  # TODO(michalk8): can't directly JIT this
  def __call__(
      self,
      rng: jax.Array,
      prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
      g_init: Optional[jax.Array] = None,
  ) -> SemidiscreteOutput:
    """TODO."""

    def cond_fn(
        it: int,
        prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
        state: SemidiscreteState,
    ) -> bool:
      del prob
      loss = state.losses[it - 1]
      err = jnp.abs(state.errors[it // self.inner_iterations - 1])

      not_converged = err > self.threshold
      not_diverged = jnp.isfinite(loss)
      # continue if not converged and not diverged
      return jnp.logical_or(
          it == 0, jnp.logical_and(not_converged, not_diverged)
      )

    def body_fn(
        it: int,
        prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
        state: SemidiscreteState,
        compute_error: bool,
    ) -> SemidiscreteState:
      rng_it = jr.fold_in(rng, it)

      lin_prob = prob.sample(rng_it, self.batch_size)
      g_old = state.g

      loss, grads = jax.value_and_grad(_semidiscrete_loss)(
          g_old, lin_prob, prob.geom.is_entropy_regularized
      )
      losses = state.losses.at[it].set(loss)

      updates, opt_state = self.optimizer.update(grads, state.opt_state, g_old)
      g_new = optax.apply_updates(g_old, updates)
      g_ema = optax.incremental_update(g_new, state.g_ema, self.potential_ema)

      # fmt: off
      error = jax.lax.cond(
          compute_error,
          lambda: _marginal_chi2_error(
              # use same rng to evaluate the errors
              rng_chi2, g_ema, prob,
              num_iters=self.error_iterations,
              batch_size=self.error_batch_size or self.batch_size,
          ),
          lambda: jnp.array(jnp.inf, dtype=dtype),
      )
      # fmt: on
      errors = state.errors.at[it // self.inner_iterations].set(error)

      state = SemidiscreteState(
          it=it,
          g=g_new,
          g_ema=g_ema,
          opt_state=opt_state,
          losses=losses,
          errors=errors,
      )

      if self.callback is not None:
        jax.debug.callback(self.callback, state)

      return state

    _, m = prob.geom.shape
    dtype = prob.geom.dtype

    if g_init is None:
      g_init = jnp.zeros(m, dtype=dtype)
    else:
      assert g_init.shape == (m,), (g_init.shape, (m,))

    state = SemidiscreteState(
        it=jnp.array(0),
        g=g_init,
        g_ema=g_init,
        opt_state=self.optimizer.init(g_init),
        losses=jnp.full((self.max_iterations,), fill_value=jnp.inf,
                        dtype=dtype),
        errors=jnp.full((self.max_iterations // self.inner_iterations),
                        fill_value=jnp.inf,
                        dtype=dtype),
    )

    rng, rng_chi2 = jr.split(rng, 2)
    state: SemidiscreteState = fixed_point_loop.fixpoint_iter(
        cond_fn,
        body_fn,
        min_iterations=self.min_iterations,
        max_iterations=self.max_iterations,
        inner_iterations=self.inner_iterations,
        constants=prob,
        state=state,
    )

    below_thr = state.errors[state.it // self.inner_iterations
                            ] <= self.threshold
    finite_loss = jnp.isfinite(state.losses[state.it])

    return SemidiscreteOutput(
        it=state.it,
        g=state.g_ema,
        prob=prob,
        losses=state.losses,
        errors=state.errors,
        converged=jnp.logical_and(below_thr, finite_loss),
    )


def _soft_c_transform(
    g: jax.Array, prob: linear_problem.LinearProblem
) -> Tuple[jax.Array, jax.Array]:
  cost = prob.geom.cost_matrix
  epsilon = prob.geom.epsilon
  z = (g[None, :] - cost) / epsilon
  return -epsilon * math_utils.logsumexp(z, b=prob.b, axis=-1), z


def _hard_c_transform(
    g: jax.Array, prob: linear_problem.LinearProblem
) -> Tuple[jax.Array, jax.Array]:
  cost = prob.geom.cost_matrix
  z = g[None, :] - cost
  return -jnp.max(z, axis=-1), z


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _semidiscrete_loss(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
    is_soft: bool,
) -> jax.Array:
  f = _soft_c_transform(g, prob) if is_soft else _hard_c_transform(g, prob)
  return -jnp.mean(f) - jnp.dot(g, prob.b)


def _semidiscrete_loss_fwd(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
    is_soft: bool,
) -> Tuple[jax.Array, Tuple[jax.Array, linear_problem.LinearProblem]]:
  f, z = _soft_c_transform(g, prob) if is_soft else _hard_c_transform(g, prob)
  return -jnp.mean(f) - jnp.dot(g, prob.b), (z, prob)


def _semidiscrete_loss_bwd(
    is_soft: bool,
    res: jax.Array,
    g: jax.Array,
) -> Tuple[jax.Array, None]:
  z, prob = res
  n, _ = prob.geom.shape
  grad = jsp.special.softmax(z, axis=-1) if is_soft else _hardmax(z)
  grad = grad.sum(0) * (1.0 / n) - prob.b
  return g * grad, None


# TODO(michalk8):
def _hardmax(z: jax.Array) -> jax.Array:
  max_val = jnp.max(z, axis=-1, keepdims=True)
  is_max = jnp.abs(z - max_val) <= 1e-8
  num_max = jnp.sum(is_max, axis=-1, keepdims=True)
  return is_max / num_max


_semidiscrete_loss.defvjp(_semidiscrete_loss_fwd, _semidiscrete_loss_bwd)


def _marginal_chi2_error(
    rng: jax.Array,
    g: jax.Array,
    prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
    *,
    num_iters: int,
    batch_size: int,
) -> jax.Array:

  def body(chi2_err_avg: jax.Array, it: jax.Array) -> Tuple[jax.Array, None]:
    rng_it = jr.fold_in(rng, it)
    matrix = out.matrix(rng_it, batch_size)

    p2 = m * (batch_size * batch_size) * (matrix @ matrix.T)
    chi2 = (p2.sum() - p2.trace()) / (batch_size * (batch_size - 1)) - 1.0

    chi2_err_avg = chi2_err_avg + chi2 / num_iters
    return chi2_err_avg, None

  out = SemidiscreteOutput(
      it=-1, g=g, prob=prob, losses=None, errors=None, converged=False
  )
  _, m = prob.geom.shape

  chi2_err = jnp.zeros(())
  chi2_err, _ = jax.lax.scan(body, init=chi2_err, xs=jnp.arange(num_iters))
  return chi2_err
