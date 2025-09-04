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


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteState:
  """TODO."""
  it: jax.Array
  g: jax.Array
  g_avg: Optional[jax.Array]
  opt_state: Any
  losses: jax.Array
  errors: jax.Array

  def to_output(  # noqa: D102
      self, prob: semidiscrete_linear_problem.SemidiscreteLinearProblem
  ) -> "SemidiscreteOutput":
    g = self.g if self.g_avg is None else self.g_avg
    # TODO(michalk8): convergence
    return SemidiscreteOutput(
        g=g,
        prob=prob,
        losses=self.losses,
        errors=self.errors,
    )


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteOutput:
  """TODO."""
  g: jax.Array
  prob: semidiscrete_linear_problem.SemidiscreteLinearProblem
  losses: jax.Array
  errors: jax.Array

  def matrix(
      self,
      rng: jax.Array,
      num_samples: int,
  ) -> jax.Array:
    """TODO."""
    if self.is_entropy_regularized:
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

  @property
  def is_entropy_regularized(self) -> bool:
    """TODO."""
    return self.prob.geom.epsilon != 0.0

  def materialize(
      self, rng: jax.Array, num_samples: int
  ) -> sinkhorn.SinkhornOutput:
    """TODO."""
    epsilon = self.prob.geom.epsilon
    prob = self.prob.materialize(rng, num_samples)

    f, _ = _c_transform(self.g, prob)
    # TODO(michalk8): comment on why
    f_tilde = f + epsilon * jnp.log(1.0 / num_samples)
    g_tilde = self.g + epsilon * jnp.log(self.prob.b)

    # TODO(michalk8): enable this?
    # return out.set_cost(prob, lse_mode=True, use_danskin=True)
    return sinkhorn.SinkhornOutput(
        potentials=(f_tilde, g_tilde),
        ot_prob=prob,
    )


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteSolver:
  """TODO."""
  batch_size: int = dataclasses.field(metadata={"static": True})
  min_iterations: int = dataclasses.field(metadata={"static": True})
  max_iterations: int = dataclasses.field(metadata={"static": True})
  optimizer: optax.GradientTransformation = dataclasses.field(
      metadata={"static": True}
  )
  inner_iterations: Optional[int] = dataclasses.field(
      default=1000, metadata={"static": True}
  )
  threshold: float = dataclasses.field(default=1e-3, metadata={"static": True})
  warmup_iterations: Optional[int] = dataclasses.field(
      default=None, metadata={"static": True}
  )
  callback: Callable[[jax.Array, SemidiscreteState], None] = dataclasses.field(
      default=None, metadata={"static": True}
  )

  # TODO(michalk8): can't directly JIT this
  def __call__(
      self,
      rng: jax.Array,
      prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
      g_init: Optional[jax.Array] = None,
  ) -> SemidiscreteOutput:
    """TODO."""

    def cond_fn(
        it: jax.Array,
        prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
        state: SemidiscreteState,
    ) -> jax.Array:
      # TODO
      del prob
      loss = state.losses[it - 1]
      err = jnp.abs(state.errors[it // self.max_iterations])
      not_converged = err > self.threshold
      not_diverged = jnp.isfinite(loss)
      jax.debug.print("{} {} {}", it, loss, err)
      return jnp.logical_or(
          it == 0, jnp.logical_and(not_converged, not_diverged)
      )

    def body_fn(
        it: jax.Array,
        prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
        state: SemidiscreteState,
        compute_error: jax.Array,
    ) -> SemidiscreteState:
      rng_it = jr.fold_in(rng, it)
      rng_mat, rng_err, rng_cb = jr.split(rng_it, 3)
      lin_prob = prob.materialize(rng_mat, self.batch_size)
      g = state.g

      loss, grads = jax.value_and_grad(_semidiscrete_loss)(g, lin_prob)

      updates, opt_state = self.optimizer.update(grads, state.opt_state, g)
      g = optax.apply_updates(g, updates)

      # fmt: off
      if self.warmup_iterations is None:
        g_avg, g_err = None, g
      else:
        w = it - self.warmup_iterations
        g_avg = jax.lax.cond(
            it < self.warmup_iterations,
            (lambda g, g_avg: g),
            (lambda g, g_avg: (1.0 / (w + 1)) * g + (w / (w + 1)) * g_avg),
            g, state.g_avg
        )
        g_err = g_avg

      error = jax.lax.cond(
          compute_error,
          lambda: _marginal_chi2_error(
              rng_err, g_err, prob, num_iters=100, batch_size=self.batch_size
          ),
          lambda: jnp.array(jnp.inf, dtype=g.dtype),
      )
      # fmt: on
      errors = state.errors.at[it // self.inner_iterations].set(error)
      losses = state.losses.at[it].set(loss)

      state = SemidiscreteState(
          it=it,
          g=g,
          g_avg=g_avg,
          opt_state=opt_state,
          losses=losses,
          errors=errors,
      )

      if self.callback is not None:
        jax.debug.callback(self.callback, rng_it, state)

      return state

    use_averaging = self.warmup_iterations is not None
    _, m = prob.geom.shape
    if g_init is None:
      g_init = jnp.zeros(m)
    else:
      assert g_init.shape == (m,), (g_init.shape, (m,))

    state = SemidiscreteState(
        it=jnp.array(0),
        g=g_init,
        g_avg=g_init if use_averaging else None,
        opt_state=self.optimizer.init(g_init),
        # TODO(michalk8): default values?
        losses=jnp.full((self.max_iterations,), fill_value=jnp.inf),
        errors=jnp.full((self.max_iterations // self.inner_iterations),
                        fill_value=jnp.inf),
    )

    state = fixed_point_loop.fixpoint_iter(
        cond_fn,
        body_fn,
        min_iterations=self.min_iterations,
        max_iterations=self.max_iterations,
        inner_iterations=self.inner_iterations,
        constants=prob,
        state=state,
    )

    return state.to_output(prob)


def _c_transform(
    g: jax.Array, prob: linear_problem.LinearProblem
) -> Tuple[jax.Array, jax.Array]:
  n, m = prob.geom.shape
  assert g.shape == (m,), (g.shape, (m,))
  cost = prob.geom.cost_matrix
  epsilon = prob.geom.epsilon

  if epsilon == 0.0:
    # hard min
    z = g[None, :] - cost
    return -jnp.max(z, axis=-1), z
  # soft min
  z = (g[None, :] - cost) / epsilon
  return -epsilon * math_utils.logsumexp(z, b=prob.b, axis=-1), z


@jax.custom_vjp
def _semidiscrete_loss(
    g: jax.Array, prob: linear_problem.LinearProblem
) -> jax.Array:
  f, _ = _c_transform(g, prob)
  return -jnp.mean(f) - jnp.dot(g, prob.b)


def _semidiscrete_loss_fwd(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
) -> Tuple[jax.Array, Tuple[jax.Array, linear_problem.LinearProblem]]:
  f, tmp = _c_transform(g, prob)
  return -jnp.mean(f) - jnp.dot(g, prob.b), (tmp, prob)


def _semidiscrete_loss_bwd(
    res: Tuple[jax.Array, linear_problem.LinearProblem], g: jax.Array
) -> Tuple[jax.Array, None]:
  z, prob = res
  n, _ = prob.geom.shape
  if prob.geom.epsilon != 0.0:
    grad = jsp.special.softmax(z, axis=-1)
  else:
    grad = _hardmax(z)
  grad = grad.sum(0) * (1.0 / n) - prob.b
  return g * grad, None


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
    matrix = SemidiscreteOutput(g, prob, None, None).matrix(rng_it, batch_size)
    m = matrix.shape[1]

    p2 = m * (batch_size ** 2) * (matrix @ matrix.T)
    chi2 = (p2.sum() - p2.trace()) / (batch_size * (batch_size - 1)) - 1.0

    chi2_err_avg = chi2_err_avg + chi2 / num_iters
    return chi2_err_avg, None

  chi2_err = jnp.zeros(())
  chi2_err, _ = jax.lax.scan(body, init=chi2_err, xs=jnp.arange(num_iters))
  return chi2_err
