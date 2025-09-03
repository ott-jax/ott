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
from typing import Any, Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu

import optax

from ott.math import utils as math_utils
from ott.problems.linear import linear_problem, semidiscrete_linear_problem
from ott.solvers.linear import sinkhorn

Stats = Dict[Literal["loss", "grad_norm"], float]


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteState:
  """TODO."""
  g: jax.Array
  g_avg: Optional[jax.Array]
  opt_state: Any


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteOutput:
  """TODO."""
  g: jax.Array
  prob: semidiscrete_linear_problem.SemidiscreteLinearProblem

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
  ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """TODO."""

    def body(carry: Dict[str, jax.Array],
             it: jax.Array) -> Tuple[Dict[str, jax.Array], None]:
      rng_it = jr.fold_in(rng, it)
      matrix = self.matrix(rng_it, batch_size)

      # TODO(michalk8): check (esp. the -1)
      p2 = m * (matrix @ matrix.T)
      chi2 = (p2.sum() - p2.trace()) / (batch_size * (batch_size - 1)) - 1.0

      marginal_b = matrix.sum(0)

      perp = jnp.exp(-jsp.special.xlogy(matrix, matrix).sum(-1)).mean()

      carry = jax.tree.map(
          lambda x, y: x + y / num_iters,
          carry,
          {
              "chi2": chi2,
              "perp": perp,
              "marginal_b": marginal_b
          },
      )
      return carry, None

    _, m = self.prob.geom.shape
    init = {
        "chi2": jnp.zeros(()),
        "perp": jnp.zeros(()),
        "marginal_b": jnp.zeros((m,)),
    }

    res, _ = jax.lax.scan(body, init=init, xs=jnp.arange(num_iters))
    marginal_err = jnp.linalg.norm(res["marginal_b"] - self.prob.b, ord=jnp.inf)
    return res["chi2"], {"perp": res["perp"], "tv": marginal_err}

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
  num_iters: int = dataclasses.field(metadata={"static": True})
  optimizer: optax.GradientTransformation = dataclasses.field(
      metadata={"static": True}
  )
  warmup_iters: Optional[int] = dataclasses.field(
      default=None, metadata={"static": True}
  )

  def __call__(
      self,
      rng: jax.Array,
      prob: semidiscrete_linear_problem.SemidiscreteLinearProblem,
      g_init: Optional[jax.Array] = None,
  ) -> Tuple[SemidiscreteOutput, Stats]:
    """TODO."""

    def step(state: SemidiscreteState,
             it: jax.Array) -> Tuple[SemidiscreteState, Stats]:
      rng_it = jr.fold_in(rng, it)
      lin_prob = prob.materialize(rng_it, self.batch_size)
      g = state.g

      loss, grads = jax.value_and_grad(_semidiscrete_loss)(g, lin_prob)

      grad_norm = jnp.linalg.norm(grads)
      updates, opt_state = self.optimizer.update(grads, state.opt_state, g)
      g = optax.apply_updates(g, updates)

      if self.warmup_iters is None:
        g_avg = None
      else:
        w = it - self.warmup_iters
        g_avg = jax.lax.cond(
            it < self.warmup_iters, lambda g, g_avg: g, lambda g, g_avg:
            (1.0 / (w + 1)) * g + (w / (w + 1)) * g_avg, g, state.g_avg
        )

      state = SemidiscreteState(g=g, g_avg=g_avg, opt_state=opt_state)
      return state, {"loss": loss, "grad_norm": grad_norm}

    use_averaging = self.warmup_iters is not None
    _, m = prob.geom.shape
    if g_init is None:
      g_init = jnp.zeros(m)
    else:
      assert g_init.shape == (m,), (g_init.shape, (m,))

    state = SemidiscreteState(
        g=g_init,
        g_avg=g_init if use_averaging else None,
        opt_state=self.optimizer.init(g_init)
    )
    state, stats = jax.lax.scan(step, init=state, xs=jnp.arange(self.num_iters))

    g = state.g_avg if use_averaging else state.g
    out = SemidiscreteOutput(g=g, prob=prob)

    return out, stats


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
