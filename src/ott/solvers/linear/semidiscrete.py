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

Stats = Dict[Literal["loss", "grad_norm"], float]


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteState:
  """TODO."""
  g: jax.Array
  opt_state: Any


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteOutput:
  """TODO."""
  g: jax.Array
  prob: semidiscrete_linear_problem.SemidiscreteLinearProblem


@jtu.register_dataclass
@dataclasses.dataclass
class SemidiscreteSolver:
  """TODO."""
  batch_size: int = dataclasses.field(metadata={"static": True})
  num_iters: int = dataclasses.field(metadata={"static": True})
  optimizer: optax.GradientTransformation = dataclasses.field(
      metadata={"static": True}
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

      state = SemidiscreteState(g=g, opt_state=opt_state)
      return state, {"loss": loss, "grad_norm": grad_norm}

    _, m = prob.geom.shape
    if g_init is None:
      g_init = jnp.zeros(m)
    else:
      assert g_init.shape == (m,), (g_init.shape, (m,))

    state = SemidiscreteState(g=g_init, opt_state=self.optimizer.init(g_init))
    state, stats = jax.lax.scan(step, init=state, xs=jnp.arange(self.num_iters))
    out = SemidiscreteOutput(g=state.g, prob=prob)

    return out, stats


def _c_transform(
    g: jax.Array, prob: linear_problem.LinearProblem
) -> Tuple[jax.Array, jax.Array]:
  _, m = prob.geom.shape
  assert g.shape == (m,), (g.shape, (m,))
  cost = prob.geom.cost_matrix
  epsilon = prob.geom.epsilon

  if epsilon is None:
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
  z, _ = _c_transform(g, prob)
  return -jnp.mean(z) - jnp.dot(g, prob.b)


def _semidiscrete_loss_fwd(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
) -> Tuple[jax.Array, Tuple[jax.Array, linear_problem.LinearProblem]]:
  z, tmp = _c_transform(g, prob)
  return -jnp.mean(z) - jnp.dot(g, prob.b), (tmp, prob)


def _semidiscrete_loss_bwd(
    res: Tuple[jax.Array, linear_problem.LinearProblem], g: jax.Array
) -> Tuple[jax.Array, None]:
  z, prob = res
  n, _ = prob.geom.shape
  if True:  # TODO(michalk8):
    grad = jsp.special.softmax(z, axis=-1).sum(0)
  else:
    max_val = jnp.max(z, axis=-1, keepdims=True)
    is_max = jnp.abs(z - max_val) <= 1e-8
    num_max = jnp.sum(is_max, axis=-1, keepdims=True)
    grad = jnp.sum(is_max / num_max, axis=0)
  # TODO(michalk8): check
  grad = grad * (1.0 / n) - prob.b
  return g * grad, None


_semidiscrete_loss.defvjp(_semidiscrete_loss_fwd, _semidiscrete_loss_bwd)
