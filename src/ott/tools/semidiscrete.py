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
import functools
from typing import Any, Callable, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

import optax

from ott.geometry import costs
from ott.math import utils as math_utils

__all__ = ["semidiscrete"]

Callback = Callable[[jax.Array, int, jax.Array, jax.Array, jax.Array], None]


def _min_operator(
    g: jax.Array,
    x: jax.Array,
    y: jax.Array,
    *,
    epsilon: Optional[float],
    cost_fn: costs.CostFn,
) -> jax.Array:
  m = y.shape[0]
  assert g.shape == (m,), g.shape
  cost = cost_fn.all_pairs(x, y)  # [n, m]
  if epsilon is None:  # hard min
    z = g[None, :] - cost
    return -jnp.max(z, axis=-1)
  # soft min
  z = (g[None, :] - cost) / epsilon - jnp.log(m)
  return -epsilon * math_utils.logsumexp(z, axis=-1)


# TODO(michalk8): custom JVP
def _semidiscrete_loss(
    g: jax.Array,
    x: jax.Array,
    y: jax.Array,
    *,
    epsilon: float,
    cost_fn: costs.CostFn,
) -> jax.Array:
  z = _min_operator(g, x, y, epsilon=epsilon, cost_fn=cost_fn)
  return -jnp.mean(z) - jnp.mean(g)


def semidiscrete(
    rng: jax.Array,
    sampler: Callable[[jax.Array, tuple[int, ...]], jax.Array],
    y: jax.Array,
    optimizer: optax.GradientTransformation,
    g: Optional[jax.Array] = None,
    num_iters: int = 1000,
    batch_size: int = 32,
    data_sharding: Optional[jax.sharding.Sharding] = None,
    epsilon: Optional[float] = None,
    cost_fn: Optional[costs.CostFn] = None,
    callbacks: Iterable[Callback] = (),
) -> jax.Array:
  """TODO."""

  @functools.partial(
      jax.jit,
      in_shardings=(data_sharding, data_sharding, None),
      out_shardings=(data_sharding, None, None),
  )
  def update_potential(
      g: jax.Array,
      x: jax.Array,
      opt_state: Any,
  ) -> Tuple[jax.Array, Any, Tuple[jax.Array, jax.Array]]:
    loss, grads = jax.value_and_grad(_semidiscrete_loss)(
        g, x, y, epsilon=epsilon, cost_fn=cost_fn
    )
    grad_norm = jnp.linalg.norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, g)
    g = optax.apply_updates(g, updates)
    return g, opt_state, (loss, grad_norm)

  n, *rest = y.shape
  x_shape = (batch_size, *rest)
  callbacks = tuple(callbacks)

  if g is None:
    g = jnp.zeros((n,), device=data_sharding)
  else:
    assert g.shape == (n,), g.shape
    g = jax.device_put(g, data_sharding)

  if cost_fn is None:
    cost_fn = costs.SqEuclidean()

  opt_state = optimizer.init(g)
  for it in range(num_iters):
    rng, rng_dist, *rng_callbacks = jr.split(rng, 2 + len(callbacks))
    x = sampler(rng_dist, x_shape)
    g, opt_state, (loss, grad_norm) = update_potential(g, x, opt_state)

    # TODO(michalk8): don't pass grad_norm?
    # TODO(michalk8): track losses + return
    for rng_cb, callback in zip(rng_callbacks, callbacks):
      callback(rng_cb, it, g, loss, grad_norm)

  return g


# TODO(michalk8): format, etc.
def print_callback(print_every: int = 1000) -> Callback:

  def callback(
      rng: jax.Array,
      it: int,
      g: jax.Array,
      loss: jax.Array,
      grad_norm: jax.Array,
  ) -> None:
    del rng, g
    if it % print_every == 0:
      print(  # noqa: T201
          f"It. {it}, loss={loss.item():.5f}, grad_norm={grad_norm:.5f}"
      )

  return callback
