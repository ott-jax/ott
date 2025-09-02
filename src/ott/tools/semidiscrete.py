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
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

import optax

from ott.geometry import costs
from ott.math import utils as math_utils

__all__ = ["semidiscrete"]

Stats = Dict[Literal["loss", "grad_norm"], float]
Callback = Callable[[jax.Array, int, jax.Array, Stats], None]


def _min_operator(
    g: jax.Array,
    x: jax.Array,
    y: jax.Array,
    epsilon: Optional[float],
    cost_fn: costs.CostFn,
) -> Tuple[jax.Array, jax.Array]:
  m = y.shape[0]
  assert g.shape == (m,), (g.shape, y.shape)
  cost = cost_fn.all_pairs(x, y)  # [n, m]
  if epsilon is None:  # hard min
    z = g[None, :] - cost
    return -jnp.max(z, axis=-1), z
  # soft min
  z = (g[None, :] - cost) / epsilon - jnp.log(m)
  return -epsilon * math_utils.logsumexp(z, axis=-1), z


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _semidiscrete_loss(
    g: jax.Array,
    x: jax.Array,
    y: jax.Array,
    epsilon: float,
    cost_fn: costs.CostFn,
) -> jax.Array:
  z, _ = _min_operator(g, x, y, epsilon=epsilon, cost_fn=cost_fn)
  return -jnp.mean(z) - jnp.mean(g)


def _semidiscrete_loss_fwd(
    g: jax.Array,
    x: jax.Array,
    y: jax.Array,
    epsilon: float,
    cost_fn: costs.CostFn,
) -> Tuple[jax.Array, Tuple[jax.Array, int, int]]:
  z, tmp = _min_operator(g, x, y, epsilon=epsilon, cost_fn=cost_fn)
  save = (tmp, x.shape[0], y.shape[0])
  return -jnp.mean(z) - jnp.mean(g), save


def _semidiscrete_loss_bwd(
    epsilon: Optional[float], _, res: Tuple[jax.Array, int, int], g: jax.Array
) -> Tuple[jax.Array, None, None]:
  (z, n, m) = res
  if epsilon is not None:
    grad = jsp.special.softmax(z, axis=-1).sum(0)
  else:
    max_val = jnp.max(z, axis=-1, keepdims=True)
    is_max = jnp.abs(z - max_val) <= 1e-8
    num_max = jnp.sum(is_max, axis=-1, keepdims=True)
    grad = jnp.sum(is_max / num_max, axis=0)
  grad = grad * (1.0 / n) - (1.0 / m)
  return g * grad, None, None


_semidiscrete_loss.defvjp(_semidiscrete_loss_fwd, _semidiscrete_loss_bwd)


def semidiscrete(
    rng: jax.Array,
    sampler: Callable[[jax.Array, tuple[int, ...]], jax.Array],
    y: jax.Array,
    optimizer: optax.GradientTransformation,
    opt_state: Optional[Any] = None,
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
      in_shardings=(data_sharding, None, None),
      out_shardings=(data_sharding, None, None),
      donate_argnums=(0, 1),
  )
  def update_potential(
      g: jax.Array,
      opt_state: Any,
      x: jax.Array,
  ) -> Tuple[jax.Array, Any, Dict[str, jax.Array]]:
    loss, grads = jax.value_and_grad(_semidiscrete_loss)(
        g, x, y, epsilon=epsilon, cost_fn=cost_fn
    )
    grad_norm = jnp.linalg.norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, g)
    g = optax.apply_updates(g, updates)
    return g, opt_state, {"loss": loss, "grad_norm": grad_norm}

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
  if opt_state is None:
    opt_state = optimizer.init(g)

  for it in range(num_iters):
    rng, rng_dist, *rng_callbacks = jr.split(rng, 2 + len(callbacks))
    # TODO(michalk8): sharding?
    x = sampler(rng_dist, x_shape)

    g, opt_state, stats = update_potential(g, opt_state, x)
    stats = jax.tree.map(lambda x: x.item() if jnp.isscalar(x) else x, stats)

    # TODO(michalk8): track losses + return
    for rng_cb, callback in zip(rng_callbacks, callbacks):
      callback(rng_cb, it, g, stats)

  return g


# TODO(michalk8): add marginal TV/chi-squared error
# TODO(michalk8): format, etc.
def print_callback(print_every: int = 1000) -> Callback:

  def callback(
      rng: jax.Array,
      it: int,
      g: jax.Array,
      stats: Stats,
  ) -> None:
    del rng, g
    if it % print_every == 0:
      loss, grad_norm = stats["loss"], stats["grad_norm"]
      print(  # noqa: T201
          f"It. {it}, loss={loss:.5f}, grad_norm={grad_norm:.5f}"
      )

  return callback
