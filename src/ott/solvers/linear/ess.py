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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu


@jtu.register_dataclass
@dataclasses.dataclass(repr=False, frozen=True)
class ElipticalSliceSampler:
  """TODO."""
  A: jax.Array
  b: jax.Array
  x: Optional[jax.Array] = None
  num_burnin: int = dataclasses.field(default=100, metadata={"static": True})

  # TODO(mcihalk8): change API

  def burn_in(
      self,
      rng: jax.Array,
      x: Optional[jax.Array] = None,
  ) -> "ElipticalSliceSampler":
    """TODO."""
    if x is None:
      raise NotImplementedError("TODO: find initial feasible point.")
    state = ElipticalSliceSampler(x=x, A=self.A, b=self.b)
    return jax.lax.fori_loop(
        0, self.num_burnin, lambda it, s: s(jr.fold_in(rng, it)), state
    )

  def __call__(self, rng: jax.Array) -> "ElipticalSliceSampler":
    """TODO."""
    assert self.x.ndim == 2, self.x.shape
    rng_nu, rng_angles = jr.split(rng, 2)

    nu = jr.normal(rng_nu, self.x.shape, dtype=self.A.dtype)
    # [B, M]
    p, q = self.x @ self.A.T, nu @ self.A.T
    bias = self.b[None, :]  # [1, m]

    alpha, beta = _intersection_angles(p, q, bias)
    left, right = _active_angles(alpha, beta)
    # [B, 1]
    theta = _draw_angles(rng_angles, left, right)[:, None]

    candidate = self.x * jnp.cos(theta) + jnp.sin(theta)  # (B, ...)
    candidate = jnp.where(self._is_feasible(candidate), candidate, self.x)
    return ElipticalSliceSampler(
        x=candidate,
        A=self.A,
        b=self.b,
    )

  def _is_feasible(self, x: jax.Array) -> jax.Array:
    return ((x @ self.A.T - self.b) < 0.0).all(keepdims=True, axis=-1)


def _intersection_angles(p: jax.Array, q: jax.Array,
                         bias: jax.Array) -> Tuple[jax.Array, jax.Array]:
  radius = jnp.sqrt(p ** 2 + q ** 2)
  ratio = bias / radius

  has_solution = ratio < 1.0
  arccos = jnp.arccos(ratio)
  arccos = jnp.where(has_solution, arccos, 0.0)
  arctan = jnp.arctan2(q, p)

  theta1 = arctan + arccos
  theta2 = arctan - arccos
  # translate every angle to [0, 2 * pi]
  theta1 = theta1 + (theta1 < 0.0) * (2.0 * jnp.pi)
  theta2 = theta2 + (theta2 < 0.0) * (2.0 * jnp.pi)

  alpha = jnp.minimum(theta1, theta2)
  beta = jnp.maximum(theta1, theta2)

  return alpha, beta


def _active_angles(alpha: jax.Array,
                   beta: jax.Array) -> Tuple[jax.Array, jax.Array]:
  b, *_ = alpha.shape
  batch_ixs = jnp.arange(b).reshape(-1, 1)
  indices = jnp.argsort(alpha, descending=False, axis=-1)

  left = jax.lax.cummax(beta[batch_ixs, indices], axis=1)
  left = jnp.concatenate([jnp.zeros((b, 1), dtype=left.dtype), left], axis=-1)

  right = alpha[batch_ixs, indices]
  right = jnp.concatenate([
      right,
      jnp.full((b, 1), fill_value=2.0 * jnp.pi, dtype=right.dtype)
  ],
                          axis=-1)

  gap = jnp.clip(right - left, min=0.0)
  delta = jnp.clip(gap * 0.25, max=1e-6)
  return left + delta, right - delta


def _draw_angles(
    rng: jax.Array, left: jax.Array, right: jax.Array
) -> jax.Array:
  b, *_ = left.shape
  batch_ixs = jnp.arange(b)
  csum = jnp.clip(right - left, min=0.0).cumsum(axis=-1)

  u = csum[:, -1] * jr.uniform(rng, (b,), dtype=csum.dtype)
  ixs = jax.vmap(jnp.searchsorted)(csum, u)

  padded_csum = jnp.concatenate([jnp.zeros((b, 1), dtype=csum.dtype), csum],
                                axis=-1)
  # [B,]
  return u - padded_csum[batch_ixs, ixs] + left[batch_ixs, ixs]
