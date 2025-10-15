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
from typing import Any, Callable, Optional, Sequence

import jax
from jax import numpy as jnp

from flax import nnx

__all__ = ["MLP"]


class MLP(nnx.Module):
  """TODO."""

  def __init__(
      self,
      dim: int,
      *,
      hidden_dims: Sequence[int] = (),
      cond_dim: int = 0,
      act_fn: Callable[[jax.Array], jax.Array] = nnx.silu,
      time_enc_num_freqs: Optional[int] = None,
      dropout_rate: float = 0.0,
      rngs: nnx.Rngs,
      **kwargs: Any,
  ):
    if time_enc_num_freqs is None:
      time_enc_num_freqs = max(1, min(dim // 16, 64))

    time_dim = 2 * time_enc_num_freqs
    hidden_dims = [dim + time_dim + cond_dim] + list(hidden_dims) + [dim]
    block_fn = functools.partial(
        Block, act_fn=act_fn, dropout_rate=dropout_rate, **kwargs
    )
    self.blocks = nnx.Sequential(
        *(
            block_fn(in_dim, out_dim, rngs=rngs)
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:])
        )
    )
    self.time_enc_num_freqs = time_enc_num_freqs

  def __call__(
      self,
      t: jax.Array,
      x: jax.Array,
      *,
      y: Optional[jax.Array] = None,
      rngs: nnx.Rngs | None = None,
  ) -> jax.Array:
    """TODO."""
    t_emb = _encode_time(t, self.time_enc_num_freqs)
    h = [t_emb, x] if y is None else [t_emb, x, y]
    h = jnp.concatenate(h, axis=-1)
    return self.blocks(h)


class Block(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      *,
      act_fn,
      dropout_rate: float = 0.0,
      rngs: nnx.Rngs,
      **kwargs: Any
  ):
    self.lin = nnx.Linear(in_dim, out_dim, rngs=rngs, **kwargs)
    self.act_fn = act_fn
    self.dropout = nnx.Dropout(dropout_rate)

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.dropout(self.act_fn(self.lin(x)))


def _encode_time(t: jax.Array, num_freqs: int) -> jax.Array:
  freq = 2 * (1 + jnp.arange(num_freqs)) * jnp.pi
  t = freq * t[:, None]
  return jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=-1)
