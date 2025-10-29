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
  """MLP velocity field.

  Args:
    dim: Dimensionality of the velocity field.
    hidden_dims: Hidden dimensions.
    cond_dim: Dimensionality of the condition vector.
    act_fn: Activation function.
    dropout_rate: Dropout rate.
    rngs: Random number generator used for initialization.
    kwargs: Keyword arguments for :class:`~flax.nnx.Linear`.
  """

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
    super().__init__()
    if time_enc_num_freqs is None:
      time_enc_num_freqs = max(1, min(dim // 16, 64))

    time_dim = 2 * time_enc_num_freqs
    hidden_dims = [dim + time_dim + cond_dim] + list(hidden_dims)
    block_fn = functools.partial(
        Block, act_fn=act_fn, dropout_rate=dropout_rate, **kwargs
    )

    blocks = [
        *(
            block_fn(in_dim, out_dim, rngs=rngs)
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:])
        )
    ]
    final_block = block_fn(
        hidden_dims[-1],
        dim,
        act_fn=jax.nn.identity,
        rngs=rngs,
    )

    self.blocks = nnx.Sequential(*blocks, final_block)
    self.time_enc_num_freqs = time_enc_num_freqs

  def __call__(
      self,
      t: jax.Array,
      x: jax.Array,
      cond: Optional[jax.Array] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ) -> jax.Array:
    """Compute the velocity.

    Args:
      t: Time array of shape ``[batch,]``.
      x: Input array of shape ``[batch, dim]``.
      cond: Condition array of shape ``[batch, cond_dim]``.
      rngs: Random number generator for dropout.

    Returns:
      The velocity array of shape ``[batch, dim]``.
    """
    t_emb = _encode_time(t, self.time_enc_num_freqs)
    h = [t_emb, x] if cond is None else [t_emb, x, cond]
    h = jnp.concatenate(h, axis=-1)
    return self.blocks(h, rngs=rngs)


class Block(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      *,
      act_fn: Callable[[jax.Array], jax.Array],
      dropout_rate: float = 0.0,
      rngs: nnx.Rngs,
      **kwargs: Any
  ):
    super().__init__()
    self.lin = nnx.Linear(in_dim, out_dim, rngs=rngs, **kwargs)
    self.act_fn = act_fn
    self.dropout = nnx.Dropout(dropout_rate)

  def __call__(
      self, x: jax.Array, *, rngs: Optional[nnx.Rngs] = None
  ) -> jax.Array:
    return self.dropout(self.act_fn(self.lin(x)), rngs=rngs)


def _encode_time(t: jax.Array, num_freqs: int) -> jax.Array:
  assert num_freqs > 0, "Number of frequencies must be positive."
  assert t.ndim == 1, t.shape
  freq = (2 * jnp.pi) * jnp.arange(1, num_freqs + 1)
  t = freq * t[:, None]
  return jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=-1)
