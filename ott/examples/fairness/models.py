# coding=utf-8
"""A model for to embed structured features."""

from typing import Any, Tuple

import flax.linen as nn
import jax.numpy as jnp


class FeaturesEncoder(nn.Module):
  """Encodes structured features."""
  input_dims: Tuple[int]
  embed_dim: int = 32

  @nn.compact
  def __call__(self, x):
    result = []
    index = 0
    for d in self.input_dims:
      arr = x[..., index:index+d]
      result.append(arr if d == 1 else nn.Dense(self.embed_dim)(arr))
      index += d
    return jnp.concatenate(result, axis=-1)


class AdultModel(nn.Module):
  """A model to predict if the income is above 50k (adult dataset)."""
  encoder_cls: Any
  hidden: Tuple[int] = (64, 64)

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = self.encoder_cls()(x)
    for h in self.hidden:
      x = nn.Dense(h)(x)
      x = nn.relu(x)
    x = nn.Dense(1)(x)
    x = nn.sigmoid(x)
    return x[..., 0]
