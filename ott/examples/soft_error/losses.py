# coding=utf-8
"""Defines classification losses."""

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
from ott.tools import soft_sort


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray):
  logits = nn.log_softmax(logits)
  return -jnp.sum(labels * logits) / labels.shape[0]


def soft_error_loss(
    logits: jnp.ndarray, labels: jnp.ndarray, epsilon: float = 1e-2):
  """Average distance between the top rank and the rank of the true class."""
  ranks_fn = functools.partial(soft_sort.ranks, axis=-1, epsilon=epsilon)
  ranks_fn = jax.jit(ranks_fn)
  soft_ranks = ranks_fn(logits)
  return jnp.mean(nn.relu(
      labels.shape[-1] - 1 - jnp.sum(labels * soft_ranks, axis=1)))


def get(name: str = 'cross_entropy'):
  """Returns the loss function corresponding to the input name."""
  losses = {
      'soft_error': soft_error_loss,
      'cross_entropy': cross_entropy_loss
  }
  result = losses.get(name, None)
  if result is None:
    raise ValueError(
        f'Unknown loss {name}. Possible values: {",".join(losses)}')
  return result
