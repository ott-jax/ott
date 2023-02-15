from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["make_square_loss", "make_kl_loss"]


class Loss(NamedTuple):
  func: Callable[[jnp.ndarray], jnp.ndarray]
  is_linear: bool


class GWLoss(NamedTuple):
  f1: Loss
  f2: Loss
  h1: Loss
  h2: Loss


def make_square_loss() -> GWLoss:  # noqa: D103
  f1 = Loss(lambda x: x ** 2, is_linear=False)
  f2 = Loss(lambda y: y ** 2, is_linear=False)
  h1 = Loss(lambda x: x, is_linear=True)
  h2 = Loss(lambda y: 2.0 * y, is_linear=True)
  return GWLoss(f1, f2, h1, h2)


def make_kl_loss(clipping_value: float = 1e-8) -> GWLoss:  # noqa: D103
  f1 = Loss(lambda x: -jax.scipy.special.entr(x) - x, is_linear=False)
  f2 = Loss(lambda y: y, is_linear=True)
  h1 = Loss(lambda x: x, is_linear=True)
  h2 = Loss(lambda y: jnp.log(jnp.clip(y, clipping_value)), is_linear=False)
  return GWLoss(f1, f2, h1, h2)
