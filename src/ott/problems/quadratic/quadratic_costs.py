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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["make_square_loss", "make_kl_loss"]


class Loss(NamedTuple):  # noqa: D101
  func: Callable[[jnp.ndarray], jnp.ndarray]
  is_linear: bool


class GWLoss(NamedTuple):  # noqa: D101
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
