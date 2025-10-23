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
import jax
import jax.numpy as jnp

from flax import nnx

__all__ = ["EMA", "init_ema", "update_ema"]


class EMA(nnx.Module):
  """Exponential moving average (EMA) of a model.

  Args:
    model: Model to average.
    decay: EMA decay factor.
  """

  def __init__(self, model: nnx.Module, *, decay: float):
    super().__init__()
    self.ema = init_ema(model)
    self.decay = decay

  def __call__(self, model: nnx.Module) -> None:
    """Update the EMA.

    Args:
      model: Model to average.

    Returns:
      Nothing, just updates the EMA in-place.
    """
    update_ema(model, ema=self.ema, decay=self.decay)


def init_ema(model: nnx.Module) -> nnx.Module:
  """Create initial exponential moving average (EMA) state.

  Args:
    model: Model to average.

  Returns:
    Copy of the model with parameters set to 0s.
  """
  graphdef, state, rest = nnx.split(model, nnx.Param, ...)
  ema_state = jax.tree.map(jnp.zeros_like, state)
  # copy rest of the params, like RNGs, batch stats, etc.
  rest = jax.tree.map(lambda r: r.copy(), rest)
  return nnx.merge(graphdef, ema_state, rest)


def update_ema(model: nnx.Module, *, ema: nnx.Module, decay: float) -> None:
  """Update the EMA of a model.

  Args:
    model: Model to average.
    ema: EMA of the model.
    decay: Decay factor.

  Returns:
    Nothing, just updates the EMA in-place.
  """

  def update_fn(p_model: nnx.Param, p_ema: nnx.Param) -> nnx.Param:
    return p_ema * decay + p_model * (1.0 - decay)

  state, rest = nnx.state(model, nnx.Param, ...)
  graphdef, ema_state, _ = nnx.split(ema, nnx.Param, ...)
  rest = jax.tree.map(lambda r: r.copy(), rest)
  ema_state = jax.tree.map(update_fn, state, ema_state)
  nnx.update(ema, ema_state, rest)
