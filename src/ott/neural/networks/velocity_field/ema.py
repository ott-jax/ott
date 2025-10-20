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
import jax.numpy as jnp
import jax.tree_util as jtu

from flax import nnx

__all__ = ["EMA", "init_ema", "update_ema"]


class EMA(nnx.Module):
  """TODO."""

  def __init__(self, net: nnx.Module, *, decay: float):
    super().__init__()
    self.ema = init_ema(net)
    self.decay = decay

  def update(self, net: nnx.Module) -> None:
    """TODO."""
    update_ema(net, self.ema, decay=self.decay)


def init_ema(net: nnx.Module) -> nnx.Module:
  """TODO."""
  graphdef, state, rest = nnx.split(net, nnx.Param, ...)
  ema_state = jtu.tree_map(jnp.zeros_like, state)
  rest = jtu.tree_map(lambda r: r.copy(), rest)
  return nnx.merge(graphdef, ema_state, rest)


def update_ema(net: nnx.Module, ema: nnx.Module, *, decay: float) -> None:
  """TODO."""

  def update_fn(p_net: nnx.Param, p_ema: nnx.Param) -> nnx.Param:
    return p_ema * decay + p_net * (1.0 - decay)

  state, rest = nnx.state(net, nnx.Param, ...)
  graphdef, ema_state, _ = nnx.split(ema, nnx.Param, ...)
  rest = jtu.tree_map(lambda r: r.copy(), rest)
  ema_state = jtu.tree_map(update_fn, state, ema_state)
  nnx.update(ema, ema_state, rest)
