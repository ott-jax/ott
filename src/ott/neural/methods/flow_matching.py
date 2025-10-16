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
from typing import Callable, Dict, Literal, Optional

import jax

import optax
from flax import nnx

__all__ = ["flow_matching_step"]


def flow_matching_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: Dict[Literal["t", "x_t", "v_t", "cond"], jax.Array],
    *,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = optax.squared_error,
    ema_update_fn: Optional[Callable[[nnx.Module], None]] = None,
    dropout_rngs: nnx.Rngs,
) -> Dict[Literal["loss", "grad_norm"], jax.Array]:
  """TODO."""

  def compute_loss(model: nnx.Module, rngs: nnx.Rngs) -> jax.Array:
    t, x_t, v_t = batch["t"], batch["x_t"], batch["v_t"]
    cond = batch.get("cond")
    v_pred = model(t, x_t, cond, rngs=rngs)
    return loss_fn(v_pred, v_t).mean()

  loss, grads = nnx.value_and_grad(compute_loss)(model, dropout_rngs)
  optimizer.update(model, grads)
  if ema_update_fn is not None:
    ema_update_fn(model)

  grad_norm = optax.global_norm(grads)

  return {"loss": loss, "grad_norm": grad_norm}


def evaluate_velocity_field(
    model: nnx.Module,
    x0: jax.Array,
    cond: Optional[jax.Array] = None,
):
  """TODO."""
  pass
