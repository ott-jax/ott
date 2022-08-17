# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions related to momemtum."""

from typing import TYPE_CHECKING

import jax.numpy as jnp

from ott.core import dataclasses

if TYPE_CHECKING:
  from ott.core import sinkhorn


@dataclasses.register_pytree_node
class Momentum:
  """Momentum for Sinkhorn updates, either constant or adaptive."""

  start: int = 0
  value: float = 1.0
  inner_iterations: int = 1

  def weight(self, state: "sinkhorn.SinkhornState", iteration: int) -> float:
    """Compute momentum term if needed, using previously seen errors."""
    return jnp.where(
        iteration >= jnp.where(self.start == 0, jnp.inf, self.start),
        self.at(state), self.value
    )

  def at(self, state: "sinkhorn.SinkhornState") -> float:
    """Momentum formula :cite:`lehmann:21`, eq. 5."""
    idx = self.start // self.inner_iterations
    error_ratio = jnp.minimum(
        state.errors[idx - 1, -1] / state.errors[idx - 2, -1], 0.99
    )
    power = 1.0 / self.inner_iterations
    return 2.0 / (1.0 + jnp.sqrt(1.0 - error_ratio ** power))

  def __call__(
      self,
      weight: float,
      value: jnp.ndarray,
      new_value: jnp.ndarray,
      lse_mode: bool = True
  ) -> jnp.ndarray:
    if lse_mode:
      value = jnp.where(jnp.isfinite(value), value, 0.0)
      return (1.0 - weight) * value + weight * new_value
    else:
      value = jnp.where(value > 0.0, value, 1.0)
      return value ** (1.0 - weight) * new_value ** weight
