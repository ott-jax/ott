# coding=utf-8
# Copyright 2021 Google LLC.
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

import jax.numpy as jnp
from ott.core import dataclasses


@dataclasses.register_pytree_node
class Momentum:
  """Momentum for Sinkhorn updates, either constant or adaptive."""

  start: int = 0
  value: float = 1.0

  def weight(self, iteration, errors, inner_iterations):
    """Computes momentum term if needed, using previously seen errors."""
    return jnp.where(
        iteration >= jnp.where(self.start == 0, jnp.inf, self.start),
        self.at(errors, self.start // inner_iterations, inner_iterations),
        self.value)

  def at(self, errors, idx, inner_iterations):
    """Momentum formula, https://arxiv.org/pdf/2012.12562v1.pdf, p.7 and (5)."""
    error_ratio = jnp.minimum(errors[idx - 1, -1] / errors[idx - 2, -1], .99)
    power = 1.0 / inner_iterations
    return 2.0 / (1.0 + jnp.sqrt(1.0 - error_ratio ** power))
