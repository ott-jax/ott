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
from typing import Optional

import jax
import jax.numpy as jnp

from ott.geometry import costs
from ott.math import utils as math_utils

__all__ = ["min_operator"]


def min_operator(
    g: jax.Array,
    x: jax.Array,
    y: jax.Array,
    epsilon: Optional[float],
    cost_fn: costs.CostFn,
) -> jax.Array:
  """TODO.

  Args:
    g: TODO.
    x: TODO.
    y: TODO.
    epsilon: TODO.
    cost_fn: TODO.

  Returns:
    TODO.
  """
  n = x.shape[0]
  assert g.shape == (n,), g.shape
  cost = cost_fn.all_pairs(x, y)
  if epsilon is None:  # hard min
    z = g[:, None] - cost
    return -jnp.max(z, axis=-1)
  # soft min
  z = (g[:, None] - cost) / epsilon - jnp.log(n)
  return -epsilon * math_utils.logsumexp(z, axis=-1)
