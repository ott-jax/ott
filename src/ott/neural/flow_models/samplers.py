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

__all__ = ["uniform_sampler"]


def uniform_sampler(
    rng: jax.Array,
    num_samples: int,
    low: float = 0.0,
    high: float = 1.0,
    offset: Optional[float] = None
) -> jnp.ndarray:
  r"""Sample from a uniform distribution.

  Sample :math:`t` from a uniform distribution :math:`[low, high]`.
  If `offset` is not :obj:`None`, one element :math:`t` is sampled from
  :math:`[low, high]` and the K samples are constructed via
  :math:`(t + k)/K \mod (high - low - offset) + low`.

  Args:
    rng: Random number generator.
    num_samples: Number of samples to generate.
    low: Lower bound of the uniform distribution.
    high: Upper bound of the uniform distribution.
    offset: Offset of the uniform distribution. If :obj:`None`, no offset is
      used.

  Returns:
    An array with `num_samples` samples of the time :math:`t`.
  """
  if offset is None:
    return jax.random.uniform(rng, (num_samples, 1), minval=low, maxval=high)

  t = jax.random.uniform(rng, (1, 1), minval=low, maxval=high)
  mod_term = ((high - low) - offset)
  return (t + jnp.arange(num_samples)[:, None] / num_samples) % mod_term
