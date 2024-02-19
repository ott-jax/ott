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

import flax.linen as nn

__all__ = ["CyclicalTimeEncoder"]


class CyclicalTimeEncoder(nn.Module):
  r"""A cyclical time encoder.

  Encodes time :math:`t` as :math:`cos(\tilde{t})` and :math:`sin(\tilde{t})`
  where :math:`\tilde{t} = [2\pi  t, 2\pi 2 t,\ldots, 2\pi n_frequencies t]`.

  Args:
    n_frequencies: Frequency of cyclical encoding.
  """
  n_frequencies: int = 128

  @nn.compact
  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
    """Encode time :math:`t` into a cyclical representation.

    Args:
      t: Time of shape ``[n, 1]``.

    Returns:
      Encoded time of shape ``[n, 2 * n_frequencies]``.
    """
    freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
    t = freq * t
    return jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=-1)
