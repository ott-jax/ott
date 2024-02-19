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

  Encodes time :math:`t` as :math:`cos(\hat{t})` and :math:`sin(\hat{t})`
  where :math:`\hat{t} = [2\pi t, 2\pi 2 t,\dots, 2\pi n_f t]`.

  Args:
    n_freqs: Frequency :math:`n_f` of the cyclical encoding.
  """
  n_freqs: int = 128

  @nn.compact
  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
    """Encode time :math:`t` into a cyclical representation.

    Args:
      t: Time of shape ``[n, 1]``.

    Returns:
      Encoded time of shape ``[n, 2 * n_freqs]``.
    """
    freq = 2 * jnp.arange(self.n_freqs) * jnp.pi
    t = freq * t
    return jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=-1)
