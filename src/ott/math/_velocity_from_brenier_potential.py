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
import functools
from typing import Callable

import jax
import jax.numpy as jnp

from ott import math

__all__ = ["velocity_from_brenier_potential"]


def velocity_from_brenier_potential(
    potential: Callable[[jnp.ndarray], jnp.ndarray]
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Get optimal time-dependent velocity field from :term:`Brenier potential`.

  The solution is computed numerically using a :term:`Legendre transform`.

  Args:
    potential: A convex potential.
    kwargs: passed to the solver function returned by the
      :func:`~ott.math.legendre` transform.

  Returns:
    A time-parameterized velocity field ``vel(t,x)`` that expects a single ``t``
    time value and an array ``x`` of size ``[d]`` where ``d`` is the input
    dimension of ``potential``.
  """

  @functools.partial(jax.vmap, in_axes=[0, 0])
  def vel(t: jnp.array, z: jnp.array) -> jnp.array:

    def pot_t(x: jnp.ndarray) -> jnp.ndarray:
      return 0.5 * (1 - t) * jnp.sum(x ** 2) + t * potential(x)

    x = jax.grad(math.legendre(pot_t))(z)
    return jax.grad(potential)(x) - x

  return vel
