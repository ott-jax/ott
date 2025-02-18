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
import math
from typing import Any, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from scipy.stats import qmc

__all__ = ["sobol_sphere"]


def sobol_sphere(
    n: int,
    d: int,
    seed: int = 0,
    **kwargs: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """TODO."""
  num_radii = int(math.sqrt(n))
  num_samples, num_0s = divmod(n, num_radii)

  radius = jnp.linspace(
      1.0 / (num_radii + 1), num_radii / (num_radii + 1), num_radii
  )
  sphere = _sobol_sphere(num_samples, d, seed=seed, **kwargs)
  points = sphere[None] * radius[:, None, None]
  points = points.reshape(-1, d)

  weights = jnp.full(points.shape[0] + (num_0s > 0), fill_value=1.0 / n)
  if num_0s:
    points = jnp.vstack([points, jnp.zeros([1, d])])
    weights = weights.at[-1].set(num_0s / n)

  return points, weights


def _sobol_sphere(n: int, d: int, seed: int = 0, **kwargs: Any) -> jnp.ndarray:
  # TODO(michalk8): make as pure callback
  sampler = qmc.Sobol(d=d, seed=seed, scramble=True, **kwargs)
  samples = sampler.random_base2(m=math.ceil(math.log2(n)))
  samples = jnp.asarray(samples)  # TODO(michalk8): dtype
  theta = jsp.special.ndtri(samples)[:n]
  eps = jnp.finfo(theta).tiny
  theta /= jnp.linalg.norm(theta, keepdims=True, axis=-1) + eps
  return theta
