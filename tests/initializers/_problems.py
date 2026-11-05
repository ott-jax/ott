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
"""Problems shared by the initializer tests."""

import jax
import jax.numpy as jnp
import jax.random as jr

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem

__all__ = ["create_ot_problem"]


def create_ot_problem(
    rng: jax.Array,
    n: int,
    m: int,
    epsilon: float = 1e-2,
    batch_size: int | None = None,
) -> linear_problem.LinearProblem:
  """Two well-separated Gaussian clouds in 2D, with uniform marginals."""
  rng_x, rng_y = jr.split(rng)
  x = jr.normal(rng_x, (n, 2)) + jnp.array([-1.0, 1.0]) * 5
  y = jr.normal(rng_y, (m, 2))
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon, batch_size=batch_size)
  return linear_problem.LinearProblem(
      geom, a=jnp.ones(n) / n, b=jnp.ones(m) / m
  )
