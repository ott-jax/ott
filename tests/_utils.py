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
"""Data generators shared across the test suite."""
import dataclasses
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem

__all__ = [
    "PointClouds", "QuadClouds", "random_weights", "random_clouds", "proj"
]


@dataclasses.dataclass(frozen=True)
class PointClouds:
  """Two weighted point clouds, the ingredients of a linear OT problem."""
  x: jnp.ndarray  # (n, dim)
  y: jnp.ndarray  # (m, dim)
  a: jnp.ndarray  # (n,), on the simplex
  b: jnp.ndarray  # (m,), on the simplex

  @property
  def n(self) -> int:
    return self.x.shape[0]

  @property
  def m(self) -> int:
    return self.y.shape[0]

  @property
  def dim(self) -> int:
    return self.x.shape[-1]

  def geom(self, **kwargs: Any) -> pointcloud.PointCloud:
    """Point cloud geometry between :attr:`x` and :attr:`y`."""
    return pointcloud.PointCloud(self.x, self.y, **kwargs)

  def problem(
      self,
      *,
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      **kwargs: Any,
  ) -> linear_problem.LinearProblem:
    """Linear problem on :meth:`geom`, with marginals :attr:`a`, :attr:`b`."""
    return linear_problem.LinearProblem(
        self.geom(**kwargs), a=self.a, b=self.b, tau_a=tau_a, tau_b=tau_b
    )


@dataclasses.dataclass(frozen=True)
class QuadClouds(PointClouds):
  """Two weighted clouds plus the intra-domain costs of a quadratic problem."""
  cx: jnp.ndarray  # (n, n)
  cy: jnp.ndarray  # (m, m)

  def quad_problem(
      self,
      *,
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      **kwargs: Any,
  ) -> quadratic_problem.QuadraticProblem:
    """Quadratic problem between the intra-domain geometries of x and y."""
    return quadratic_problem.QuadraticProblem(
        pointcloud.PointCloud(self.x, **kwargs),
        pointcloud.PointCloud(self.y, **kwargs),
        a=self.a,
        b=self.b,
        tau_a=tau_a,
        tau_b=tau_b,
    )


def random_weights(
    rng: jax.Array,
    n: int,
    *,
    offset: float = 0.1,
    zero_at: Sequence[int] = (),
) -> jnp.ndarray:
  """Sample ``n`` random weights on the simplex, 0 at ``zero_at``."""
  a = jr.uniform(rng, (n,)) + offset
  a = a.at[jnp.asarray(zero_at, dtype=int)].set(0.0)
  return a / jnp.sum(a)


def random_clouds(
    rng: jax.Array,
    *,
    n: int = 13,
    m: int = 17,
    dim: int = 4,
    offset: float = 0.1,
    zero_a: Sequence[int] = (),
    zero_b: Sequence[int] = (),
) -> PointClouds:
  """Sample two uniform point clouds with random weights."""
  rng_x, rng_y, rng_a, rng_b = jr.split(rng, 4)
  return PointClouds(
      x=jr.uniform(rng_x, (n, dim)),
      y=jr.uniform(rng_y, (m, dim)),
      a=random_weights(rng_a, n, offset=offset, zero_at=zero_a),
      b=random_weights(rng_b, m, offset=offset, zero_at=zero_b),
  )


def proj(matrix: jnp.ndarray, nu: float = 1.0) -> jnp.ndarray:
  """Project a matrix onto the Stiefel manifold, scaled by ``nu``."""
  assert nu > 0.0, nu
  u, _, v_h = jnp.linalg.svd(matrix, full_matrices=False)
  return u.dot(v_h) * jnp.sqrt(nu)
