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
"""Helpers shared across the test suite.

Kept free of :mod:`pytest` so that both the fixtures in ``conftest.py`` and
tests that need bespoke sizes can use them.

Prefer the default sizes of :func:`random_clouds` over ad-hoc ones:
reusing shapes lets XLA
serve a compiled kernel from its cache instead of recompiling it per
module, which is where most of the suite's runtime goes.
"""
import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr

from ott.geometry import pointcloud

__all__ = [
    "root_key", "PointClouds", "QuadClouds", "random_probs", "random_clouds",
    "proj"
]


def root_key() -> jax.Array:
  """A fresh root key.

  Each call returns a distinct array with the same value, so fixtures can
  seed themselves without consuming a key another fixture also uses -- which
  ``jax_debug_key_reuse`` would (rightly) reject.
  """
  return jr.key(0)


@dataclasses.dataclass(frozen=True)
class PointClouds:
  """Two weighted point clouds, the ingredients of a linear OT problem."""
  x: jnp.ndarray  # (n, dim)
  y: jnp.ndarray  # (m, dim)
  a: jnp.ndarray  # (n,), on the simplex
  b: jnp.ndarray  # (m,), on the simplex

  @property
  def n(self) -> int:
    """Number of source points."""
    return self.x.shape[0]

  @property
  def m(self) -> int:
    """Number of target points."""
    return self.y.shape[0]

  @property
  def dim(self) -> int:
    """Dimensionality of both clouds."""
    return self.x.shape[-1]

  def geom(self, **kwargs) -> pointcloud.PointCloud:
    """Point cloud geometry between :attr:`x` and :attr:`y`."""
    return pointcloud.PointCloud(self.x, self.y, **kwargs)


@dataclasses.dataclass(frozen=True)
class QuadClouds(PointClouds):
  """Two weighted clouds plus the intra-domain costs of a quadratic problem."""
  cx: jnp.ndarray  # (n, n)
  cy: jnp.ndarray  # (m, m)


def random_probs(
    rng: jax.Array,
    n: int,
    *,
    offset: float = 0.1,
    zero_at: Sequence[int] = (),
) -> jnp.ndarray:
  """Sample ``n`` random weights on the simplex.

  Args:
    rng: Random key.
    n: Number of weights.
    offset: Added before normalizing, to keep the weights away from 0.
    zero_at: Indices forced to exactly 0, to exercise the handling of
      empty marginals.

  Returns:
    Array of shape ``(n,)`` summing to 1.
  """
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
  """Sample two uniform point clouds with random weights.

  The points only depend on ``rng``, ``n``, ``m`` and ``dim``, so variants
  differing solely in their weights share the same points, and therefore the
  same cost matrix.

  Args:
    rng: Random key.
    n: Number of source points.
    m: Number of target points.
    dim: Dimensionality of both clouds.
    offset: Passed to :func:`random_probs`.
    zero_a: Indices of :attr:`~PointClouds.a` forced to 0.
    zero_b: Indices of :attr:`~PointClouds.b` forced to 0.
  """
  rng_x, rng_y, rng_a, rng_b = jr.split(rng, 4)
  return PointClouds(
      x=jr.uniform(rng_x, (n, dim)),
      y=jr.uniform(rng_y, (m, dim)),
      a=random_probs(rng_a, n, offset=offset, zero_at=zero_a),
      b=random_probs(rng_b, m, offset=offset, zero_at=zero_b),
  )


def proj(matrix: jnp.ndarray, nu: float = 1.0) -> jnp.ndarray:
  """Project a matrix onto the (scaled) Stiefel manifold.

  Args:
    matrix: Matrix to project.
    nu: Positive scale of the projection.
  """
  assert nu > 0.0, nu
  u, _, v_h = jnp.linalg.svd(matrix, full_matrices=False)
  return u.dot(v_h) * jnp.sqrt(nu)
