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
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.solvers import linear

__all__ = ["sliced_w"]

Randfunc_t = Callable[[jax.Array, (int, int)], jnp.ndarray]


def sliced_w(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    proj_matrix: Optional[jnp.ndarray] = None,
    n_proj: int = 100,
    random_direction_generator: Optional[Randfunc_t] = None,
    rng: Optional[jax.Array] = None,
) -> jnp.ndarray:
  r"""Compute the Sliced Wasserstein distance between two weighted point clouds.

  Args:
    x: Array[n, dim] of source points' coordinates
    y: Array[m, dim] of target points' coordinates
    a: Array[n,] of source probability weights
    b: Array[m,] of target probability weights
    cost_fn: cost function. Must be submodular function of two real arguments,
      i.e. such that :math:`\partial c(x,y)/\partial x \partial y <0`. Defaults
      to square Euclidean distance.
    n_proj: number of projections.
    random_direction_generator: random matrix generator. Must take `rng` value
      and shape parameter to return a matrix with randomly distributed entries
      of that shape.
    rng: random key.
    proj_matrix: Array of dimension `[n_proj, dim]` that prestores projections.

  Returns:
    The sliced Wasserstein distance.
  """
  dim = x.shape[1]
  if proj_matrix is not None:
    assert dim == proj_matrix.shape[1]
    n_proj = proj_matrix.shape[0]
  else:
    if random_direction_generator is None:

      def random_direction_generator(rng: jax.Array, shape: Tuple[int, int]):
        proj_m = jax.random.normal(rng, shape)
        return proj_m / jnp.linalg.norm(proj_m, axis=1)[:, None]

    rng = jax.random.PRNGKey(0) if rng is None else rng
    proj_matrix = random_direction_generator(rng, (n_proj, dim))

  x_proj, y_proj = x @ proj_matrix.T, y @ proj_matrix.T
  geom = pointcloud.PointCloud(x_proj, y_proj, cost_fn=cost_fn)
  out = linear.solve_univariate(geom, a, b)
  return jnp.sum(out.ot_costs)
