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
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import costs, pointcloud
from ott.solvers import linear

__all__ = ["sliced_wasserstein"]

Projector_t = Callable[[int, int], jnp.ndarray]


def sliced_wasserstein(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    cost_fn: Optional[costs.CostFn] = None,
    proj_fn: Optional[Projector_t] = None,
    return_univariate_output_obj: bool = False,
    kwargs_univariate_solver: Optional[Any] = None,
    **kwargs
) -> jnp.ndarray:
  r"""Compute the Sliced Wasserstein distance between two weighted point clouds.

  Follows the approach outlined in :cite:`rabin:12` to compute a proxy for OT
  distances that relies on unidimensional projections.

  Args:
    x: Array[n, dim] of source points' coordinates
    y: Array[m, dim] of target points' coordinates
    a: Array[n,] of source probability weights
    b: Array[m,] of target probability weights
    cost_fn: cost function. Must be submodular function of two real arguments,
      i.e. such that :math:`\partial c(x,y)/\partial x \partial y <0`. If
      obj:`None`, use :class:`~ott.geometry.costs.SqEuclidean`.
    proj_fn: projection function, mapping any `[b,d]` matrix of coordinates to
      `[b,n_proj]` coordinates, on which 1D transport (for `n_proj` directions)
      are subsequently computed. Default is to use `n_proj` random vectors
      distributed on the `d`-sphere and project data onto them.
    return_univariate_output_obj: whether to retun a detailed univarite output
      object.
    kwargs_univariate_solver: passed on to solver
      :class:`ott.solvers.linear.solve_univariate`, to recover e.g.
      detailed information on univariate outputs on each slices (transport or
      dual variables).
    **kwargs: parameters to be passed on to `proj_fn`. Could for instance
      include, as done with default setting, a `rng` key and specifying `n_proj`
      directions.

  Returns:
    The sliced Wasserstein distance, possibly with a
    :class:`~ott.solvers.linear.univariate.UnivariateOutput` object.
  """
  dim = x.shape[1]
  if proj_fn is None:

    def proj_fn(
        input: jnp.ndarray,
        rng: Optional[jax.Array] = None,
        n_proj: int = 1000
    ):
      rng = utils.default_prng_key(rng)
      proj_m = jax.random.normal(rng, (n_proj, dim))
      proj_m /= jnp.linalg.norm(proj_m, axis=1)[:, None]
      return x @ proj_m.T

  x_proj, y_proj = proj_fn(x, **kwargs), proj_fn(y, **kwargs),
  geom = pointcloud.PointCloud(x_proj, y_proj, cost_fn=cost_fn)
  kw = {} if kwargs_univariate_solver is None else kwargs_univariate_solver
  out = linear.solve_univariate(geom, a, b, **kw)

  if not return_univariate_output_obj:
    return jnp.sum(out.ot_costs)
  return jnp.sum(out.ot_costs), out
