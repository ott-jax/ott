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
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.solvers.linear import univariate

__all__ = ["random_proj_sphere", "sliced_wasserstein"]

Projector = Callable[[jnp.ndarray, int, jax.Array], jnp.ndarray]


def random_proj_sphere(
    x: jnp.ndarray,
    n_proj: int = 1000,
    rng: Optional[jax.Array] = None
) -> jnp.ndarray:
  """Project data on directions sampled randomly from sphere.

  Args:
    x: Array of size ``[n, dim]``.
    n_proj: Number of randomly generated projections.
    rng: Key used to sample feature extractors.

  Returns:
    Array of size ``[n, n_proj]`` features.
  """
  rng = utils.default_prng_key(rng)
  dim = x.shape[-1]
  proj_m = jax.random.normal(rng, (n_proj, dim))
  proj_m /= jnp.linalg.norm(proj_m, axis=1, keepdims=True)
  return x @ proj_m.T


def sliced_wasserstein(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    cost_fn: Optional[costs.CostFn] = None,
    proj_fn: Optional[Projector] = None,
    weights: Optional[jnp.ndarray] = None,
    return_transport: bool = False,
    return_dual_variables: bool = False,
    **kwargs: Any,
) -> Tuple[jnp.ndarray, univariate.UnivariateOutput]:
  r"""Compute the Sliced Wasserstein distance between two weighted point clouds.

  Follows the approach outlined in :cite:`rabin:12` to compute a proxy for OT
  distances that relies on creating features (possibly randomly) for data,
  through e.g., projections, and then sum the 1D Wasserstein distances between
  these features' univariate distributions on both source and target samples.

  Args:
    x: Array of shape ``[n, dim]`` of source points' coordinates.
    y: Array of shape ``[m, dim]`` of target points' coordinates.
    a: Array of shape ``[n,]`` of source probability weights.
    b: Array of shape ``[m,]`` of target probability weights.
    cost_fn: Cost function. Must be a submodular function of two real arguments,
      i.e. such that :math:`\partial c(x,y)/\partial x \partial y <0`. If
      :obj:`None`, use :class:`~ott.geometry.costs.SqEuclidean`.
    proj_fn: Projection function, mapping any ``[b, dim]`` matrix of coordinates
      to ``[b, n_proj]`` matrix of features, on which 1D transports (for
      ``n_proj`` directions) are subsequently computed independently.
      By default, use :func:`~ott.tools.sliced.random_proj_sphere`.
    weights: Array of shape ``[n_proj,]`` of weights used to average the
      ``n_proj`` 1D Wasserstein contributions (one for each feature) and form
      the sliced Wasserstein distance. Uniform by default, resulting in average
      of all these values.
    return_transport: Whether to store ``n_proj`` transport plans in the output.
    return_dual_variables: Whether to store ``n_proj`` pairs of dual vectors
      in the output.
    kwargs: Keyword arguments to ``proj_fn``. Could for instance
      include, as done with default projector, number of ``n_proj`` projections,
      as well as a ``rng`` key to sample as many directions.

  Returns:
    The sliced Wasserstein distance with the corresponding output object.
  """
  if proj_fn is None:
    proj_fn = random_proj_sphere

  x_proj, y_proj = proj_fn(x, **kwargs), proj_fn(y, **kwargs),
  geom = pointcloud.PointCloud(x_proj, y_proj, cost_fn=cost_fn)

  out = linear.solve_univariate(
      geom,
      a,
      b,
      return_transport=return_transport,
      return_dual_variables=return_dual_variables
  )
  return jnp.average(out.ot_costs, weights=weights), out
