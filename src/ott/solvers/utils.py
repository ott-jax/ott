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
from typing import Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.solvers import linear, quadratic

__all__ = [
    "match_linear",
    "match_quadratic",
    "sample_joint",
    "sample_conditional",
    "uniform_sampler",
]

ScaleCost_t = Union[float, Literal["mean", "max_cost", "median"]]


def match_linear(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray],
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    scale_cost: ScaleCost_t = 1.0,
    **kwargs: Any
) -> jnp.ndarray:
  """Compute solution to a linear OT problem.

  Args:
    x: Source point cloud of shape ``[n, d]``.
    y: Target point cloud of shape ``[m, d]``.
    cost_fn: Cost function.
    epsilon: Regularization parameter.
    scale_cost: Scaling of the cost matrix.
    kwargs: Additional arguments for :func:`ott.solvers.linear.solve`.

  Returns:
    Optimal transport matrix.
  """
  geom = pointcloud.PointCloud(
      x, y, cost_fn=cost_fn, epsilon=epsilon, scale_cost=scale_cost
  )
  out = linear.solve(geom, **kwargs)
  return out.matrix


def match_quadratic(
    xx: jnp.ndarray,
    yy: jnp.ndarray,
    x: Optional[jnp.ndarray] = None,
    y: Optional[jnp.ndarray] = None,
    scale_cost: ScaleCost_t = 1.0,
    cost_fn: Optional[costs.CostFn] = None,
    **kwargs: Any
) -> jnp.ndarray:
  """Compute solution to a quadratic OT problem.

  Args:
    xx: Source point cloud of shape ``[n, d1]``.
    yy: Target point cloud of shape ``[m, d2]``.
    x: Linear (fused) term of the source point cloud.
    y: Linear (fused) term of the target point cloud.
    scale_cost: Scaling of the cost matrix.
    cost_fn: Cost function.
    kwargs: Additional arguments for :func:`ott.solvers.quadratic.solve`.

  Returns:
    Optimal transport matrix.
  """
  geom_xx = pointcloud.PointCloud(xx, cost_fn=cost_fn, scale_cost=scale_cost)
  geom_yy = pointcloud.PointCloud(yy, cost_fn=cost_fn, scale_cost=scale_cost)
  if x is None:
    geom_xy = None
  else:
    geom_xy = pointcloud.PointCloud(
        x, y, cost_fn=cost_fn, scale_cost=scale_cost
    )

  out = quadratic.solve(geom_xx, geom_yy, geom_xy, **kwargs)
  return out.matrix


def sample_joint(rng: jax.Array,
                 tmat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Sample jointly from a transport matrix.

  Args:
    rng: Random number generator.
    tmat: Transport matrix of shape ``[n, m]``.

  Returns:
    Source and target indices of shape ``[n,]`` and ``[m,]``, respectively.
  """
  n, m = tmat.shape
  tmat_flattened = tmat.flatten()
  indices = jax.random.choice(
      rng, len(tmat_flattened), p=tmat_flattened, shape=[n]
  )
  src_ixs = indices // m
  tgt_ixs = indices % m
  return src_ixs, tgt_ixs


def sample_conditional(
    rng: jax.Array,
    tmat: jnp.ndarray,
    *,
    k: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Sample conditionally from a transport matrix.

  Args:
    rng: Random number generator.
    tmat: Transport matrix of shape ``[n, m]``.
    k: Expected number of samples to sample per source sample.

  Returns:
    Source and target indices of shape ``[n, k]`` and ``[m, k]``, respectively.
  """
  assert k > 0, "Number of samples per source must be positive."
  n, m = tmat.shape

  src_marginals = tmat.sum(axis=1)
  rng, rng_ixs = jax.random.split(rng, 2)
  indices = jax.random.choice(rng_ixs, a=n, p=src_marginals, shape=(n,))
  tmat = tmat[indices]

  rngs = jax.random.split(rng, n)
  tgt_ixs = jax.vmap(
      lambda rng, row: jax.random.choice(rng, a=m, p=row, shape=(k,)),
      in_axes=[0, 0],
  )(rngs, tmat)  # (m, k)

  src_ixs = jnp.repeat(indices[:, None], k, axis=1)  # (n, k)
  return src_ixs, tgt_ixs


def uniform_sampler(
    rng: jax.Array,
    num_samples: int,
    low: float = 0.0,
    high: float = 1.0,
    offset: Optional[float] = None
) -> jnp.ndarray:
  r"""Sample from a uniform distribution.

  Sample :math:`t` from a uniform distribution :math:`[low, high]`.
  If `offset` is not :obj:`None`, one element :math:`t` is sampled from
  :math:`[low, high]` and the K samples are constructed via
  :math:`(t + k)/K \mod (high - low - offset) + low`.

  Args:
    rng: Random number generator.
    num_samples: Number of samples to generate.
    low: Lower bound of the uniform distribution.
    high: Upper bound of the uniform distribution.
    offset: Offset of the uniform distribution.
      If :obj:`None`, no offset is used.

  Returns:
    An array of shape ``[num_samples, 1]``.
  """
  if offset is None:
    return jax.random.uniform(rng, (num_samples, 1), minval=low, maxval=high)

  t = jax.random.uniform(rng, (1, 1), minval=low, maxval=high)
  mod_term = ((high - low) - offset)
  return (t + jnp.arange(num_samples)[:, None] / num_samples) % mod_term
