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
    "cyclical_time_encoder",
    "uniform_sampler",
    "multivariate_normal",
]

ScaleCost_t = Union[float, Literal["mean", "max_cost", "median"]]


def match_linear(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    scale_cost: ScaleCost_t = 1.0,
    **kwargs: Any
) -> jnp.ndarray:
  """Compute solution to a linear OT problem.

  Args:
    x: Linear term of the source point cloud.
    y: Linear term of the target point cloud.
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
    # TODO(michalk8): expose for all the costs
    scale_cost: ScaleCost_t = 1.0,
    cost_fn: Optional[costs.CostFn] = None,
    **kwargs: Any
) -> jnp.ndarray:
  """Compute solution to a quadratic OT problem.

  Args:
    xx: Quadratic (incomparable) term of the source point cloud.
    yy: Quadratic (incomparable) term of the target point cloud.
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
  """Sample from a transport matrix.
  
  Args:
    rng: Random number generator.
    tmat: Transport matrix.

  Returns:
    Source and target indices sampled from the transport matrix.
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
    uniform_marginals: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Sample indices from a transport matrix.

  Args:
    rng: Random number generator.
    tmat: Transport matrix.
    k: Expected number of samples to sample per row.
    uniform_marginals: If :obj:`True`, sample exactly `k` samples
      per row, otherwise sample proportionally to the sums of the
      rows of the transport matrix.

  Returns:
    Source and target indices sampled from the transport matrix.
  """
  assert k > 0, "Number of samples per row must be positive."
  n, m = tmat.shape

  if uniform_marginals:
    indices = jnp.arange(n)
  else:
    src_marginals = tmat.sum(axis=1)
    rng, rng_ixs = jax.random.split(rng, 2)
    indices = jax.random.choice(
        rng_ixs, a=n, p=src_marginals, shape=(len(src_marginals),)
    )
    tmat = tmat[indices]

  tgt_ixs = jax.vmap(
      lambda row: jax.random.choice(rng, a=m, p=row, shape=(k,))
  )(tmat)  # (m, k)

  src_ixs = jnp.repeat(indices[:, None], k, axis=1)  # (n, k)
  return src_ixs, tgt_ixs


def cyclical_time_encoder(t: jnp.ndarray, n_freqs: int = 128) -> jnp.ndarray:
  r"""Encode time :math:`t` into a cyclical representation.

  Time :math:`t` is encoded as :math:`cos(\hat{t})` and :math:`sin(\hat{t})`
  where :math:`\hat{t} = [2\pi t, 2\pi 2 t,\dots, 2\pi n_f t]`.

  Args:
    t: Time of shape ``[n, 1]``.
    n_freqs: Frequency :math:`n_f` of the cyclical encoding.

  Returns:
    Encoded time of shape ``[n, 2 * n_freqs]``.
  """
  freq = 2 * jnp.arange(n_freqs) * jnp.pi
  t = freq * t
  return jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=-1)


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
    offset: Offset of the uniform distribution. If :obj:`None`, no offset is
      used.

  Returns:
    An array with `num_samples` samples of the time :math:`t`.
  """
  if offset is None:
    return jax.random.uniform(rng, (num_samples, 1), minval=low, maxval=high)

  t = jax.random.uniform(rng, (1, 1), minval=low, maxval=high)
  mod_term = ((high - low) - offset)
  return (t + jnp.arange(num_samples)[:, None] / num_samples) % mod_term


def multivariate_normal(
    rng: jax.Array,
    shape: Tuple[int, ...],
    dim: int,
    mean: float = 0.0,
    cov: float = 1.0
) -> jnp.ndarray:
  """TODO."""
  mean = jnp.full(dim, fill_value=mean)
  cov = jnp.diag(jnp.full(dim, fill_value=cov))
  return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)
