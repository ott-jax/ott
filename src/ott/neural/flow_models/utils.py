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
  """TODO."""
  geom = pointcloud.PointCloud(
      x, y, cost_fn=cost_fn, epsilon=epsilon, scale_cost=scale_cost
  )
  out = linear.solve(geom, **kwargs)
  return out.matrix


def match_quadratic(
    xx: jnp.ndarray,
    yy: jnp.ndarray,
    xy: Optional[jnp.ndarray] = None,
    # TODO(michalk8): expose for all the costs
    scale_cost: ScaleCost_t = 1.0,
    cost_fn: Optional[costs.CostFn] = None,
    **kwargs: Any
) -> jnp.ndarray:
  """TODO."""
  geom_xx = pointcloud.PointCloud(xx, cost_fn=cost_fn, scale_cost=scale_cost)
  geom_yy = pointcloud.PointCloud(yy, cost_fn=cost_fn, scale_cost=scale_cost)
  if xy is None:
    geom_xy = None
  else:
    geom_xy = pointcloud.PointCloud(xy, cost_fn=cost_fn, scale_cost=scale_cost)

  out = quadratic.solve(geom_xx, geom_yy, geom_xy, **kwargs)
  return out.matrix


def sample_joint(rng: jax.Array,
                 tmat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """TODO."""
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
  """TODO."""
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
