from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.solvers import linear

__all__ = ["match_linear", "sample_joint"]


def match_linear(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    # TODO(michalk8): expose rest of the geom arguments
    **kwargs: Any
) -> jnp.ndarray:
  """TODO."""
  geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=epsilon)
  out = linear.solve(geom, **kwargs)
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
