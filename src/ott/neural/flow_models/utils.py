from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import costs, pointcloud
from ott.solvers import linear

__all__ = [
    "match_linear", "sample_joint", "sample_conditional", "resample_data"
]


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


def resample_data(*data: Optional[jnp.ndarray],
                  ixs: jnp.ndarray) -> Tuple[Optional[jnp.ndarray], ...]:
  """TODO."""
  if ixs.ndim == 2:
    ixs = ixs.reshape(-1)
  assert ixs.ndim == 1, ixs.shape
  data = jtu.tree_map(lambda arr: None if arr is None else arr[ixs], data)
  return data[0] if len(data) == 1 else data
