# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from ott.core import fixed_point_loop
from ott.geometry import pointcloud


class KPPState(NamedTuple):
  key: jnp.ndarray
  centroids: jnp.ndarray
  centroid_dists: jnp.ndarray

  def set(self, **kwargs: Any) -> 'KMeansState':
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)


class KMeansConstants(NamedTuple):
  geom: pointcloud.PointCloud
  weights: jnp.ndarray
  k: int
  tol: float


class KMeansState(NamedTuple):
  centroids: jnp.ndarray
  assignment: jnp.ndarray
  distortions: jnp.ndarray

  def set(self, **kwargs: Any) -> 'KMeansState':
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)


class KMeansOutput(NamedTuple):
  centroids: jnp.ndarray
  assignment: jnp.ndarray
  distortion: float
  converged: bool

  @classmethod
  def from_state(cls, state: KMeansState) -> "KMeansOutput":
    err = state.distortions
    distortion = jnp.nanmin(jnp.where(err == -1, jnp.nan, err))
    return cls(
        centroids=state.centroids,
        assignment=state.assignment,
        distortion=distortion,
        converged=jnp.sum(err == -1) > 0
    )


# TODO(michalk8): refactor to return directly the centroids
def _random_init(
    geom: pointcloud.PointCloud, k: int, key: jnp.ndarray
) -> jnp.ndarray:
  ixs = jnp.arange(geom.shape[0])
  return jax.random.choice(key, ixs, shape=(k,), replace=False)


def _kmeans_plus_plus(
    geom: pointcloud.PointCloud,
    k: int,
    key: jnp.ndarray,
    n_local_trials: Optional[int] = None,
) -> jnp.ndarray:

  def init_fn(geom: pointcloud.PointCloud, key: jnp.ndarray) -> KPPState:
    key, next_key = jax.random.split(key, 2)
    ix = jax.random.choice(key, jnp.arange(10), shape=())
    centroids = jnp.full((k, geom.cost_rank), jnp.inf).at[0].set(geom.x[ix])
    dists = geom.subset(None, ix, batch_size=None).cost_matrix.ravel()
    return KPPState(key=next_key, centroids=centroids, centroid_dists=dists)

  def cond_fn(
      iteration: int, const: Tuple[pointcloud.PointCloud, jnp.ndarray],
      state: KPPState
  ) -> bool:
    del iteration, const, state
    return True

  def body_fn(
      iteration: int, const: Tuple[pointcloud.PointCloud, jnp.ndarray],
      state: KPPState, compute_error: bool
  ) -> KPPState:
    del compute_error
    # TODO(michalk8): verify impl.
    key, next_key = jax.random.split(state.key, 2)
    geom, ixs = const
    # TODO(michalk8): check if needs to be normalized
    probs = state.centroid_dists  # / state.centroid_dists.sum()
    ixs = jax.random.choice(key, ixs, shape=(n_local_trials,), p=probs)
    candidate_dists = geom.subset(ixs, None, batch_size=None).cost_matrix

    best_ix = jnp.argmin(candidate_dists.sum(1))
    centroids = state.centroids.at[iteration + 1].set(geom.x[best_ix])
    centroid_dists = candidate_dists[best_ix]

    return state.set(
        key=next_key, centroids=centroids, centroid_dists=centroid_dists
    )

  if n_local_trials is None:
    n_local_trials = 2 + int(math.log(k))

  state = init_fn(geom, key)
  constants = (geom, jnp.arange(geom.shape[0]))
  state = fixed_point_loop.fixpoint_iter(
      cond_fn,
      body_fn,
      min_iterations=k - 1,
      max_iterations=k - 1,
      inner_iterations=1,
      constants=constants,
      state=state
  )

  return state


def _kmeans(
    key: jnp.ndarray,
    geom: pointcloud.PointCloud,
    k: int,
    tol: float = 1e-4,
    weights: Optional[jnp.ndarray] = None,
    # TODO(michalk8): allow callables?
    init_random: bool = True,
    min_iter: int = 20,
    max_iter: int = 20,
) -> KMeansOutput:

  def update_centroids(
      consts: KMeansConstants, state: KMeansState
  ) -> jnp.ndarray:
    data = jnp.hstack([consts.geom.x, constants.weights[:, None]])
    data = jax.ops.segment_sum(
        data, state.assignment, num_segments=consts.k, unique_indices=True
    )
    centroids, weights = data[:, :-1], data[:, -1]
    return centroids / weights[:, None]

  def init_fn(geom: pointcloud.PointCloud) -> KMeansState:
    if init_random:
      ixs = _random_init(geom, k=k, key=key)
    else:
      ixs = _kmeans_plus_plus(geom, k=k, key=key)
    geom = geom.subset(None, ixs, batch_size=None)
    cost_matrix = geom.cost_matrix  # (n, k)
    assignment = jnp.argmin(cost_matrix, axis=1)  # (n,)
    centroids = geom.x[ixs]
    # weights are NOT normalized
    distortion = jnp.mean(jnp.min(cost_matrix, axis=1) * weights)  # ()
    distortions = jnp.full((max_iter + 1,), -1.).at[0].set(distortion)

    return KMeansState(
        centroids=centroids,
        assignment=assignment,
        distortions=distortions,
    )

  def cond_fn(
      iteration: int, const: KMeansConstants, state: KMeansState
  ) -> bool:
    # TODO(michalk8): verify
    # TODO(michalk8): add strict convergence criterion
    err = state.distortions
    return jnp.logical_or(
        iteration < 1, err[iteration - 1] - err[iteration] > const.tol
    )

  def body_fn(
      iteration: int, const: KMeansConstants, state: KMeansState,
      compute_error: bool
  ) -> KMeansState:
    del compute_error
    centroids = update_centroids(const, state)
    # TODO(michalk8): correctly initialize
    cost_matrix = pointcloud.PointCloud(const.geom.x, centroids).cost_matrix
    assignment = jnp.argmin(cost_matrix, axis=1)
    # weights are NOT normalized
    # TODO(michalk8): benchmark which is faster
    # distortion = jnp.mean(jnp.min(cost_matrix, axis=1) * weights)
    distortion = jnp.mean(cost_matrix[jnp.arange(len(assignment)), assignment])
    distortions = state.distortions.at[iteration + 1].set(distortion)

    return state.set(
        centroids=centroids, assignment=assignment, distortions=distortions
    )

  state = init_fn(geom)
  constants = KMeansConstants(geom=geom, weights=weights, k=k, tol=tol)
  state = fixed_point_loop.fixpoint_iter(
      cond_fn,
      body_fn,
      min_iterations=min_iter,
      max_iterations=max_iter,
      inner_iterations=1,
      constants=constants,
      state=state
  )
  return KMeansOutput.from_state(state)


def kmeans_new(
    # TODO(michalk8): handle LRCGeom
    geom: pointcloud.PointCloud,
    k: int,
    n_iter: int = 20,
    tol: float = 1e-4,
    weights: Optional[jnp.ndarray] = None,
    # TODO(michalk8): change default
    init_random: bool = True,
    seed: int = 0,
    min_iter: int = 20,
    max_iter: int = 20,
) -> KMeansOutput:
  # TODO(michalk8): handle cosine distance?
  # TODO(michalk8): center PC as in sklearn?
  keys = jax.random.split(jax.random.PRNGKey(seed), n_iter)

  # TODO(michalk8): consider normalizing
  if weights is None:
    weights = jnp.ones(geom.shape[0])
  assert weights.shape == (geom.shape[0],)

  out = jax.vmap(
      _kmeans, in_axes=[0] + [None] * 7
  )(keys, geom, k, tol, weights, init_random, min_iter, max_iter)

  best_ix = jnp.argmin(out.distortion)
  return jax.tree_util.tree_map(lambda arr: arr[best_ix], out)
