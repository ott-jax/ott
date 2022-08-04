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
import functools
import math
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import fixed_point_loop
from ott.geometry import pointcloud

__all__ = ["kmeans", "KMeansOutput"]

Init_t = Union[Literal["k-means++", "random"],
               Callable[[pointcloud.PointCloud, int, jnp.ndarray], jnp.ndarray]]


class KPPState(NamedTuple):
  key: jnp.ndarray
  centroids: jnp.ndarray
  centroid_dists: jnp.ndarray


class KMeansState(NamedTuple):
  centroids: jnp.ndarray
  prev_assignment: jnp.ndarray
  assignment: jnp.ndarray
  errors: jnp.ndarray


class KMeansOutput(NamedTuple):
  centroids: jnp.ndarray
  assignment: jnp.ndarray
  inner_errors: Optional[jnp.ndarray]
  iteration: int
  error: float
  converged: bool

  @classmethod
  def from_state(
      cls,
      state: KMeansState,
      *,
      tol: float,
      store_inner_errors: bool = False
  ) -> "KMeansOutput":
    errs = state.errors
    error = jnp.nanmin(jnp.where(errs == -1, jnp.nan, errs))
    # TODO(michal8): explain
    converged = jnp.logical_or(
        jnp.sum(errs == -1) > 0, (errs[-2] - errs[-1]) <= tol
    )
    return cls(
        centroids=state.centroids,
        assignment=state.assignment,
        inner_errors=errs if store_inner_errors else None,
        iteration=jnp.sum(errs > -1),
        error=error,
        converged=converged,
    )


def _random_init(
    geom: pointcloud.PointCloud, k: int, key: jnp.ndarray
) -> jnp.ndarray:
  ixs = jnp.arange(geom.shape[0])
  ixs = jax.random.choice(key, ixs, shape=(k,), replace=False)
  return geom.subset(ixs, None).x


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
    dists = geom.subset(ix, None).cost_matrix[0]
    return KPPState(key=next_key, centroids=centroids, centroid_dists=dists)

  def body_fn(
      iteration: int, const: Tuple[pointcloud.PointCloud, jnp.ndarray],
      state: KPPState, compute_error: bool
  ) -> KPPState:
    del compute_error
    key, next_key = jax.random.split(state.key, 2)
    geom, ixs = const
    # TODO(michalk8): check if needs to be normalized
    probs = state.centroid_dists  # / state.centroid_dists.sum()
    ixs = jax.random.choice(key, ixs, shape=(n_local_trials,), p=probs)
    geom = geom.subset(ixs, None)

    candidate_dists = jnp.minimum(geom.cost_matrix, state.centroid_dists)
    best_ix = jnp.argmin(candidate_dists.sum(1))

    centroids = state.centroids.at[iteration + 1].set(geom.x[best_ix])
    centroid_dists = candidate_dists[best_ix]

    return KPPState(
        key=next_key, centroids=centroids, centroid_dists=centroid_dists
    )

  if n_local_trials is None:
    n_local_trials = 2 + int(math.log(k))
  assert n_local_trials > 0, n_local_trials

  state = init_fn(geom, key)
  constants = (geom, jnp.arange(geom.shape[0]))
  state = fixed_point_loop.fixpoint_iter(
      lambda *_, **__: True,
      body_fn,
      min_iterations=k - 1,
      max_iterations=k - 1,
      inner_iterations=1,
      constants=constants,
      state=state
  )

  return state.centroids


@functools.partial(jax.vmap, in_axes=[0] + [None] * 9)
def _kmeans(
    key: jnp.ndarray,
    geom: pointcloud.PointCloud,
    k: int,
    weights: Optional[jnp.ndarray] = None,
    init: Init_t = "k-means++",
    n_local_trials: Optional[int] = None,
    tol: float = 1e-4,
    min_iter: int = 20,
    max_iter: int = 20,
    store_inner_errors: bool = False,
) -> KMeansOutput:

  def update_assignment(geom: pointcloud.PointCloud,
                        centroids: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    (x, _, *args), aux_data = geom.tree_flatten()
    cost_matrix = pointcloud.PointCloud.tree_unflatten(
        aux_data, [x, centroids] + args
    ).cost_matrix

    assignment = jnp.argmin(cost_matrix, axis=1)
    err = jnp.mean(
        cost_matrix[jnp.arange(len(assignment)), assignment] * weights
    )
    return assignment, err

  def update_centroids(state: KMeansState) -> jnp.ndarray:
    data = jax.ops.segment_sum(
        weighted_x, state.assignment, num_segments=k, unique_indices=True
    )
    centroids, ws = data[:, :-1], data[:, -1]
    return centroids / ws[:, None]

  def init_fn(init: Init_t) -> KMeansState:
    if init == "k-means++":
      init = functools.partial(_kmeans_plus_plus, n_local_trials=n_local_trials)
    elif init == "random":
      init = _random_init
    if not callable(init_fn):
      raise TypeError(
          f"Expected `init` to be 'k-means++', 'random' "
          f"or a callable, found `{init_fn!r}`."
      )

    centroids = init(geom, k, key)
    assignment, err = update_assignment(geom, centroids)
    prev_assignment = jnp.full_like(assignment, -1)
    errors = jnp.full((max_iter,), -1.).at[0].set(err)

    return KMeansState(
        centroids=centroids,
        prev_assignment=prev_assignment,
        assignment=assignment,
        errors=errors,
    )

  def cond_fn(iteration: int, const: Any, state: KMeansState) -> bool:
    del const
    errs = state.errors
    assignment_changed = jnp.any(state.prev_assignment != state.assignment)
    # below is always satisfied for `iteration=0`,
    # but the assignment condition never holds at `iteration=0`
    tol_not_satisfied = errs[iteration - 1] - errs[iteration] > tol
    return jnp.logical_or(tol_not_satisfied, assignment_changed)

  def body_fn(
      iteration: int, const: Any, state: KMeansState, compute_error: bool
  ) -> KMeansState:
    del compute_error, const

    centroids = update_centroids(state)
    assignment, err = update_assignment(geom, centroids)
    errors = state.errors.at[iteration + 1].set(err)

    return KMeansState(
        centroids=centroids,
        prev_assignment=state.assignment,
        assignment=assignment,
        errors=errors,
    )

  weighted_x = jnp.hstack([weights[:, None] * geom.x, weights[:, None]])
  state = fixed_point_loop.fixpoint_iter(
      cond_fn,
      body_fn,
      min_iterations=min_iter,
      max_iterations=max_iter - 1,
      inner_iterations=1,
      constants=None,
      state=init_fn(init)
  )
  return KMeansOutput.from_state(
      state, tol=tol, store_inner_errors=store_inner_errors
  )


def kmeans(
    geom: Union[jnp.ndarray, pointcloud.PointCloud],
    k: int,
    weights: Optional[jnp.ndarray] = None,
    init: Init_t = "k-means++",
    n_init: int = 10,
    n_local_trials: Optional[int] = None,
    tol: float = 1e-4,
    min_iter: int = 0,
    max_iter: int = 100,
    store_inner_errors: bool = False,
    seed: int = 0,
) -> KMeansOutput:
  if isinstance(geom, jnp.ndarray):
    geom = pointcloud.PointCloud(geom)
  assert geom.is_squared_euclidean

  if geom.is_online:
    # to allow materializing the cost matrix
    children, aux_data = geom.tree_flatten()
    aux_data["batch_size"] = None
    geom = type(geom).tree_unflatten(aux_data, children)

  # TODO(michalk8): consider normalizing?
  if weights is None:
    weights = jnp.ones(geom.shape[0])
  assert weights.shape == (geom.shape[0],)

  keys = jax.random.split(jax.random.PRNGKey(seed), n_init)
  out = _kmeans(
      keys, geom, k, weights, init, n_local_trials, tol, min_iter, max_iter,
      store_inner_errors
  )
  best_ix = jnp.argmin(out.error)
  return jax.tree_util.tree_map(lambda arr: arr[best_ix], out)
