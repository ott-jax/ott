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
import functools
import math
from typing import Callable, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import costs, pointcloud
from ott.math import fixed_point_loop

__all__ = ["k_means", "KMeansOutput"]

Init_t = Union[Literal["k-means++", "random"],
               Callable[[pointcloud.PointCloud, int, jnp.ndarray], jnp.ndarray]]


class KPPState(NamedTuple):  # noqa: D101
  rng: jax.Array
  centroids: jnp.ndarray
  centroid_dists: jnp.ndarray


class KMeansState(NamedTuple):  # noqa: D101
  centroids: jnp.ndarray
  prev_assignment: jnp.ndarray
  assignment: jnp.ndarray
  errors: jnp.ndarray
  center_shift: float


class KMeansConst(NamedTuple):  # noqa: D101
  geom: pointcloud.PointCloud
  x_weights: jnp.ndarray

  @property
  def x(self) -> jnp.ndarray:
    """Array of shape ``[n, ndim]`` containing the unweighted point cloud."""
    return self.geom.x

  @property
  def weighted_x(self):
    """Array of shape ``[n, ndim]`` containing the weighted point cloud."""
    return self.x_weights[:, :-1]

  @property
  def weights(self) -> jnp.ndarray:
    """Array of shape ``[n, 1]`` containing weights for each point."""
    return self.x_weights[:, -1:]


class KMeansOutput(NamedTuple):
  """Output of the :func:`~ott.tools.k_means.k_means` algorithm.

  Args:
    centroids: Array of shape ``[k, ndim]`` containing the centroids.
    assignment: Array of shape ``[n,]`` containing the labels.
    converged: Whether the algorithm has converged.
    iteration: The number of iterations run.
    error: (Weighted) sum of squared distances from each point to its closest
      center.
    inner_errors: Array of shape ``[max_iterations,]`` containing the ``error``
      at every iteration.
  """
  centroids: jnp.ndarray
  assignment: jnp.ndarray
  converged: bool
  iteration: int
  error: float
  inner_errors: Optional[jnp.ndarray]

  @classmethod
  def _from_state(
      cls,
      state: KMeansState,
      *,
      tol: float,
      store_inner_errors: bool = False
  ) -> "KMeansOutput":
    errs = state.errors
    mask = errs == -1
    error = jnp.nanmin(jnp.where(mask, jnp.nan, errs))

    assignment_same = jnp.all(state.prev_assignment == state.assignment)
    tol_satisfied = jnp.logical_or(jnp.any(mask), (errs[-2] - errs[-1]) <= tol)
    converged = jnp.logical_or(assignment_same, tol_satisfied)

    return cls(
        centroids=state.centroids,
        assignment=state.assignment.astype(int),
        converged=converged,
        iteration=jnp.sum(~mask),
        error=error,
        inner_errors=errs if store_inner_errors else None,
    )


def _random_init(
    geom: pointcloud.PointCloud, k: int, rng: jax.Array
) -> jnp.ndarray:
  ixs = jnp.arange(geom.shape[0])
  ixs = jax.random.choice(rng, ixs, shape=(k,), replace=False)
  return geom.subset(ixs, None).x


def _k_means_plus_plus(
    geom: pointcloud.PointCloud,
    k: int,
    rng: jax.Array,
    n_local_trials: Optional[int] = None,
) -> jnp.ndarray:

  def init_fn(geom: pointcloud.PointCloud, rng: jax.Array) -> KPPState:
    rng, next_rng = jax.random.split(rng, 2)
    ix = jax.random.choice(rng, jnp.arange(geom.shape[0]), shape=())
    centroids = jnp.full((k, geom.cost_rank), jnp.inf).at[0].set(geom.x[ix])
    dists = geom.subset([ix], None).cost_matrix[0]
    return KPPState(rng=next_rng, centroids=centroids, centroid_dists=dists)

  def body_fn(
      iteration: int, const: Tuple[pointcloud.PointCloud, jnp.ndarray],
      state: KPPState, compute_error: bool
  ) -> KPPState:
    del compute_error
    rng, next_rng = jax.random.split(state.rng, 2)
    geom, ixs = const

    # no need to normalize when `replace=True`
    probs = state.centroid_dists
    ixs = jax.random.choice(
        rng, ixs, shape=(n_local_trials,), p=probs, replace=True
    )
    geom = geom.subset(ixs, None)

    candidate_dists = jnp.minimum(geom.cost_matrix, state.centroid_dists)
    best_ix = jnp.argmin(candidate_dists.sum(1))

    centroids = state.centroids.at[iteration + 1].set(geom.x[best_ix])
    centroid_dists = candidate_dists[best_ix]

    return KPPState(
        rng=next_rng, centroids=centroids, centroid_dists=centroid_dists
    )

  if n_local_trials is None:
    n_local_trials = 2 + int(math.log(k))
  assert n_local_trials > 0, n_local_trials

  state = init_fn(geom, rng)
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


@functools.partial(jax.vmap, in_axes=[None, 0, 0, 0], out_axes=0)
def _reallocate_centroids(
    const: KMeansConst,
    ix: jnp.ndarray,
    centroid: jnp.ndarray,
    weight: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  is_empty = weight <= 0.0
  new_centroid = (1 - is_empty) * centroid + is_empty * const.x[ix]  # (ndim,)
  centroid_to_remove = is_empty * const.weighted_x[ix]  # (ndim,)
  weight_to_remove = is_empty * const.weights[ix]  # (1,)
  return new_centroid, jnp.concatenate([centroid_to_remove, weight_to_remove])


def _update_assignment(
    const: KMeansConst,
    centroids: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  (x, _, *args), aux_data = const.geom.tree_flatten()
  cost_matrix = type(
      const.geom
  ).tree_unflatten(aux_data, [x, centroids] + args).cost_matrix

  assignment = jnp.argmin(cost_matrix, axis=1)
  dist_to_centers = cost_matrix[jnp.arange(len(assignment)), assignment]
  return assignment, dist_to_centers


def _update_centroids(
    const: KMeansConst, k: int, assignment: jnp.ndarray,
    dist_to_centers: jnp.ndarray
) -> jnp.ndarray:
  # TODO(michalk8):
  # cannot put `k` into `const`, see https://github.com/ott-jax/ott/issues/129
  x_weights = jax.ops.segment_sum(const.x_weights, assignment, num_segments=k)
  centroids, ws = x_weights[:, :-1], x_weights[:, -1:]

  far_ixs = jnp.argsort(dist_to_centers)[-k:][::-1]
  centroids, to_remove = _reallocate_centroids(const, far_ixs, centroids, ws)
  to_remove = jax.ops.segment_sum(
      to_remove, assignment[far_ixs], num_segments=k
  )
  centroids -= to_remove[:, :-1]
  ws -= to_remove[:, -1:]

  return centroids * jnp.where(ws > 0.0, 1.0 / ws, 1.0)


@functools.partial(jax.vmap, in_axes=[0] + [None] * 9)
def _k_means(
    rng: jax.Array,
    geom: pointcloud.PointCloud,
    k: int,
    weights: Optional[jnp.ndarray] = None,
    init: Init_t = "k-means++",
    n_local_trials: Optional[int] = None,
    tol: float = 1e-4,
    min_iterations: int = 0,
    max_iterations: int = 300,
    store_inner_errors: bool = False,
) -> KMeansOutput:

  def init_fn(init: Init_t) -> KMeansState:
    if init == "k-means++":
      init = functools.partial(
          _k_means_plus_plus, n_local_trials=n_local_trials
      )
    elif init == "random":
      init = _random_init
    if not callable(init):
      raise TypeError(
          f"Expected `init` to be 'k-means++', 'random' "
          f"or a callable, found `{init_fn!r}`."
      )

    centroids = init(geom, k, rng)
    if centroids.shape != (k, geom.cost_rank):
      raise ValueError(
          f"Expected initial centroids to have shape "
          f"`{k, geom.cost_rank}`, found `{centroids.shape}`."
      )
    n = geom.shape[0]
    # TODO(michalk8): find a better solution for the below error
    # not using floats for the assignment when `fixpoint_iter_backprop` is used:

    # .../jax/_src/dtypes.py:370:
    # .0 = <set_iterator object at 0x7f4d002a1dc0>
    # >   CUB = set.intersection(*(UB[n] for n in N))
    # E   jax._src.traceback_util.UnfilteredStackTrace:
    #   KeyError: dtype([('float0', 'V')])
    # E   The stack trace below excludes JAX-internal frames.
    # E   The preceding is the original exception that occurred, unmodified.
    prev_assignment = jnp.full((n,), -2.0)
    assignment = jnp.full((n,), -1.0)
    errors = jnp.full((max_iterations,), -1.0)

    return KMeansState(
        centroids=centroids,
        prev_assignment=prev_assignment,
        assignment=assignment,
        center_shift=jnp.inf,
        errors=errors,
    )

  def cond_fn(iteration: int, const: KMeansConst, state: KMeansState) -> bool:
    del iteration, const
    assignment_not_same = jnp.any(state.prev_assignment != state.assignment)
    tol_not_satisfied = state.center_shift > tol
    return jnp.logical_and(assignment_not_same, tol_not_satisfied)

  def body_fn(
      iteration: int, const: KMeansConst, state: KMeansState,
      compute_error: bool
  ) -> KMeansState:
    del compute_error

    assignment, dist_to_centers = _update_assignment(const, state.centroids)
    centroids = _update_centroids(const, k, assignment, dist_to_centers)
    err = jnp.sum(const.weights[:, 0] * dist_to_centers)
    center_shift = jnp.linalg.norm(state.centroids - centroids, ord="fro") ** 2

    return KMeansState(
        centroids=centroids,
        prev_assignment=state.assignment,
        assignment=assignment.astype(float),
        center_shift=center_shift,
        errors=state.errors.at[iteration].set(err)
    )

  def finalize_fn(const: KMeansConst, state: KMeansState) -> KMeansState:
    last_iter = jnp.sum(state.errors != -1) - 1

    assignment, dist_to_centers = _update_assignment(const, state.centroids)
    err = jnp.sum(const.weights[:, 0] * dist_to_centers)

    return state._replace(
        assignment=assignment.astype(float),
        errors=state.errors.at[last_iter].set(err)
    )

  force_scan = min_iterations == max_iterations
  fixpoint_fn = (  # prefer auto-diff if possible
      fixed_point_loop.fixpoint_iter if force_scan else
      fixed_point_loop.fixpoint_iter_backprop
  )
  x_weights = jnp.hstack([weights[:, None] * geom.x, weights[:, None]])
  const = KMeansConst(geom, x_weights)

  state = fixpoint_fn(
      cond_fn,
      body_fn,
      min_iterations=min_iterations,
      max_iterations=max_iterations,
      inner_iterations=1,
      constants=const,
      state=init_fn(init)
  )
  state = jax.lax.cond(
      jnp.all(state.prev_assignment == state.assignment), (lambda _, s: s),
      finalize_fn, const, state
  )

  return KMeansOutput._from_state(
      state, tol=tol, store_inner_errors=store_inner_errors
  )


def k_means(
    geom: Union[jnp.ndarray, pointcloud.PointCloud],
    k: int,
    weights: Optional[jnp.ndarray] = None,
    init: Init_t = "k-means++",
    n_init: int = 10,
    n_local_trials: Optional[int] = None,
    tol: float = 1e-4,
    min_iterations: int = 0,
    max_iterations: int = 300,
    store_inner_errors: bool = False,
    rng: Optional[jax.Array] = None,
) -> KMeansOutput:
  r"""K-means clustering using Lloyd's algorithm :cite:`lloyd:82`.

  Args:
    geom: Point cloud of shape ``[n, ndim]`` to cluster. If passed as an array,
      :class:`~ott.geometry.costs.SqEuclidean` cost is assumed.
    k: The number of clusters.
    weights: The weights of input points. These weights are considered when
      computing the centroids and inertia. If ``None``, use uniform weights.
    init: Initialization method. Can be one of the following:

      - **'k-means++'** - select initial centroids that are
        :math:`\mathcal{O}(\log k)`-optimal :cite:`arthur:07`.
      - **'random'** - randomly select ``k`` points from the ``geom``.
      - :func:`callable` - a function which takes the point cloud, the number of
        clusters and a random key and returns the centroids as an array of shape
        ``[k, ndim]``.

    n_init: Number of times k-means will run with different initial seeds.
    n_local_trials: Number of local trials when ``init = 'k-means++'``.
      If ``None``, :math:`2 + \lfloor log(k) \rfloor` is used.
    tol: Relative tolerance with respect to the Frobenius norm of the centroids'
      shift between two consecutive iterations.
    min_iterations: Minimum number of iterations.
    max_iterations: Maximum number of iterations.
    store_inner_errors: Whether to store the errors (inertia) at each iteration.
    rng: Random key for seeding the initializations.

  Returns:
    The k-means clustering.
  """
  assert geom.shape[
      0] >= k, f"Cannot cluster `{geom.shape[0]}` points into `{k}` clusters."
  if isinstance(geom, jnp.ndarray):
    geom = pointcloud.PointCloud(geom)
  if isinstance(geom.cost_fn, costs.Cosine):
    geom = geom._cosine_to_sqeucl()
  assert geom.is_squared_euclidean
  rng = utils.default_prng_key(rng)

  if geom.is_online:
    # to allow materializing the cost matrix
    children, aux_data = geom.tree_flatten()
    aux_data["batch_size"] = None
    geom = type(geom).tree_unflatten(aux_data, children)

  if weights is None:
    weights = jnp.ones(geom.shape[0])
  assert weights.shape == (geom.shape[0],)

  rngs = jax.random.split(rng, n_init)
  out = _k_means(
      rngs, geom, k, weights, init, n_local_trials, tol, min_iterations,
      max_iterations, store_inner_errors
  )
  best_ix = jnp.argmin(out.error)
  return jax.tree_util.tree_map(lambda arr: arr[best_ix], out)
