# Copyright 2021 CR.Sparse Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The implementation in this file is based on the example provided by
Sabrina J. Mielke
in https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA#scrollTo=LJyoi46rIJr7
"""

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jax.numpy.linalg import norm

from ott.core import fixed_point_loop
from ott.geometry import pointcloud


class KMeansState(NamedTuple):
  """The state for K-means algorithm"""

  centroids: jnp.ndarray
  """Current set of centroids"""
  assignment: jnp.ndarray
  """Current assignment of points to centroids"""
  distortion: float
  """ Current mean distance"""
  prev_distortion: float
  """ Previous mean distance"""
  iterations: int
  """The number of iterations it took to complete"""


class KMeansSolution(NamedTuple):
  """The solution for K-means algorithm"""

  centroids: jnp.ndarray
  """Current set of centroids"""
  assignment: jnp.ndarray
  """Current assignment of points to centroids"""
  distortion: float
  """ Current mean distance"""
  key: jnp.ndarray
  """ The PRNG key seed for the k-means run with least distortion"""
  iterations: int
  """The number of iterations it took to complete"""


def find_nearest(point, centroids):
  """Returns the index of the nearest centroid for a specific point
    Args:
        point (jax.numpy.ndarray) : A specific point
        centroids (jax.numpy.ndarray) : An array of centroids
    Returns:
        (int) : The index of the nearest centroid
    """
  return jnp.argmin(vmap(norm)(centroids - point))


find_nearest_jit = jit(find_nearest)


def find_assignment(points, centroids):
  """Finds the assignment of each point to a specific centroid
    Args:
        points (jax.numpy.ndarray) : Each row of the points matrix is a point.
        centroids (jax.numpy.ndarray) : An array of centroids
    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple consisting of
        #. An assignment array of each point to a cluster
        #. Distance of each point from corresponding cluster centroid
    """
  assignment = vmap(lambda point: find_nearest(point, centroids))(points)
  errors = centroids[assignment, :] - points
  distances = vmap(norm)(errors)
  return assignment, distances


find_assignment_jit = jit(find_assignment)


def assignment_counts(assignment, k):
  """Returns the number of points in each cluster based on the current assignment
    If a cluster has no points, we return 1.
    """
  return ((assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis]
          ).sum(axis=1, keepdims=True).clip(min=1))


def find_new_centroids(assignment, points, k):
  """Finds new centroids based on current assignment
    Args:
        assignment (jax.numpy.ndarray) : current assignment of each point to a specific cluster
        points (jax.numpy.ndarray) : Each row of the points matrix is a point.
        k (int): The number of clusters
    """
  counts = assignment_counts(assignment, k)
  new_centroids = (
      jnp.sum(
          jnp.where(
              # axes: (data points, clusters, data dimension)
              assignment[:, jnp.newaxis, jnp.newaxis]
              == jnp.arange(k)[jnp.newaxis, :, jnp.newaxis],
              points[:, jnp.newaxis, :],
              0.0,
          ),
          axis=0,
      ) / counts
  )
  return new_centroids


find_new_centroids_jit = jit(find_new_centroids, static_argnums=(2,))


def kmeans_with_seed(key, points, k, thresh=1e-5, max_iters=100):
  """Runs the k-means algorithm for a specific random initialization
    Args:
        key: a PRNG key used as the random key for choosing initial centroids
        points (jax.numpy.ndarray): Each row of the points matrix is a point.
        k (int): The number of clusters
        thresh (float): Convergence threshold on change in distortion
        max_iters (int): Maximum number of iterations for k-means algorithm
    Returns:
        (KMeansState): A named tuple consisting of:
        centroids for each cluster, assignment of each point to a cluster,
        current distorition, previous distortion, number of iterations
        for convergence.
    """
  # number of points
  n = points.shape[0]

  def init():
    # select k points as initial centroids randomly
    indices = random.permutation(key, jnp.arange(n))[:k]
    # the initial centroids
    centroids = points[indices, :]
    # assign all points to centroids and compute distances
    assignment, distances = find_assignment(points, centroids)
    distortion = jnp.mean(distances)
    # algorithm state
    return KMeansState(
        centroids=centroids,
        assignment=assignment,
        distortion=distortion,
        prev_distortion=jnp.inf,
        iterations=0,
    )

  def body(state):
    # update centroids
    centroids = find_new_centroids(state.assignment, points, k)
    # update assignment
    assignment, distances = find_assignment(points, centroids)
    # mean distance
    distortion = jnp.mean(distances)
    # algorithm state
    return KMeansState(
        centroids=centroids,
        assignment=assignment,
        distortion=distortion,
        prev_distortion=state.distortion,
        iterations=state.iterations + 1,
    )

  def cond(state):
    # check if the mean distance has updated enough
    gap = state.prev_distortion - state.distortion
    # print(state.prev_distortion, state.distortion, gap, thresh, gap > thresh)
    return jnp.logical_and(gap > thresh, state.iterations < max_iters)

  # state = init()
  # while cond(state):
  #     state = body(state)
  state = lax.while_loop(cond, body, init())
  return state


kmeans_with_seed_jit = jit(kmeans_with_seed, static_argnums=(2, 3))


def kmeans(key, points, k, iter=20, thresh=1e-5, max_iters=100):
  r"""Clusters points using k-means algorithm
    Args:
        key: a PRNG key used as the random key
        points (jax.numpy.ndarray): Each row of the points matrix is a point.
          From the statistical point of view, each row is an observation vector
          and each column is a feature.
        k (int): The number of clusters
        iter (int): The number of times k-means will be restarted with
          different seeds. The result with least amount of distortion is returned.
        thresh (float): Convergence threshold on change in distortion
        max_iters (int): Maximum number of iterations for each replicate of k-means algorithm
    Returns:
        (KMeansSolution): A named tuple consisting of:
        * centroids : centroid for each cluster
        * assignment: assignment of each point to a cluster
        * distortion: distortion after current assignment
        * key: The PRNG key seed for the k-means run with the least distortion
        * iterations: number of iterations taken in convergence
    Let the k centroids be :math:`m_1, m_2, \dots, m_k`.
    Let the n points be :math:`x_1, x_2, \dots, x_n`.
    Let the assignment of i-th point to j-th cluster be given by
    :math:`a_1, a_2, \dots, a_n` where :math:`1 \leq a_i = j \leq k`.
    Then the distance of i-th point from its centroid is given by:
    .. math::
        d_i = \| x_i - m_{a_i} \|_2
    The distortion is given by the mean of all the distances.
    """
  # keys for each restart of kmeans algorithm
  keys = random.split(key, iter)
  # individual run of k-means algorithm
  kmeans_core = lambda key: kmeans_with_seed(
      key, points, k, thresh=thresh, max_iters=max_iters
  )
  # Run all restarts of kmeans using vmap
  results = vmap(kmeans_core, 0, 0)(keys)
  # Find the run with the least distortion
  i = jnp.argmin(results.distortion)
  return KMeansSolution(
      centroids=results.centroids[i],
      assignment=results.assignment[i],
      distortion=results.distortion[i],
      key=keys[i],
      iterations=results.iterations[i],
  )


def _random_init(
    geom: pointcloud.PointCloud, k: int, key: jnp.ndarray
) -> jnp.ndarray:
  ixs = jnp.arange(geom.shape[0])
  return jax.random.choice(key, ixs, shape=(k,), replace=False)


def _kmeans_plus_plus(
    geom: pointcloud.PointCloud, k: int, key: jnp.ndarray
) -> jnp.ndarray:
  pass


def _post_init(
    geom: pointcloud.PointCloud, centroids: jnp.ndarray, weights: jnp.ndarray
) -> KMeansState:
  geom = geom.subset(None, centroids, batch_size=None)
  cost_matrix = geom.cost_matrix  # (n, k)
  assignment = jnp.argmin(cost_matrix, axis=1)  # (n,)
  # weights are normalized
  distortion = jnp.sum(jnp.min(cost_matrix, axis=1) * weights)  # ()

  return KMeansState(
      centroids=centroids,
      assignment=assignment,
      distortion=distortion,
      prev_distortion=jnp.inf,
      iterations=0
  )


class KMeansConstants(NamedTuple):
  geom: pointcloud.PointCloud
  weights: jnp.ndarray
  k: int
  tol: float


class KMeansState2(NamedTuple):
  """The state for K-means algorithm"""

  centroids: jnp.ndarray
  assignment: jnp.ndarray
  distortions: jnp.ndarray

  def set(self, **kwargs: Any) -> 'KMeansState2':
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)


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
) -> KMeansState2:

  def init_fn(geom: pointcloud.PointCloud) -> KMeansState2:
    if init_random:
      centroids = _random_init(geom, k=k, key=key)
    else:
      centroids = _kmeans_plus_plus(geom, k=k, key=key)
    geom = geom.subset(None, centroids, batch_size=None)
    cost_matrix = geom.cost_matrix  # (n, k)
    assignment = jnp.argmin(cost_matrix, axis=1)  # (n,)
    # weights are normalized
    distortion = jnp.sum(jnp.min(cost_matrix, axis=1) * weights)  # ()
    distortions = jnp.full((max_iter,), -1.).at[0].set(distortion)

    return KMeansState2(
        centroids=centroids,
        assignment=assignment,
        distortions=distortions,
    )

  def cond_fn(
      iteration: int, const: KMeansConstants, state: KMeansState2
  ) -> bool:
    d = state.distortions
    return d[iteration - 1] - d[iteration] > const.tol

  def body_fn(
      iteration: int, const: KMeansConstants, state: KMeansState2,
      compute_error: bool
  ) -> KMeansState2:
    del compute_error
    d = state.distortions.at[iteration].set(-2)
    return state.set(distortions=d)

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
  return state


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
) -> KMeansSolution:
  # TODO(michalk8): center PC?
  keys = jax.random.split(jax.random.PRNGKey(seed), n_iter)

  if weights is None:
    weights = jnp.ones(geom.shape[0])
  weights /= jnp.sum(weights)
  assert weights.shape == (geom.shape[0],)

  results = jax.vmap(
      _kmeans, in_axes=[0] + [None] * 7
  )(keys, geom, k, tol, weights, init_random, min_iter, max_iter)

  return results

  i = jnp.argmin(results.distortion)
  return KMeansSolution(
      centroids=results.centroids[i],
      assignment=results.assignment[i],
      distortion=results.distortion[i],
      key=keys[i],
      iterations=results.iterations[i],
  )
