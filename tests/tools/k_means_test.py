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
import os
import sys
from typing import Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, pointcloud
from ott.tools import k_means
from sklearn import datasets
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.cluster._k_means_common import _is_same_clustering


def make_blobs(
    *args: Any,
    cost_fn: Optional[Literal["sqeucl", "cosine"]] = None,
    **kwargs: Any
) -> Tuple[Union[jnp.ndarray, pointcloud.PointCloud], jnp.ndarray, jnp.ndarray]:
  X, y, c = datasets.make_blobs(*args, return_centers=True, **kwargs)
  X, y, c = jnp.asarray(X), jnp.asarray(y), jnp.asarray(c)
  if cost_fn is None:
    pass
  elif cost_fn == "sqeucl":
    X = pointcloud.PointCloud(X, cost_fn=costs.SqEuclidean())
  elif cost_fn == "cosine":
    X = pointcloud.PointCloud(X, cost_fn=costs.Cosine())
  else:
    raise NotImplementedError(cost_fn)

  return X, y, c


def compute_assignment(
    x: jnp.ndarray,
    centers: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, float]:
  if weights is None:
    weights = jnp.ones(x.shape[0])
  cost_matrix = pointcloud.PointCloud(x, centers).cost_matrix
  assignment = jnp.argmin(cost_matrix, axis=1)
  dist_to_centers = cost_matrix[jnp.arange(len(assignment)), assignment]

  return assignment, jnp.sum(weights * dist_to_centers)


class TestKmeansPlusPlus:

  @pytest.mark.fast.with_args("n_local_trials", [None, 1, 5], only_fast=-1)
  def test_n_local_trials(self, rng: jax.random.PRNGKeyArray, n_local_trials):
    n, k = 150, 4
    rng1, rng2 = jax.random.split(rng)
    geom, _, c = make_blobs(
        n_samples=n, centers=k, cost_fn="sqeucl", random_state=0
    )
    centers1 = k_means._k_means_plus_plus(geom, k, rng1, n_local_trials)
    centers2 = k_means._k_means_plus_plus(geom, k, rng2, 20)

    shift1 = jnp.linalg.norm(centers1 - c, ord="fro") ** 2
    shift2 = jnp.linalg.norm(centers2 - c, ord="fro") ** 2

    assert shift1 > shift2

  @pytest.mark.fast.with_args("k", [4, 5, 10], only_fast=0)
  def test_matches_sklearn(self, rng: jax.random.PRNGKeyArray, k: int):
    ndim = 2
    geom, _, _ = make_blobs(
        n_samples=200,
        centers=k,
        n_features=ndim,
        cost_fn="sqeucl",
        random_state=0
    )
    gt_centers, _ = kmeans_plusplus(np.asarray(geom.x), k, random_state=1)
    pred_centers = k_means._k_means_plus_plus(geom, k, rng)

    _, gt_inertia = compute_assignment(geom.x, gt_centers)
    _, pred_inertia = compute_assignment(geom.x, pred_centers)

    assert pred_centers.shape == (k, ndim)
    np.testing.assert_array_equal(
        pred_centers.max(axis=0) <= geom.x.max(axis=0), True
    )
    np.testing.assert_array_equal(
        pred_centers.min(axis=0) >= geom.x.min(axis=0), True
    )
    # the largest was 70.56378
    assert jnp.abs(pred_inertia - gt_inertia) <= 100

  def test_initialization_differentiable(self, rng: jax.random.PRNGKeyArray):

    def callback(x: jnp.ndarray) -> float:
      geom = pointcloud.PointCloud(x)
      centers = k_means._k_means_plus_plus(geom, k=3, rng=rng)
      _, inertia = compute_assignment(x, centers)
      return inertia

    X, _, _ = make_blobs(n_samples=34, random_state=0)
    fun = jax.value_and_grad(callback)
    ineria1, grad = fun(X)
    ineria2, _ = fun(X - 0.1 * grad)

    assert ineria2 < ineria1


class TestKmeans:

  @pytest.mark.fast()
  @pytest.mark.parametrize("k", [1, 6])
  def test_k_means_output(self, rng: jax.random.PRNGKeyArray, k: int):
    max_iter, ndim = 10, 4
    geom, gt_assignment, _ = make_blobs(
        n_samples=50, n_features=ndim, centers=k, random_state=42
    )
    gt_assignment = np.array(gt_assignment)

    res = k_means.k_means(
        geom, k, max_iterations=max_iter, store_inner_errors=False, rng=rng
    )
    pred_assignment = np.array(res.assignment)

    assert res.centroids.shape == (k, ndim)
    assert res.converged
    assert res.error >= 0.
    assert res.inner_errors is None
    assert _is_same_clustering(pred_assignment, gt_assignment, k)

  @pytest.mark.fast()
  def test_k_means_simple_example(self):
    expected_labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
    expected_centers = np.asarray([[0.75, 1], [0.25, 0]])

    x = jnp.asarray([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])
    init = lambda *_: jnp.array([[0.5, 0.5], [3, 3]])

    res = k_means.k_means(x, k=2, init=init)

    np.testing.assert_array_equal(res.assignment, expected_labels)
    np.testing.assert_allclose(res.centroids, expected_centers)
    np.testing.assert_allclose(res.error, 0.25)
    assert res.iteration == 3

  @pytest.mark.fast.with_args(
      "init",
      ["k-means++", "random", "callable", "wrong-callable"],
      only_fast=1,
  )
  def test_init_method(self, rng: jax.random.PRNGKeyArray, init: str):
    if init == "callable":
      init_fn = lambda geom, k, _: geom.x[:k]
    elif init == "wrong-callable":
      init_fn = lambda geom, k, _: geom.x[:k + 1]
    else:
      init_fn = init

    k = 3
    geom, _, _ = make_blobs(n_samples=50, centers=k + 1)
    if init == "wrong-callable":
      with pytest.raises(ValueError, match=r"Expected initial centroids"):
        _ = k_means.k_means(geom, k, init=init_fn)
    else:
      _ = k_means.k_means(geom, k, init=init_fn)

  def test_k_means_plus_plus_better_than_random(
      self, rng: jax.random.PRNGKeyArray
  ):
    k = 5
    rng1, rng2 = jax.random.split(rng, 2)
    geom, _, _ = make_blobs(n_samples=50, centers=k, random_state=10)

    res_random = k_means.k_means(geom, k, init="random", rng=rng1)
    res_kpp = k_means.k_means(geom, k, init="k-means++", rng=rng2)

    assert res_random.converged
    assert res_kpp.converged
    assert res_kpp.iteration < res_random.iteration
    assert res_kpp.error <= res_random.error

  def test_larger_n_init_helps(self, rng: jax.random.PRNGKeyArray):
    k = 10
    geom, _, _ = make_blobs(n_samples=150, centers=k, random_state=0)

    res = k_means.k_means(geom, k, n_init=3, rng=rng)
    res_larger_n_init = k_means.k_means(geom, k, n_init=20, rng=rng)

    assert res_larger_n_init.error < res.error

  @pytest.mark.parametrize("max_iter", [8, 16])
  def test_store_inner_errors(
      self, rng: jax.random.PRNGKeyArray, max_iter: int
  ):
    ndim, k = 10, 4
    geom, _, _ = make_blobs(
        n_samples=40, n_features=ndim, centers=k, random_state=43
    )

    res = k_means.k_means(
        geom, k, max_iterations=max_iter, store_inner_errors=True, rng=rng
    )

    errors = res.inner_errors
    assert errors.shape == (max_iter,)
    assert res.iteration == jnp.sum(errors > 0.)
    # check if error is decreasing
    np.testing.assert_array_equal(jnp.diff(errors[::-1]) >= 0., True)

  def test_strict_tolerance(self, rng: jax.random.PRNGKeyArray):
    k = 11
    geom, _, _ = make_blobs(n_samples=200, centers=k, random_state=39)

    res = k_means.k_means(geom, k=k, tol=1., rng=rng)
    res_strict = k_means.k_means(geom, k=k, tol=0., rng=rng)

    assert res.converged
    assert res_strict.converged
    assert res.iteration < res_strict.iteration

  @pytest.mark.parametrize(
      "tol", [1e-3, 0.], ids=["weak-convergence", "strict-convergence"]
  )
  def test_convergence_force_scan(
      self, rng: jax.random.PRNGKeyArray, tol: float
  ):
    k, n_iter = 9, 20
    geom, _, _ = make_blobs(n_samples=100, centers=k, random_state=37)

    res = k_means.k_means(
        geom,
        k=k,
        tol=tol,
        min_iterations=n_iter,
        max_iterations=n_iter,
        store_inner_errors=True,
        rng=rng
    )

    assert res.converged
    assert res.iteration == n_iter
    np.testing.assert_array_equal(res.inner_errors == -1, False)

  def test_k_means_min_iterations(self, rng: jax.random.PRNGKeyArray):
    k, min_iter = 8, 12
    geom, _, _ = make_blobs(n_samples=160, centers=k, random_state=38)

    res = k_means.k_means(
        geom,
        k - 2,
        store_inner_errors=True,
        min_iterations=min_iter,
        max_iterations=20,
        tol=0.,
        rng=rng
    )

    assert res.converged
    assert jnp.sum(res.inner_errors != -1) >= min_iter

  def test_weight_scaling_effects_only_inertia(
      self, rng: jax.random.PRNGKeyArray
  ):
    k = 10
    rng1, rng2 = jax.random.split(rng)
    geom, _, _ = make_blobs(n_samples=130, centers=k, random_state=3)
    weights = jnp.abs(jax.random.normal(rng1, shape=(geom.shape[0],)))
    weights_scaled = weights / jnp.sum(weights)

    res = k_means.k_means(geom, k=k - 1, weights=weights)
    res_scaled = k_means.k_means(geom, k=k - 1, weights=weights_scaled)

    np.testing.assert_allclose(
        res.centroids, res_scaled.centroids, rtol=1e-5, atol=1e-5
    )
    assert _is_same_clustering(
        np.array(res.assignment), np.array(res_scaled.assignment), k
    )
    np.testing.assert_allclose(
        res.error, res_scaled.error * jnp.sum(weights), rtol=1e-3, atol=1e-3
    )

  @pytest.mark.fast()
  def test_empty_weights(self, rng: jax.random.PRNGKeyArray):
    n, ndim, k, d = 20, 2, 3, 5.
    gen = np.random.RandomState(0)
    x = gen.normal(size=(n, ndim))
    x[:, 0] += d
    x[:, 1] += d
    y = gen.normal(size=(n, ndim))
    y[:, 0] -= d
    y[:, 1] -= d
    z = gen.normal(size=(n, ndim))
    z[:, 0] += d
    z[:, 1] -= d
    w = gen.normal(size=(n, ndim))
    w[:, 0] -= d
    w[:, 1] += d
    x = jnp.concatenate((x, y, z, w))
    # ignore `x` by setting its weights to 0
    weights = jnp.ones((x.shape[0],)).at[:n].set(0.)

    expected_centroids = jnp.stack([w.mean(0), z.mean(0), y.mean(0)])
    res = k_means.k_means(x, k=k, weights=weights, rng=rng)

    cost = pointcloud.PointCloud(res.centroids, expected_centroids).cost_matrix
    ixs = jnp.argmin(cost, axis=1)

    np.testing.assert_array_equal(jnp.sort(ixs), jnp.arange(k))

    total_shift = jnp.sum(cost[jnp.arange(k), ixs])
    np.testing.assert_allclose(total_shift, 0., rtol=1e-3, atol=1e-3)

  def test_cosine_cost_fn(self):
    k = 4
    geom, _, _ = make_blobs(n_samples=75)
    geom_scaled = pointcloud.PointCloud(geom * 10., cost_fn=costs.Cosine())
    geom = pointcloud.PointCloud(geom, cost_fn=costs.Cosine())

    res_scaled = k_means.k_means(geom_scaled, k=k)
    res = k_means.k_means(geom, k=k)

    np.testing.assert_allclose(
        res_scaled.error, res.error, rtol=1e-5, atol=1e-5
    )
    assert _is_same_clustering(
        np.array(res_scaled.assignment), np.array(res.assignment), k
    )

  @pytest.mark.fast.with_args("init", ["k-means++", "random"], only_fast=0)
  def test_k_means_jitting(
      self, rng: jax.random.PRNGKeyArray, init: Literal["k-means++", "random"]
  ):

    def callback(x: jnp.ndarray) -> k_means.KMeansOutput:
      return k_means.k_means(
          x, k=k, init=init, store_inner_errors=True, rng=rng
      )

    k = 7
    x, _, _ = make_blobs(n_samples=150, centers=k, random_state=0)
    res = jax.jit(callback)(x)
    res_jit: k_means.KMeansOutput = jax.jit(callback)(x)

    np.testing.assert_allclose(
        res.centroids, res_jit.centroids, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(res.assignment, res_jit.assignment)
    np.testing.assert_allclose(res.error, res_jit.error, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        res.inner_errors, res_jit.inner_errors, rtol=1e-5, atol=1e-5
    )
    assert res.iteration == res_jit.iteration
    assert res.converged == res_jit.converged

  @pytest.mark.skipif(
      sys.platform == "darwin" and os.environ.get("CI", "false") == "true",
      reason="Fails on macOS CI."
  )
  @pytest.mark.parametrize(("jit", "force_scan"), [(True, False),
                                                   (False, True)],
                           ids=["jit-while-loop", "nojit-for-loop"])
  def test_k_means_differentiability(
      self, rng: jax.random.PRNGKeyArray, jit: bool, force_scan: bool
  ):

    def inertia(x: jnp.ndarray, w: jnp.ndarray) -> float:
      return k_means.k_means(
          x,
          k=k,
          weights=w,
          min_iterations=20 if force_scan else 1,
          max_iterations=20,
          rng=rng1,
      ).error

    k, eps, tol = 4, 1e-3, 1e-3
    x, _, _ = make_blobs(n_samples=150, centers=k, random_state=41)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    w = jnp.abs(jax.random.normal(rng2, (x.shape[0],)))

    v_x = jax.random.normal(rng3, shape=x.shape)
    v_x = (v_x / jnp.linalg.norm(v_x, axis=-1, keepdims=True)) * eps
    v_w = jax.random.normal(rng4, shape=w.shape) * eps
    v_w = (v_w / jnp.linalg.norm(v_w, axis=-1, keepdims=True)) * eps

    grad_fn = jax.grad(inertia, (0, 1))
    if jit:
      grad_fn = jax.jit(grad_fn)
    grad_x, grad_w = grad_fn(x, w)

    expected = inertia(x + v_x, w) - inertia(x - v_x, w)
    actual = 2 * jnp.vdot(v_x, grad_x)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    expected = inertia(x, w + v_w) - inertia(x, w - v_w)
    actual = 2 * jnp.vdot(v_w, grad_w)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

  @pytest.mark.parametrize("tol", [1e-3, 0.])
  @pytest.mark.parametrize(("n", "k"), [(37, 4), (128, 6)])
  def test_clustering_matches_sklearn(
      self, rng: jax.random.PRNGKeyArray, n: int, k: int, tol: float
  ):
    x, _, _ = make_blobs(n_samples=n, centers=k, random_state=41)

    res_kmeans = KMeans(n_clusters=k, n_init=20, tol=tol, random_state=0).fit(x)
    res_ours = k_means.k_means(x, k, n_init=20, tol=tol, rng=rng)
    gt_labels = res_kmeans.labels_
    pred_labels = np.array(res_ours.assignment)

    assert _is_same_clustering(pred_labels, gt_labels, k)
    np.testing.assert_allclose(
        res_ours.error, res_kmeans.inertia_, rtol=1e-3, atol=1e-3
    )
