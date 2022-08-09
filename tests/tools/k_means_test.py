from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn import datasets
from sklearn.cluster._kmeans import kmeans_plusplus
from typing_extensions import Literal

from ott.geometry import costs, pointcloud
from ott.tools import k_means


def make_blobs(
    *args: Any,
    cost_fn: Optional[Literal['sqeucl', 'cosine']] = None,
    **kwargs: Any
) -> Tuple[Union[jnp.ndarray, pointcloud.PointCloud], jnp.ndarray, jnp.ndarray]:
  X, y, c = datasets.make_blobs(*args, return_centers=True, **kwargs)
  X, y, c = jnp.asarray(X), jnp.asarray(y), jnp.asarray(c)
  if cost_fn is None:
    pass
  elif cost_fn == 'sqeucl':
    X = pointcloud.PointCloud(X, cost_fn=costs.Euclidean())
  elif cost_fn == 'cosine':
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

  @pytest.mark.parametrize("n_local_trials", [None, 1, 5])
  def test_n_local_trials(self, rng: jnp.ndarray, n_local_trials):
    n, k = 150, 4
    key1, key2 = jax.random.split(rng)
    geom, _, c = make_blobs(
        n_samples=n, centers=k, cost_fn='sqeucl', random_state=0
    )
    centers1 = k_means._k_means_plus_plus(geom, k, key1, n_local_trials)
    centers2 = k_means._k_means_plus_plus(geom, k, key2, 20)

    shift1 = ((centers1 - c) ** 2).sum()
    shift2 = ((centers2 - c) ** 2).sum()

    assert shift1 > shift2

  @pytest.mark.parametrize("k", [4, 5, 10])
  def test_matches_sklearn(self, rng: jnp.ndarray, k: int):
    ndim = 2
    geom, _, _ = make_blobs(
        n_samples=200,
        centers=k,
        n_features=ndim,
        cost_fn='sqeucl',
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

  def test_initialization_differentiable(self, rng: jnp.ndarray):

    def callback(x: jnp.ndarray) -> float:
      geom = pointcloud.PointCloud(x)
      centers = k_means._k_means_plus_plus(geom, k=3, key=rng)
      _, inertia = compute_assignment(x, centers)
      return inertia

    X, _, _ = make_blobs(n_samples=34, random_state=0)
    fun = jax.value_and_grad(callback)
    ineria1, grad = fun(X)
    ineria2, _ = fun(X - 0.1 * grad)

    assert ineria2 < ineria1


class TestKmeans:

  @pytest.mark.parametrize("k", [1, 6])
  def test_k_means_output(self, rng: jnp.ndarray, k: int):
    max_iter, ndim = 10, 4
    geom, _, _ = make_blobs(
        n_samples=50, n_features=ndim, centers=k, random_state=42
    )
    res = k_means.k_means(
        geom, k, max_iterations=max_iter, store_inner_errors=True, key=rng
    )

    assert res.centroids.shape == (k, ndim)
    assert res.converged
    assert res.error >= 0.
    errors = res.inner_errors
    assert errors.shape == (max_iter,)
    assert res.iteration == jnp.sum(errors > 0.)
    # check if error is decreasing
    np.testing.assert_array_equal(jnp.diff(errors[::-1]) >= 0., True)

  @pytest.mark.parametrize(
      "init", ["k-means++", "random", "callable", "wrong-callable"]
  )
  def test_init_method(self, rng: jnp.ndarray, init: str):
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

  def test_k_means_plus_plus_better_than_random(self, rng: jnp.ndarray):
    k = 5
    key1, key2 = jax.random.split(rng, 2)
    geom, _, _ = make_blobs(n_samples=50, centers=k)

    res_random = k_means.k_means(geom, k, init="random", key=key1)
    res_kpp = k_means.k_means(geom, k, init="k-means++", key=key2)

    assert res_kpp.error < res_random.error

  @pytest.mark.parametrize("n_init", [5, 7])
  def test_larger_n_init_should_help(self, rng: jnp.ndarray, n_init: int):
    k = 5
    key1, key2 = jax.random.split(rng, 2)
    geom, _, _ = make_blobs(n_samples=150, centers=k, random_state=0)

    res = k_means.k_means(geom, k, n_init=n_init, key=key1)
    res_larger_n_init = k_means.k_means(geom, k, n_init=2 * n_init, key=key1)

    assert res_larger_n_init.error < res.error

  def test_store_inner_errors(self):
    pass

  def test_weights(self):
    pass

  def test_cosine_cost_fn(self):
    pass

  def test_jitting(self):
    pass

  def test_differentiability(self):
    pass

  def test_matches_sklearn(self):
    pass
