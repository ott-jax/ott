import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import Sinkhorn, linear_problems
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from ott.tools.gaussian_mixture import gaussian


class TestEntropicPotentials:

  @pytest.mark.fast.with_args(eps=[5e-2, 1e-1], only_fast=0)
  def test_entropic_potentials_dist(self, rng: jnp.ndarray, eps: float):
    n1, n2, d = 64, 96, 2
    key1, key2, key3, key4 = jax.random.split(rng, 4)

    mean1, mean2 = jnp.zeros(d), jnp.ones(d) * 2
    cov1, cov2 = jnp.eye(d), jnp.array([[2, 0], [0, 0.5]])
    g1 = gaussian.Gaussian.from_mean_and_cov(mean1, cov1)
    g2 = gaussian.Gaussian.from_mean_and_cov(mean2, cov2)
    x = g1.sample(key1, n1)
    y = g2.sample(key2, n2)

    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problems.LinearProblem(geom)
    out = Sinkhorn()(prob)
    assert out.converged
    potentials = out.to_dual_potentials()

    expected_dist = jnp.sum(out.matrix * geom.cost_matrix)
    actual_dist = potentials.distance(x, y)
    rel_error = jnp.abs(expected_dist - actual_dist) / expected_dist
    assert rel_error < 2 * eps

  @pytest.mark.fast.with_args(forward=[False, True], only_fast=0)
  def test_entropic_potentials_displacement(
      self, rng: jnp.ndarray, forward: bool
  ):
    n1, n2, d = 96, 128, 2
    eps = 1e-2
    key1, key2, key3, key4 = jax.random.split(rng, 4)

    mean1, mean2 = jnp.zeros(d), jnp.ones(d) * 2
    cov1, cov2 = jnp.eye(d), jnp.array([[1.5, 0], [0, 0.8]])
    g1 = gaussian.Gaussian.from_mean_and_cov(mean1, cov1)
    g2 = gaussian.Gaussian.from_mean_and_cov(mean2, cov2)
    x = g1.sample(key1, n1)
    y = g2.sample(key2, n2)

    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problems.LinearProblem(geom)
    out = Sinkhorn()(prob)
    assert out.converged
    potentials = out.to_dual_potentials()

    x_test = g1.sample(key3, n1 + 1)
    y_test = g2.sample(key4, n2 + 2)
    if forward:
      expected_points = g1.transport(g2, x_test)
      actual_points = potentials.transport(x_test, forward=forward)
    else:
      expected_points = g2.transport(g1, y_test)
      actual_points = potentials.transport(y_test, forward=forward)

    # TODO(michalk8): better error measure
    error = jnp.mean(jnp.sum((expected_points - actual_points) ** 2, axis=-1))
    assert error <= 0.3

  @pytest.mark.parametrize("jit", [False, True])
  def test_distance_differentiability(self, rng: jnp.ndarray, jit: bool):
    key1, key2, key3 = jax.random.split(rng, 3)
    n, m, d = 18, 36, 5

    x = jax.random.normal(key1, (n, d))
    y = jax.random.normal(key2, (m, d))
    prob = linear_problems.LinearProblem(pointcloud.PointCloud(x, y))
    v_x = jax.random.normal(key3, shape=x.shape)
    v_x = (v_x / jnp.linalg.norm(v_x, axis=-1, keepdims=True)) * 1e-3

    pots = Sinkhorn()(prob).to_dual_potentials()

    grad_dist = jax.grad(pots.distance)
    if jit:
      grad_dist = jax.jit(grad_dist)
    dx = grad_dist(x, y)

    expected = pots.distance(x + v_x, y) - pots.distance(x - v_x, y)
    actual = 2. * jnp.vdot(v_x, dx)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("static_b", [False, True])
  def test_potentials_sinkhorn_divergence(
      self, rng: jnp.ndarray, static_b: bool
  ):
    key1, key2, key3 = jax.random.split(rng, 3)
    n, m, d = 32, 36, 4
    eps, fwd = 1., True
    mu0, mu1 = -5., 5.

    x = jax.random.normal(key1, (n, d)) + mu0
    y = jax.random.normal(key2, (m, d)) + mu1
    x_test = jax.random.normal(key3, (n, d)) + mu0
    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problems.LinearProblem(geom)

    sink_pots = Sinkhorn()(prob).to_dual_potentials()
    div_pots = sinkhorn_divergence.sinkhorn_divergence(
        type(geom), x, y, epsilon=eps
    ).to_dual_potentials()

    sink_dist = sink_pots.distance(x, y)
    div_dist = div_pots.distance(x, y)
    assert div_dist < sink_dist

    sink_points = sink_pots.transport(x_test, forward=fwd)
    div_points = div_pots.transport(x_test, forward=fwd)

    with pytest.raises(AssertionError):
      np.testing.assert_allclose(sink_points, div_points)
    np.testing.assert_allclose(sink_points, div_points, rtol=0.08, atol=0.31)
