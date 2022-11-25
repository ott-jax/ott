import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem, potentials
from ott.solvers.linear import sinkhorn
from ott.tools import sinkhorn_divergence
from ott.tools.gaussian_mixture import gaussian


class TestDualPotentials:

  def test_device_put(self):
    pot = potentials.DualPotentials(
        lambda x: x, lambda x: x, cost_fn=costs.SqEuclidean(), corr=True
    )
    _ = jax.device_put(pot, "cpu")


class TestEntropicPotentials:

  def test_device_put(self, rng: jax.random.PRNGKeyArray):
    n = 10
    device = jax.devices()[0]
    keys = jax.random.split(rng, 5)
    f = jax.random.normal(keys[0], (n,))
    g = jax.random.normal(keys[1], (n,))

    geom = pointcloud.PointCloud(jax.random.normal(keys[2], (n, 3)))
    a = jax.random.normal(keys[4], (n, 3))
    b = jax.random.normal(keys[5], (n, 3))
    prob = linear_problem.LinearProblem(geom, a, b)

    pot = potentials.EntropicPotentials(f, g, prob)

    _ = jax.device_put(pot, device)

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
    prob = linear_problem.LinearProblem(geom)
    out = sinkhorn.Sinkhorn()(prob)
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
    prob = linear_problem.LinearProblem(geom)
    out = sinkhorn.Sinkhorn()(prob)
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

  @pytest.mark.fast.with_args(p=[1.3, 2.2], forward=[False, True], only_fast=0)
  def test_entropic_potentials_sqpnorm(
      self, rng: jnp.ndarray, p: float, forward: bool
  ):
    epsilon = None
    cost_fn = costs.SqPNorm(p=p)
    n1, n2, d = 93, 127, 2
    eps = 1e-2
    keys = jax.random.split(rng, 4)

    x = jax.random.uniform(keys[0], (n1, d))
    y = jax.random.normal(keys[1], (n2, d)) + 2

    geom = pointcloud.PointCloud(x, y, epsilon=eps, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom)
    out = sinkhorn.Sinkhorn()(prob)
    assert out.converged
    potentials = out.to_dual_potentials()

    x_test = jax.random.uniform(keys[2], (n1 + 3, d))
    y_test = jax.random.normal(keys[3], (n2 + 5, d)) + 2

    sdiv = lambda x, y: sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x, y, cost_fn=cost_fn, epsilon=epsilon
    )

    if forward:
      z = potentials.transport(x_test, forward=forward)
      div = sdiv(z, y).divergence
    else:
      z = potentials.transport(y_test, forward=forward)
      div = sdiv(x, z).divergence

    div_0 = sdiv(x, y).divergence
    assert div < .1 * div_0  # check we have moved points much closer to target.

  @pytest.mark.fast.with_args(p=[1.45, 2.2], forward=[False, True], only_fast=0)
  def test_entropic_potentials_pnorm(
      self, rng: jnp.ndarray, p: float, forward: bool
  ):
    epsilon = None
    cost_fn = costs.PNorm(p=p)
    n1, n2, d = 43, 77, 2
    eps = 1e-2
    keys = jax.random.split(rng, 4)

    x = jax.random.uniform(keys[0], (n1, d))
    y = jax.random.normal(keys[1], (n2, d)) + 2

    geom = pointcloud.PointCloud(x, y, epsilon=eps, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom)
    out = sinkhorn.Sinkhorn()(prob)
    assert out.converged
    potentials = out.to_dual_potentials()

    x_test = jax.random.uniform(keys[2], (n1 + 3, d))
    y_test = jax.random.normal(keys[3], (n2 + 5, d)) + 2

    sdiv = lambda x, y: sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x, y, cost_fn=cost_fn, epsilon=epsilon
    )

    if forward:
      z = potentials.transport(x_test, forward=forward)
      div = sdiv(z, y).divergence
    else:
      z = potentials.transport(y_test, forward=forward)
      div = sdiv(x, z).divergence

    div_0 = sdiv(x, y).divergence
    assert div < .1 * div_0  # check we have moved points much closer to target.

  @pytest.mark.parametrize("jit", [False, True])
  def test_distance_differentiability(self, rng: jnp.ndarray, jit: bool):
    key1, key2, key3 = jax.random.split(rng, 3)
    n, m, d = 18, 36, 5

    x = jax.random.normal(key1, (n, d))
    y = jax.random.normal(key2, (m, d))
    prob = linear_problem.LinearProblem(pointcloud.PointCloud(x, y))
    v_x = jax.random.normal(key3, shape=x.shape)
    v_x = (v_x / jnp.linalg.norm(v_x, axis=-1, keepdims=True)) * 1e-3

    pots = sinkhorn.Sinkhorn()(prob).to_dual_potentials()

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
    prob = linear_problem.LinearProblem(geom)

    sink_pots = sinkhorn.Sinkhorn()(prob).to_dual_potentials()
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
