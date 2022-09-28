import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import Sinkhorn, linear_problems
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from ott.tools.gaussian_mixture import gaussian


class TestEntropicPotentials:

  @pytest.mark.parametrize("eps", [1e-2, 1e-1])
  @pytest.mark.parametrize("forward", [False, True])
  def test_entropic_potentials(
      self, rng: jnp.ndarray, forward: bool, eps: float
  ):
    n1, n2, d = 64, 96, 2
    key1, key2, key3, key4 = jax.random.split(rng, 4)

    mean1, mean2 = jnp.zeros(d), jnp.ones(d) * 4
    cov1, cov2 = jnp.eye(d), jnp.array([[2, 0], [0, 0.5]])
    g1 = gaussian.Gaussian.from_mean_and_cov(mean1, cov1)
    g2 = gaussian.Gaussian.from_mean_and_cov(mean2, cov2)
    x = g1.sample(key1, n1)
    y = g2.sample(key2, n2)

    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problems.LinearProblem(geom)
    out = Sinkhorn()(prob)

    potentials = out.to_dual_potentials()

    expected_dist = out.reg_ot_cost
    actual_dist = potentials.distance(x, y)
    assert jnp.abs(expected_dist - actual_dist) < 3.5

    x_test = g1.sample(key3, n1 + 1)
    y_test = g2.sample(key4, n2 + 2)
    if forward:
      expected_points = g1.transport(g2, x_test)
      actual_points = potentials.transport(x_test, forward=forward)
    else:
      expected_points = g2.transport(g1, y_test)
      actual_points = potentials.transport(y_test, forward=forward)

    error = jnp.mean(jnp.sum((expected_points - actual_points) ** 2, axis=1))
    assert error <= 0.6

  def test_differentiability(self):
    pass

  @pytest.mark.parametrize("static_b", [False, True])
  def test_potentials_sinkhorn_divergence(
      self, rng: jnp.ndarray, static_b: bool
  ):
    key1, key2, key3 = jax.random.split(rng, 3)
    n, m, d = 32, 36, 2
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
    np.testing.assert_allclose(sink_points, div_points, rtol=0.05, atol=0.25)
