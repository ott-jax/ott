import jax
import jax.numpy as jnp
import pytest

from ott.core import Sinkhorn, linear_problems
from ott.geometry import pointcloud
from ott.tools.gaussian_mixture import gaussian


class TestEntropicMap:

  @pytest.mark.parametrize("eps", [1e-2, 1e-1])
  @pytest.mark.parametrize("forward", [False, True])
  def test_entropic_map(self, rng: jnp.ndarray, forward: bool, eps: float):
    n1, n2, d = 64, 96, 2
    key1, key2, key3, key4 = jax.random.split(rng, 4)

    mean1, mean2 = jnp.zeros(d), jnp.ones(d) * 4
    cov1, cov2 = jnp.eye(d), jnp.array([[2, 0], [0, 0.5]])
    g1 = gaussian.Gaussian.from_mean_and_cov(mean1, cov1)
    g2 = gaussian.Gaussian.from_mean_and_cov(mean2, cov2)
    x = g1.sample(key1, n1)
    y = g2.sample(key1, n2)

    geom = pointcloud.PointCloud(x, y, epsilon=eps)
    prob = linear_problems.LinearProblem(geom)
    out = Sinkhorn()(prob)

    potentials = out.to_dual_potentials()

    x_test = g1.sample(key3, n1 + 1)
    y_test = g2.sample(key4, n1 + 1)

    expected_dist = g1.w2_dist(g2)
    actual_dist = potentials.distance(x_test, y_test)
    assert jnp.abs(expected_dist - actual_dist) < 2.5

    if forward:
      expected_points = g1.transport(g2, x_test)
      actual_points = potentials.transport(x_test, forward=forward)
    else:
      expected_points = g2.transport(g1, y_test)
      actual_points = potentials.transport(y_test, forward=forward)

    error = jnp.mean(
        jnp.linalg.norm(expected_points - actual_points, axis=1) ** 2
    )
    assert error <= 0.6

  def test_sinkhorn_divergence(self):
    pass
