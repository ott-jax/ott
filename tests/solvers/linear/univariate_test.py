import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, univariate


class TestUnivariate:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    self.rng = rng
    self.n = 17
    self.m = 29
    self.rng, *rngs = jax.random.split(self.rng, 5)
    self.x = jax.random.uniform(rngs[0], [self.n])
    self.y = jax.random.uniform(rngs[1], [self.m])
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))

    #  adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @pytest.mark.fast.with_args(p=[1.0, 2.0, 1.7])
  def test_cdf_distance(self, p):
    """The Univariate distance coincides with the  sinkhorn solver"""
    univariate_solver = univariate.UnivariateSolver(method="wasserstein", p=p)
    distance = univariate_solver(self.x, self.y, self.a, self.b)

    geom = pointcloud.PointCloud(
        x=self.x[:, None],
        y=self.y[:, None],
        cost_fn=costs.PNormP(p),
        epsilon=5e-5
    )
    prob = linear_problem.LinearProblem(geom, a=self.a, b=self.b)
    sinkhorn_solver = sinkhorn.Sinkhorn(max_iterations=int(1e6))
    sinkhorn_soln = sinkhorn_solver(prob)

    np.testing.assert_allclose(
        sinkhorn_soln.primal_cost, distance, atol=0, rtol=1e-2
    )
