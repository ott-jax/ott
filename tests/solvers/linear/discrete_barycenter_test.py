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
import pytest

import jax
import jax.numpy as jnp

from ott.geometry import epsilon_scheduler, grid, pointcloud
from ott.problems.linear import barycenter_problem as bp
from ott.solvers.linear import discrete_barycenter as db


class TestDiscreteBarycenter:

  @pytest.mark.parametrize(("lse_mode", "debiased", "epsilon"),
                           [(True, True, 1e-2), (False, False, 2e-2)],
                           ids=["lse-deb", "scal-no-deb"])
  def test_discrete_barycenter_grid(
      self, lse_mode: bool, debiased: bool, epsilon: float
  ):
    """Tests the discrete barycenters on a 5x5x5 grid.

    Puts two masses on opposing ends of the hypercube with small noise in
    between. Check that their W barycenter sits (mostly) at the middle of the
    hypercube (e.g. index (5x5x5-1)/2)

    Args:
      lse_mode: bool, lse or scaling computations.
      debiased: bool, use (or not) debiasing as proposed in
      https://arxiv.org/abs/2006.02575
      epsilon: float, regularization parameter
    """
    size = jnp.array([5, 5, 5])
    grid_3d = grid.Grid(grid_size=size, epsilon=epsilon)
    a1 = jnp.ones(size).ravel()
    a2 = jnp.ones(size).ravel()
    a1 = a1.at[0].set(10000)
    a2 = a2.at[-1].set(10000)
    a1 /= jnp.sum(a1)
    a2 /= jnp.sum(a2)
    threshold = 1e-2

    bar_prob = bp.FixedBarycenterProblem(
        geom=grid_3d, a=jnp.stack((a1, a2)), weights=jnp.array([0.5, 0.5])
    )
    solver = db.FixedBarycenter(
        threshold=threshold, lse_mode=lse_mode, debiased=debiased
    )
    out = solver(bar_prob)
    bar, errors = out.histogram, out.errors

    assert bar[(jnp.prod(size) - 1) // 2] > 0.7
    assert bar[(jnp.prod(size) - 1) // 2] < 1
    err = errors[jnp.isfinite(errors)][-1]
    assert threshold > err

  @pytest.mark.parametrize(("lse_mode", "epsilon"), [(True, 1e-3),
                                                     (False, 1e-2)],
                           ids=["lse", "scale"])
  def test_discrete_barycenter_pointcloud(self, lse_mode: bool, epsilon: float):
    """Tests the discrete barycenters on pointclouds.

    Two measures supported on the same set of points (a 1D grid), barycenter is
    evaluated on a different set of points (still in 1D).

    Args:
      lse_mode: bool, lse or scaling computations
      epsilon: float
    """
    n = 50
    ma = 0.2
    mb = 0.8
    # define two narrow Gaussian bumps in segment [0,1]
    a = jnp.exp(-(jnp.arange(0, n) / (n - 1) - ma) ** 2 / 0.01) + 1e-10
    b = jnp.exp(-(jnp.arange(0, n) / (n - 1) - mb) ** 2 / 0.01) + 1e-10
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    # positions on the real line where weights are supported.
    x = jnp.atleast_2d(jnp.arange(0, n) / (n - 1)).T

    # choose a different support, half the size, for the barycenter.
    # note this is the reason why we do not use debiasing in this case.
    x_support_bar = jnp.atleast_2d((jnp.arange(0, (n / 2)) /
                                    (n / 2 - 1) - 0.5) * 0.9 + 0.5).T

    geom = pointcloud.PointCloud(x, x_support_bar, epsilon=epsilon)
    bar_prob = bp.FixedBarycenterProblem(geom, a=jnp.stack((a, b)))

    out = db.FixedBarycenter(lse_mode=lse_mode)(bar_prob)
    bar = out.histogram
    # check the barycenter has bump in the middle.
    assert bar[n // 4] > 0.1

  @pytest.mark.parametrize("lse_mode", [True, False], ids=["lse", "scale"])
  def test_epsilon_schedule_reaches_the_sinkhorn_update(self, lse_mode: bool):
    """One barycenter step must use ε(0), not the schedule's target ε.

    The entropic update is ε log a - softmin_ε(cost - f - g). Dropping the
    iteration makes the scheduler return its target, so the first step is
    run at the final regularization instead of the annealed one.
    """
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scheduler = epsilon_scheduler.Epsilon(target=0.2, init=8.0, decay=0.5)
    geom = pointcloud.PointCloud(x, epsilon=scheduler)
    a = jnp.ones((2, x.shape[0])) / x.shape[0]
    weights = jnp.array([0.4, 0.6])
    bar_prob = bp.FixedBarycenterProblem(geom, a=a, weights=weights)
    out = db.FixedBarycenter(
        threshold=1e9,
        min_iterations=0,
        max_iterations=1,
        inner_iterations=1,
        lse_mode=lse_mode,
        debiased=False,
    )(bar_prob)

    dual = geom.apply_cost(a.T, axis=0).T
    dual -= jnp.average(dual, weights=weights, axis=0)[jnp.newaxis, :]
    if lse_mode:
      f0, g0 = jnp.zeros_like(a), dual

      def scheduled(f, g, marginal):
        return geom.update_potential(
            f, g, jnp.log(marginal), iteration=0, axis=1
        )

      def at_target(f, g, marginal):
        return geom.update_potential(f, g, jnp.log(marginal), axis=1)
    else:
      f0 = jnp.ones_like(a)
      g0 = geom.scaling_from_potential(dual)

      def scheduled(f, g, marginal):
        return geom.update_scaling(g, marginal, iteration=0, axis=1)

      def at_target(f, g, marginal):
        return geom.update_scaling(g, marginal, axis=1)

    scheduled = jax.vmap(scheduled)(f0, g0, a)
    at_target = jax.vmap(at_target)(f0, g0, a)
    assert scheduler(0) == pytest.approx(8.0 * 0.2)
    assert scheduler(None) == pytest.approx(0.2)
    assert jnp.allclose(out.f, scheduled, atol=1e-5, rtol=1e-5)
    assert not jnp.allclose(out.f, at_target, atol=1e-3, rtol=1e-3)
