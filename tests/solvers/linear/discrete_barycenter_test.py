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
from typing import Union

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

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
    err = errors[errors != -1][-1]
    assert err < threshold

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

  def test_debiased_lse_matches_kernel(self):
    x = jnp.linspace(0.0, 1.0, 20)[:, None]
    a = jnp.stack([jnp.linspace(1.0, 2.0, 20), jnp.linspace(2.0, 1.0, 20)])
    geom = pointcloud.PointCloud(x, epsilon=5e-2)
    prob = bp.FixedBarycenterProblem(geom, a=a / a.sum(1, keepdims=True))

    lse, kernel = (
        db.FixedBarycenter(
            lse_mode=lse_mode,
            debiased=True,
            min_iterations=500,
            max_iterations=500
        )(prob).histogram for lse_mode in (True, False)
    )

    np.testing.assert_allclose(lse, kernel, rtol=1e-3, atol=1e-5)

  @pytest.mark.parametrize(("min_iterations", "max_iterations", "last"), [
      (10, 100, 0),
      (20, 20, 1),
  ])
  def test_min_iterations(
      self, min_iterations: int, max_iterations: int, last: int
  ):
    x = jnp.linspace(0.0, 1.0, 10)[:, None]
    a = jnp.stack([jnp.linspace(1.0, 2.0, 10), jnp.linspace(2.0, 1.0, 10)])
    geom = pointcloud.PointCloud(x, epsilon=1e-1)
    prob = bp.FixedBarycenterProblem(geom, a=a / a.sum(1, keepdims=True))
    solver = db.FixedBarycenter(
        threshold=1e3,
        min_iterations=min_iterations,
        max_iterations=max_iterations
    )

    errors = solver(prob).errors[:, 0]

    np.testing.assert_array_equal(errors[:last], jnp.inf)
    assert jnp.isfinite(errors[last])
    np.testing.assert_array_equal(errors[last + 1:], -1.0)

  @pytest.mark.usefixtures("enable_x64")
  @pytest.mark.parametrize("lse_mode", [True, False])
  def test_grad(self, rng: jax.Array, lse_mode: bool):
    rng_w, rng_v = jr.split(rng, 2)
    x = jnp.linspace(0.0, 1.0, 10)[:, None]
    a = jnp.stack([jnp.linspace(1.0, 2.0, 10), jnp.linspace(2.0, 1.0, 10)])
    a = a / a.sum(1, keepdims=True)
    w = jr.normal(rng_w, (10,))
    v = jr.normal(rng_v, x.shape)
    v = 1e-4 * v / jnp.linalg.norm(v)
    solver = db.FixedBarycenter(
        lse_mode=lse_mode, min_iterations=100, max_iterations=100
    )

    def loss(x: jnp.ndarray) -> float:
      geom = pointcloud.PointCloud(x, epsilon=1e-1)
      prob = bp.FixedBarycenterProblem(geom, a=a)
      return jnp.dot(w, solver(prob).histogram)

    grad = jax.grad(loss)(x)

    expected = loss(x + v) - loss(x - v)
    actual = 2.0 * jnp.vdot(v, grad)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-8)

  @pytest.mark.parametrize("lse_mode", [True, False])
  def test_epsilon_schedule(self, lse_mode: bool):
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # identical marginals, so that the initial duals are 0 for any epsilon
    a = jnp.ones((2, x.shape[0])) / x.shape[0]
    weights = jnp.array([0.4, 0.6])

    def first_step(
        epsilon: Union[float, epsilon_scheduler.Epsilon]
    ) -> jnp.ndarray:
      geom = pointcloud.PointCloud(x, epsilon=epsilon)
      prob = bp.FixedBarycenterProblem(geom, a=a, weights=weights)
      solver = db.FixedBarycenter(
          max_iterations=1, inner_iterations=1, lse_mode=lse_mode
      )
      return solver(prob).f

    eps = epsilon_scheduler.Epsilon(target=0.2, init=8.0, decay=0.5)
    np.testing.assert_allclose(
        first_step(eps), first_step(eps(0)), rtol=1e-5, atol=1e-5
    )
