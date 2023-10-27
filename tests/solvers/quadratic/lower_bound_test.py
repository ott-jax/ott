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

import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, pointcloud
from ott.initializers.linear import initializers
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.quadratic import lower_bound
from ott.tools import soft_sort


class TestLowerBoundSolver:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    d_x = 2
    d_y = 3
    self.n, self.m = 13, 15
    rngs = jax.random.split(rng, 4)
    self.x = jax.random.uniform(rngs[0], (self.n, d_x))
    self.y = jax.random.uniform(rngs[1], (self.m, d_y))
    # Currently Histogram Transport only supports uniform distributions:
    a = jnp.ones(self.n)
    b = jnp.ones(self.m)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.cx = jax.random.uniform(rngs[2], (self.n, self.n))
    self.cy = jax.random.uniform(rngs[3], (self.m, self.m))

  @pytest.mark.fast.with_args(
      "epsilon_sort,method,cost_fn",
      [(0.0, "subsample", costs.SqEuclidean()),
       (1e-1, "quantile", costs.PNormP(1.5)), (1.0, "equal", costs.SqPNorm(1)),
       (None, "subsample", costs.PNormP(3.1))],
      only_fast=0,
  )
  def test_lb_pointcloud(
      self, epsilon_sort: float, method: str, cost_fn: costs.CostFn
  ):
    n_sub = min([self.x.shape[0], self.y.shape[0]])
    x, y = (self.x[:n_sub],
            self.y[:n_sub]) if method == "equal" else (self.x, self.y)

    geom_x = pointcloud.PointCloud(x)
    geom_y = pointcloud.PointCloud(y)
    prob = quadratic_problem.QuadraticProblem(
        geom_x, geom_y, a=self.a, b=self.b
    )

    if epsilon_sort is not None and epsilon_sort <= 0.0:
      sort_fn = None
    else:
      sort_fn = functools.partial(
          soft_sort.sort,
          epsilon=epsilon_sort,
          min_iterations=100,
          max_iterations=100,
      )

    solver = lower_bound.LowerBoundSolver(
        epsilon=1e-1,
        sort_fn=sort_fn,
        cost_fn=cost_fn,
        method=method,
        n_subsamples=4,
    )

    out = jax.jit(solver)(prob)

    np.testing.assert_allclose(
        out.primal_cost, jnp.sum(out.geom.cost_matrix * out.matrix), rtol=1e-3
    )

    assert not jnp.isnan(out.reg_ot_cost)

  @pytest.mark.parametrize("method", ["subsample", "quantile", "equal"])
  @pytest.mark.parametrize(
      "sort_fn",
      [
          None,
          functools.partial(
              soft_sort.sort,
              epsilon=1e-3,
              implicit_diff=False,
              # soft sort uses `sorting` initializer, which uses while loop
              # which is not reverse-mode diff.
              initializer=initializers.DefaultInitializer(),
              min_iterations=10,
              max_iterations=10,
          ),
          functools.partial(
              soft_sort.sort,
              epsilon=1e-1,
              implicit_diff=implicit_lib.ImplicitDiff(),
              initializer=initializers.DefaultInitializer(),
              min_iterations=0,
              max_iterations=100,
          )
      ]
  )
  def test_lb_grad(
      self, rng: jax.random.PRNGKeyArray,
      sort_fn: Callable[[jnp.ndarray], jnp.ndarray], method: str
  ):

    def fn(x: jnp.ndarray, y: jnp.ndarray) -> float:
      geom_x = pointcloud.PointCloud(x)
      geom_y = pointcloud.PointCloud(y)
      prob = quadratic_problem.QuadraticProblem(geom_x, geom_y)

      solver = lower_bound.LowerBoundSolver(
          epsilon=5e-2,
          sort_fn=sort_fn,
          cost_fn=costs.SqEuclidean(),
          method=method,
          n_subsamples=n_sub,
      )
      return solver(prob).reg_ot_cost

    rng1, rng2 = jax.random.split(rng)
    eps, tol = 1e-4, 1e-3

    n_sub = min(self.x.shape[0], self.y.shape[0])
    if method == "equal":
      x, y = self.x[:n_sub], self.y[:n_sub]
    else:
      x, y = self.x, self.y

    grad_x, grad_y = jax.jit(jax.grad(fn, (0, 1)))(x, y)

    v_x = jax.random.normal(rng1, shape=x.shape)
    v_x = (v_x / jnp.linalg.norm(v_x, axis=-1, keepdims=True)) * eps
    expected = fn(x + v_x, y) - fn(x - v_x, y)
    actual = 2.0 * jnp.vdot(v_x, grad_x)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    v_y = jax.random.normal(rng2, shape=y.shape)
    v_y = (v_y / jnp.linalg.norm(v_y, axis=-1, keepdims=True)) * eps
    expected = (fn(x, y + v_y) - fn(x, y - v_y))
    actual = 2.0 * jnp.vdot(v_y, grad_y)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
