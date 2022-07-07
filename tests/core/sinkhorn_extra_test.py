# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests Anderson acceleration for sinkhorn."""
import functools
from typing import Any, Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import sinkhorn
from ott.geometry import costs, geometry, pointcloud

non_jitted_sinkhorn = functools.partial(sinkhorn.sinkhorn, jit=False)


class TestSinkhornAnderson:
  """Tests for Anderson acceleration."""

  @pytest.mark.fast.with_args(
      lse_mode=[True, False],
      tau_a=[1.0, .98],
      tau_b=[1.0, .985],
      shape=[(237, 153)],
      refresh_anderson_frequency=[1, 3],
      only_fast=0,
  )
  def test_anderson(
      self, rng: jnp.ndarray, lse_mode: float, tau_a: float, tau_b: float,
      shape: Tuple[int, int], refresh_anderson_frequency: int
  ):
    """Test efficiency of Anderson acceleration.

    Args:
      lse_mode: whether to run in lse (True) or kernel (false) mode.
      tau_a: unbalanced parameter w.r.t. 1st marginal
      tau_b: unbalanced parameter w.r.t. 1st marginal
      shape: shape of test problem
      refresh_anderson_frequency: how often to Anderson interpolation should be
        recomputed.
    """
    n, m = shape
    dim = 4
    rngs = jax.random.split(rng, 9)
    x = jax.random.uniform(rngs[0], (n, dim)) / dim
    y = jax.random.uniform(rngs[1], (m, dim)) / dim + .2
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    a = a.at[0].set(0)
    b = b.at[3].set(0)

    # Make weights roughly sum to 1 if unbalanced, normalize else.
    a = a / (0.5 * n) if tau_a < 1.0 else a / jnp.sum(a)
    b = b / (0.5 * m) if tau_b < 1.0 else b / jnp.sum(b)

    # Here epsilon must be small enough to valide gain in performance using
    # Anderson by large enough number of saved iterations,
    # but large enough when lse_mode=False to avoid underflow.
    epsilon = 5e-4 if lse_mode else 5e-3
    threshold = 1e-3
    iterations_anderson = []

    anderson_memory = [0, 5]
    for anderson_acceleration in anderson_memory:
      out = sinkhorn.sinkhorn(
          pointcloud.PointCloud(x, y, epsilon=epsilon),
          a=a,
          b=b,
          tau_a=tau_a,
          tau_b=tau_b,
          lse_mode=lse_mode,
          threshold=threshold,
          anderson_acceleration=anderson_acceleration,
          refresh_anderson_frequency=refresh_anderson_frequency
      )
      errors = out.errors
      clean_errors = errors[errors > -1]
      # Check convergence
      assert threshold > clean_errors[-1]
      # Record number of inner_iterations needed to converge.
      iterations_anderson.append(jnp.size(clean_errors))

    # Check Anderson acceleration speeds up execution when compared to none.
    for i in range(1, len(anderson_memory)):
      assert iterations_anderson[0] > iterations_anderson[i]


@pytest.mark.fast
class TestSinkhornBures:

  @pytest.fixture(autouse=True)
  def initialize(self):
    self.eps = 1.0
    self.n = 11
    self.m = 13
    self.dim = 7
    self.rngs = jax.random.split(jax.random.PRNGKey(0), 6)

    x = jax.random.normal(self.rngs[0], (self.n, self.dim, self.dim))
    y = jax.random.normal(self.rngs[1], (self.m, self.dim, self.dim))

    sig_x = jnp.matmul(x, jnp.transpose(x, (0, 2, 1)))
    sig_y = jnp.matmul(y, jnp.transpose(y, (0, 2, 1)))

    m_x = jax.random.uniform(self.rngs[2], (self.n, self.dim))
    m_y = jax.random.uniform(self.rngs[3], (self.m, self.dim))

    self.x = jnp.concatenate(
        (m_x.reshape((self.n, -1)), sig_x.reshape((self.n, -1))), axis=1
    )
    self.y = jnp.concatenate(
        (m_y.reshape((self.m, -1)), sig_y.reshape((self.m, -1))), axis=1
    )
    a = jax.random.uniform(self.rngs[4], (self.n,)) + .1
    b = jax.random.uniform(self.rngs[5], (self.m,)) + .1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  def test_bures_point_cloud(self):
    """Two point clouds of Gaussians, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(
        self.x,
        self.y,
        cost_fn=costs.Bures(dimension=self.dim, regularization=1e-4),
        epsilon=self.eps
    )
    errors = sinkhorn.sinkhorn(geom, a=self.a, b=self.b, lse_mode=False).errors
    err = errors[errors > -1][-1]
    assert threshold > err

  def test_regularized_unbalanced_bures(self):
    """Tests Regularized Unbalanced Bures."""
    x = jnp.concatenate((jnp.array([0.9]), self.x[0, :]))
    y = jnp.concatenate((jnp.array([1.1]), self.y[0, :]))

    rub = costs.UnbalancedBures(self.dim, 1, 0.8)
    assert not jnp.any(jnp.isnan(rub(x, y)))
    assert not jnp.any(jnp.isnan(rub(y, x)))
    np.testing.assert_allclose(rub(x, y), rub(y, x), rtol=5e-3, atol=5e-3)


class TestSinkhornOnline:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    self.dim = 3
    self.n = 1000
    self.m = 402
    self.rng, *rngs = jax.random.split(rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))
    #  adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @pytest.mark.fast.with_args("batch_size", [1, 13, 402, 1000], only_fast=-1)
  def test_online_matches_offline_size(self, batch_size: int):
    threshold, rtol, atol = 1e-1, 1e-6, 1e-6
    geom_offline = pointcloud.PointCloud(
        self.x, self.y, epsilon=1, batch_size=None
    )
    geom_online = pointcloud.PointCloud(
        self.x, self.y, epsilon=1, batch_size=batch_size
    )

    sol_online = sinkhorn.sinkhorn(
        geom_online,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=True,
        implicit_differentiation=True
    )
    errors_online = sol_online.errors
    err_online = errors_online[errors_online > -1][-1]
    assert threshold > err_online

    sol_offline = sinkhorn.sinkhorn(
        geom_offline,
        a=self.a,
        b=self.b,
        threshold=threshold,
        lse_mode=True,
        implicit_differentiation=True
    )

    np.testing.assert_allclose(
        sol_online.matrix, sol_offline.matrix, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        sol_online.a, sol_offline.a, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        sol_online.b, sol_offline.b, rtol=rtol, atol=atol
    )

  @pytest.mark.parametrize("outer_jit", [False, True])
  def test_online_sinkhorn_jit(self, outer_jit: bool):

    def callback(epsilon: float, batch_size: int) -> sinkhorn.SinkhornOutput:
      geom = pointcloud.PointCloud(
          self.x, self.y, epsilon=epsilon, batch_size=batch_size
      )
      return sinkhorn.sinkhorn(
          geom,
          a=self.a,
          b=self.b,
          threshold=threshold,
          jit=True,
          lse_mode=True,
          implicit_differentiation=True
      )

    threshold = 1e-1
    fun = jax.jit(callback, static_argnums=(1,)) if outer_jit else callback

    errors = fun(epsilon=1.0, batch_size=42).errors
    err = errors[errors > -1][-1]
    assert threshold > err


@pytest.mark.fast
class TestSinkhornUnbalanced:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    self.dim = 4
    self.n = 17
    self.m = 23
    self.rng, *rngs = jax.random.split(rng, 5)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,))
    b = jax.random.uniform(rngs[3], (self.m,))
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @pytest.mark.parametrize("momentum", [1.0, 1.5])
  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_sinkhorn_unbalanced(self, lse_mode: bool, momentum: float):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(self.x, self.y, epsilon=0.1)
    errors = sinkhorn.sinkhorn(
        geom,
        a=self.a,
        b=self.b,
        threshold=threshold,
        momentum=momentum,
        inner_iterations=10,
        norm_error=1,
        lse_mode=lse_mode,
        tau_a=0.8,
        tau_b=0.9
    ).errors
    err = errors[errors > -1][-1]
    assert threshold > err
    assert err > 0


class TestSinkhornJIT:
  """Check jitted and non jit match for Sinkhorn, and that everything jits."""

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    self.dim = 3
    self.n = 10
    self.m = 11
    self.rng, *rngs = jax.random.split(rng, 10)
    self.rngs = rngs
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,)) + .1
    b = jax.random.uniform(rngs[3], (self.m,)) + .1

    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)
    self.epsilon = 0.05
    self.geometry = geometry.Geometry(
        cost_matrix=(
            jnp.sum(self.x ** 2, axis=1)[:, jnp.newaxis] +
            jnp.sum(self.y ** 2, axis=1)[jnp.newaxis, :] -
            2 * jnp.dot(self.x, self.y.T)
        ),
        epsilon=self.epsilon
    )

  @pytest.mark.fast
  def test_jit_vs_non_jit_fwd(self):

    def assert_output_close(x: jnp.ndarray, y: jnp.ndarray) -> None:
      """Asserst SinkhornOutputs are close."""
      x = tuple(a for a in x if (a is not None and isinstance(a, jnp.ndarray)))
      y = tuple(a for a in y if (a is not None and isinstance(a, jnp.ndarray)))
      return chex.assert_tree_all_close(x, y, atol=1e-6, rtol=0)

    def f(
        g: geometry.Geometry, a: jnp.ndarray, b: jnp.ndarray
    ) -> sinkhorn.SinkhornOutput:
      return non_jitted_sinkhorn(g, a, b)

    jitted_result = sinkhorn.sinkhorn(self.geometry, self.a, self.b)
    non_jitted_result = non_jitted_sinkhorn(self.geometry, self.a, self.b)
    user_jitted_result = jax.jit(f)(self.geometry, self.a, self.b)

    assert_output_close(jitted_result, non_jitted_result)
    assert_output_close(jitted_result, user_jitted_result)

  @pytest.mark.parametrize("implicit", [False, True])
  def test_jit_vs_non_jit_bwd(self, implicit: bool):

    def loss(
        a: jnp.ndarray, x: jnp.ndarray, fun: Callable[[Any],
                                                      sinkhorn.SinkhornOutput]
    ):
      out = fun(
          geometry.Geometry(
              cost_matrix=(
                  jnp.sum(x ** 2, axis=1)[:, jnp.newaxis] +
                  jnp.sum(self.y ** 2, axis=1)[jnp.newaxis, :] -
                  2 * jnp.dot(x, self.y.T)
              ),
              epsilon=self.epsilon
          ),
          a=a,
          b=self.b,
          tau_a=0.94,
          tau_b=0.97,
          threshold=1e-4,
          lse_mode=True,
          implicit_differentiation=implicit
      )
      return out.reg_ot_cost

    def value_and_grad(a: jnp.ndarray, x: jnp.ndarray):
      return jax.value_and_grad(loss)(a, x, non_jitted_sinkhorn)

    jitted_loss, jitted_grad = jax.value_and_grad(loss)(
        self.a, self.x, sinkhorn.sinkhorn
    )
    non_jitted_loss, non_jitted_grad = jax.value_and_grad(loss)(
        self.a, self.x, non_jitted_sinkhorn
    )

    user_jitted_loss, user_jitted_grad = jax.jit(value_and_grad)(self.a, self.x)
    chex.assert_tree_all_close(jitted_loss, non_jitted_loss, atol=1e-6, rtol=0)
    chex.assert_tree_all_close(jitted_grad, non_jitted_grad, atol=1e-6, rtol=0)
    chex.assert_tree_all_close(user_jitted_loss, jitted_loss, atol=1e-6, rtol=0)
    chex.assert_tree_all_close(user_jitted_grad, jitted_grad, atol=1e-6, rtol=0)
