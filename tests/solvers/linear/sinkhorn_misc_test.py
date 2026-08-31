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
from typing import Optional

import pytest

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import costs, geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers import linear
from ott.solvers.linear import acceleration
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn
from tests import _utils


class TestSinkhornAnderson:
  """Tests for Anderson acceleration."""

  @pytest.mark.fast.with_args(
      lse_mode=[True, False],
      tau_a=[1.0, 0.98],
      tau_b=[1.0, 0.985],
      only_fast=0,
  )
  def test_anderson(
      self, rng: jax.Array, lse_mode: bool, tau_a: float, tau_b: float
  ):
    """Test efficiency of Anderson acceleration.

    Args:
      lse_mode: whether to run in lse (True) or kernel (false) mode.
      tau_a: unbalanced parameter w.r.t. 1st marginal
      tau_b: unbalanced parameter w.r.t. 1st marginal
    """
    refresh_anderson_frequency = 3
    n, m = (137, 153)
    dim = 4
    rngs = jr.split(rng, 9)
    x = jr.uniform(rngs[0], (n, dim)) / dim
    y = jr.uniform(rngs[1], (m, dim)) / dim + 0.2
    a = jr.uniform(rngs[2], (n,))
    b = jr.uniform(rngs[3], (m,))
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

    anderson_memory = [None, 5]
    for memory in anderson_memory:
      anderson = None if memory is None else acceleration.AndersonAcceleration(
          memory=memory, refresh_every=refresh_anderson_frequency
      )
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
      prob = linear_problem.LinearProblem(geom, a, b, tau_a=tau_a, tau_b=tau_b)
      solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode,
          threshold=threshold,
          anderson=anderson,
      )
      out = solver(prob)
      assert out.converged
      # Record number of inner_iterations needed to converge.
      iterations_anderson.append(out.n_iters)

    # Check Anderson acceleration speeds up execution when compared to none.
    for i in range(1, len(anderson_memory)):
      assert iterations_anderson[i] <= iterations_anderson[0]


#: Dimensionality and regularization of the Bures point clouds.
BURES_DIM = 7
BURES_EPS = 1.0

#: Regularization used by the jit-vs-non-jit comparisons.
JIT_EPSILON = 0.05


@pytest.mark.fast()
class TestSinkhornBures:

  @staticmethod
  @pytest.fixture(scope="class")
  def clouds() -> _utils.PointClouds:
    """Gaussians flattened into (mean, covariance) coordinates."""
    n, m, deg_freedom = 11, 13, BURES_DIM + 4
    rngs = jr.split(jr.key(0), 6)

    x = jr.normal(rngs[0], (n, BURES_DIM, deg_freedom))
    y = jr.normal(rngs[1], (m, BURES_DIM, deg_freedom))
    sig_x = jnp.matmul(x, jnp.transpose(x, (0, 2, 1))) / deg_freedom
    sig_y = jnp.matmul(y, jnp.transpose(y, (0, 2, 1))) / deg_freedom
    m_x = jr.uniform(rngs[2], (n, BURES_DIM))
    m_y = jr.uniform(rngs[3], (m, BURES_DIM))

    return _utils.PointClouds(
        x=jnp.concatenate((m_x.reshape((n, -1)), sig_x.reshape((n, -1))), 1),
        y=jnp.concatenate((m_y.reshape((m, -1)), sig_y.reshape((m, -1))), 1),
        a=_utils.random_probs(rngs[4], n),
        b=_utils.random_probs(rngs[5], m),
    )

  @pytest.mark.parametrize("lse_mode", [False, True])
  @pytest.mark.parametrize(("unbalanced", "thresh"), [(False, 1e-3),
                                                      (True, 1e-4)])
  def test_bures_point_cloud(
      self, clouds: _utils.PointClouds, rng: jax.Array, lse_mode: bool,
      unbalanced: bool, thresh: float
  ):
    """Two point clouds of Gaussians, tested with various parameters."""
    if unbalanced:
      rng1, rng2 = jr.split(rng, 2)
      ws_x = jnp.abs(jr.uniform(rng1, (clouds.x.shape[0], 1))) + 1e-1
      ws_y = jnp.abs(jr.uniform(rng2, (clouds.y.shape[0], 1))) + 1e-1
      ws_x = ws_x.at[0].set(0.0)
      x = jnp.concatenate([ws_x, clouds.x], axis=1)
      y = jnp.concatenate([ws_y, clouds.y], axis=1)
      cost_fn = costs.UnbalancedBures(
          dimension=BURES_DIM, gamma=0.9, sigma=0.98
      )
    else:
      x, y = clouds.x, clouds.y
      cost_fn = costs.Bures(
          dimension=BURES_DIM, sqrtm_kw={"regularization": 1e-4}
      )

    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=BURES_EPS)
    prob = linear_problem.LinearProblem(geom, clouds.a, clouds.b)
    solver = sinkhorn.Sinkhorn(threshold=thresh, lse_mode=lse_mode)
    out = solver(prob)

    assert out.converged, out.errors
    assert thresh > out.errors[out.n_iters - 1]

  def test_regularized_unbalanced_bures_cost(self, clouds: _utils.PointClouds):
    """Tests Regularized Unbalanced Bures."""
    x = jnp.concatenate((jnp.array([0.9]), clouds.x[0, :]))
    y = jnp.concatenate((jnp.array([1.1]), clouds.y[0, :]))

    rub = costs.UnbalancedBures(BURES_DIM, gamma=1.0, sigma=0.8)
    assert not jnp.any(jnp.isnan(rub(x, y)))
    assert not jnp.any(jnp.isnan(rub(y, x)))
    np.testing.assert_allclose(rub(x, y), rub(y, x), rtol=5e-3, atol=5e-3)


class TestSinkhornOnline:

  @staticmethod
  @pytest.fixture(scope="class")
  def clouds(rng: jax.Array) -> _utils.PointClouds:
    """A source cloud large enough to exercise batching, with zero weights."""
    _, *rngs = jr.split(rng, 5)
    return _utils.PointClouds(
        x=jr.uniform(rngs[0], (100, 3)),
        y=jr.uniform(rngs[1], (42, 3)),
        a=_utils.random_probs(rngs[2], 100, offset=0.0, zero_at=(0,)),
        b=_utils.random_probs(rngs[3], 42, offset=0.0, zero_at=(3,)),
    )

  @pytest.mark.fast.with_args("batch_size", [1, 13, 42, 100], only_fast=-1)
  def test_online_matches_offline_size(
      self, clouds: _utils.PointClouds, batch_size: int
  ):
    threshold, rtol, atol = 1e-1, 1e-6, 1e-6
    geom_offline = pointcloud.PointCloud(
        clouds.x, clouds.y, epsilon=1, batch_size=None
    )
    geom_online = pointcloud.PointCloud(
        clouds.x, clouds.y, epsilon=1, batch_size=batch_size
    )

    sol_online = linear.solve(geom_online)
    errors_online = sol_online.errors
    err_online = errors_online[errors_online > -1][-1]
    assert threshold > err_online

    sol_offline = linear.solve(geom_offline)

    np.testing.assert_allclose(
        sol_online.matrix, sol_offline.matrix, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        sol_online.a, sol_offline.a, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        sol_online.b, sol_offline.b, rtol=rtol, atol=atol
    )

  @pytest.mark.parametrize("jit", [False, True])
  def test_online_sinkhorn_jit(self, clouds: _utils.PointClouds, jit: bool):

    def callback(epsilon: float, batch_size: int) -> sinkhorn.SinkhornOutput:
      geom = pointcloud.PointCloud(
          clouds.x, clouds.y, epsilon=epsilon, batch_size=batch_size
      )
      prob = linear_problem.LinearProblem(geom, clouds.a, clouds.b)
      solver = sinkhorn.Sinkhorn(threshold=threshold)
      return solver(prob)

    threshold = 1e-1
    fun = jax.jit(callback, static_argnums=(1,)) if jit else callback

    errors = fun(epsilon=1.0, batch_size=42).errors
    err = errors[errors > -1][-1]
    assert threshold > err


@pytest.mark.fast()
class TestSinkhornUnbalanced:

  @staticmethod
  @pytest.fixture(scope="class")
  def clouds(rng: jax.Array) -> _utils.PointClouds:
    """Point clouds drawn exactly as this class always has."""
    _, *rngs = jr.split(rng, 5)
    return _utils.PointClouds(
        x=jr.uniform(rngs[0], (17, 4)),
        y=jr.uniform(rngs[1], (23, 4)),
        a=_utils.random_probs(rngs[2], 17, offset=0.0),
        b=_utils.random_probs(rngs[3], 23, offset=0.0),
    )

  @pytest.mark.parametrize("momentum", [1.0, 1.5])
  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_sinkhorn_unbalanced(
      self, clouds: _utils.PointClouds, lse_mode: bool, momentum: float
  ):
    """Two point clouds, tested with various parameters."""
    threshold = 1e-3
    geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=0.1)
    prob = linear_problem.LinearProblem(
        geom, clouds.a, clouds.b, tau_a=0.8, tau_b=0.9
    )
    solver = sinkhorn.Sinkhorn(
        threshold=threshold,
        lse_mode=lse_mode,
        norm_error=1,
        momentum=acceleration.Momentum(value=momentum),
        inner_iterations=10
    )

    errors = solver(prob).errors

    err = errors[errors > -1][-1]
    assert threshold > err
    assert err > 0

  @pytest.mark.fast.with_args(
      eps=[1e-1, 1e-2, None],
      tau_a=[0.9, 0.9999],  # works best for high taus
      tau_b=[0.95, 0.997],
      anderson=[
          None,
          acceleration.AndersonAcceleration(memory=5, refresh_every=3)
      ],
      only_fast=[0, -1],
  )
  def test_sinkhorn_unbalanced_recenter_acceleration(
      self,
      clouds: _utils.PointClouds,
      eps: float,
      tau_a: float,
      tau_b: float,
      anderson: Optional[acceleration.AndersonAcceleration],
  ):

    def run_sink(*, recenter: bool) -> sinkhorn.SinkhornOutput:
      geom = pointcloud.PointCloud(clouds.x, clouds.y, epsilon=eps)
      prob = linear_problem.LinearProblem(
          geom, a=clouds.a, b=clouds.b, tau_a=tau_a, tau_b=tau_b
      )
      solver = sinkhorn.Sinkhorn(
          recenter_potentials=recenter,
          anderson=anderson,
          parallel_dual_updates=False,
          lse_mode=True,
          inner_iterations=1,
          max_iterations=2000,
          threshold=1e-3
      )
      return solver(prob)

    out = run_sink(recenter=False)
    out_center = run_sink(recenter=True)

    assert out.converged
    assert out_center.converged
    assert out_center.n_iters <= out.n_iters
    np.testing.assert_allclose(out.reg_ot_cost, out.reg_ot_cost)


class TestSinkhornJIT:
  """Check jitted and non jit match for Sinkhorn, and that everything jits."""

  @staticmethod
  @pytest.fixture(scope="class")
  def clouds(rng: jax.Array) -> _utils.PointClouds:
    """Point clouds drawn exactly as this class always has."""
    _, *rngs = jr.split(rng, 10)
    return _utils.PointClouds(
        x=jr.uniform(rngs[0], (10, 3)),
        y=jr.uniform(rngs[1], (11, 3)),
        a=_utils.random_probs(rngs[2], 10),
        b=_utils.random_probs(rngs[3], 11),
    )

  @staticmethod
  @pytest.fixture(scope="class")
  def geom(clouds: _utils.PointClouds) -> geometry.Geometry:
    """Squared Euclidean geometry between :func:`clouds`' points."""
    x, y = clouds.x, clouds.y
    return geometry.Geometry(
        cost_matrix=(
            jnp.sum(x ** 2, axis=1)[:, jnp.newaxis] +
            jnp.sum(y ** 2, axis=1)[jnp.newaxis, :] - 2 * jnp.dot(x, y.T)
        ),
        epsilon=JIT_EPSILON
    )

  @pytest.mark.fast()
  def test_jit_vs_non_jit_fwd(
      self, clouds: _utils.PointClouds, geom: geometry.Geometry
  ):

    def assert_output_close(
        x: sinkhorn.SinkhornOutput, y: sinkhorn.SinkhornOutput
    ) -> None:
      """Assert SinkhornOutputs are close."""
      x = tuple(
          a for a in x
          if (a is not None and (isinstance(a, (jnp.ndarray, int))))
      )
      y = tuple(
          a for a in y
          if (a is not None and (isinstance(a, (jnp.ndarray, int))))
      )
      return chex.assert_trees_all_close(x, y, atol=1e-6, rtol=0)

    geom = geom
    jitted_result = jax.jit(linear.solve)(geom, a=clouds.a, b=clouds.b)
    non_jitted_result = linear.solve(geom, a=clouds.a, b=clouds.b)
    assert_output_close(non_jitted_result, jitted_result)

  @pytest.mark.parametrize("implicit", [False, True])
  def test_jit_vs_non_jit_bwd(
      self, clouds: _utils.PointClouds, geom: geometry.Geometry, implicit: bool
  ):

    @jax.value_and_grad
    def val_grad(a: jnp.ndarray, x: jnp.ndarray) -> float:
      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      geom = geometry.Geometry(
          cost_matrix=(
              jnp.sum(x ** 2, axis=1)[:, jnp.newaxis] +
              jnp.sum(clouds.y ** 2, axis=1)[jnp.newaxis, :] -
              2 * jnp.dot(x, clouds.y.T)
          ),
          epsilon=JIT_EPSILON
      )
      prob = linear_problem.LinearProblem(
          geom, a=a, b=clouds.b, tau_a=0.94, tau_b=0.97
      )
      solver = sinkhorn.Sinkhorn(threshold=1e-4, implicit_diff=implicit_diff)
      return solver(prob).reg_ot_cost

    jitted_loss, jitted_grad = jax.jit(val_grad)(clouds.a, clouds.x)
    non_jitted_loss, non_jitted_grad = val_grad(clouds.a, clouds.x)

    chex.assert_trees_all_close(
        jitted_loss, non_jitted_loss, atol=1e-6, rtol=0.0
    )
    chex.assert_trees_all_close(
        jitted_grad, non_jitted_grad, atol=1e-6, rtol=0.0
    )
