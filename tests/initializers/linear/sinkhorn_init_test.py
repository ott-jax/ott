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
"""Tests for Sinkhorn initializers."""
from typing import Any, Literal, Optional

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import geometry, pointcloud
from ott.initializers.linear import initializers as linear_init
from ott.initializers.nn import initializers as nn_init
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def create_sorting_problem(
    rng: jnp.ndarray,
    n: int,
    epsilon: float = 1e-2,
    batch_size: Optional[int] = None
) -> linear_problem.LinearProblem:
  # define ot problem
  x_init = jnp.array([-1., 0, .22])
  y_init = jnp.array([0., 0, 1.1])
  x_rng, y_rng = jax.random.split(rng)

  x = jnp.concatenate([x_init, 10 + jnp.abs(jax.random.normal(x_rng, (n,)))])
  y = jnp.concatenate([y_init, 10 + jnp.abs(jax.random.normal(y_rng, (n,)))])

  x = jnp.sort(x)
  y = jnp.sort(y)

  n, m = len(x), len(y)
  a = jnp.ones(n) / n
  b = jnp.ones(m) / m

  geom = pointcloud.PointCloud(
      x.reshape(-1, 1),
      y.reshape(-1, 1),
      epsilon=epsilon,
      batch_size=batch_size
  )
  return linear_problem.LinearProblem(geom=geom, a=a, b=b)


def create_ot_problem(
    rng: jnp.ndarray,
    n: int,
    m: int,
    d: int,
    epsilon: float = 1e-2,
    batch_size: Optional[int] = None
) -> linear_problem.LinearProblem:
  # define ot problem
  x_rng, y_rng = jax.random.split(rng)

  mu_a = jnp.array([-1, 1]) * 5
  mu_b = jnp.array([0, 0])

  x = jax.random.normal(x_rng, (n, d)) + mu_a
  y = jax.random.normal(y_rng, (m, d)) + mu_b

  a = jnp.ones(n) / n
  b = jnp.ones(m) / m

  geom = pointcloud.PointCloud(x, y, epsilon=epsilon, batch_size=batch_size)

  return linear_problem.LinearProblem(geom=geom, a=a, b=b)


def run_sinkhorn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    initializer: Literal["default", "sorting", "gaussian"],
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    epsilon: float = 1e-2,
    lse_mode: bool = True,
    **kwargs: Any
) -> sinkhorn.SinkhornOutput:
  if initializer == "default":
    init = linear_init.DefaultInitializer()
  elif initializer == "sorting":
    init = linear_init.SortingInitializer(**kwargs)
  elif initializer == "gaussian":
    init = linear_init.GaussianInitializer(**kwargs)
  else:
    raise NotImplementedError(initializer)

  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  prob = linear_problem.LinearProblem(geom, a, b)
  solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, initializer=init)
  return solver(prob)


@pytest.mark.fast
class TestSinkhornInitializers:

  @pytest.mark.parametrize(
      "init", [
          "default", "gaussian", "sorting",
          linear_init.DefaultInitializer(), "non-existent"
      ]
  )
  def test_create_initializer(self, init: str):
    solver = sinkhorn.Sinkhorn(initializer=init)
    expected_types = {
        "default": linear_init.DefaultInitializer,
        "gaussian": linear_init.GaussianInitializer,
        "sorting": linear_init.SortingInitializer,
    }

    if isinstance(init, linear_init.SinkhornInitializer):
      assert solver.create_initializer() is init
    elif init == "non-existent":
      with pytest.raises(NotImplementedError, match=r""):
        _ = solver.create_initializer()
    else:
      actual = solver.create_initializer()
      expected_type = expected_types[init]
      assert isinstance(actual, expected_type)

  @pytest.mark.parametrize(
      "vector_min, lse_mode", [(True, True), (True, False), (False, True)]
  )
  def test_sorting_init(self, vector_min: bool, lse_mode: bool):
    """Tests sorting dual initializer."""
    rng = jax.random.PRNGKey(42)
    n = 500
    epsilon = 1e-2

    ot_problem = create_sorting_problem(rng=rng, n=n, epsilon=epsilon)

    sink_out_base = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        initializer="default",
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon
    )

    sink_out_init = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        initializer="sorting",
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        vectorized_update=vector_min,
        lse_mode=lse_mode
    )

    # check initializer is better or equal
    if lse_mode:
      assert sink_out_base.converged
      assert sink_out_init.converged
      assert sink_out_base.n_iters > sink_out_init.n_iters

  def test_sorting_init_online(self, rng: jnp.ndarray):
    n = 100
    epsilon = 1e-2

    ot_problem = create_sorting_problem(
        rng=rng, n=n, epsilon=epsilon, batch_size=5
    )
    sort_init = linear_init.SortingInitializer(vectorized_update=True)
    with pytest.raises(AssertionError, match=r"online"):
      sort_init.init_dual_a(ot_problem, lse_mode=True)

  def test_sorting_init_square_cost(self, rng: jnp.ndarray):
    n, m, d = 100, 150, 1
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon)
    sort_init = linear_init.SortingInitializer(vectorized_update=True)
    with pytest.raises(AssertionError, match=r"square"):
      sort_init.init_dual_a(ot_problem, lse_mode=True)

  def test_default_initializer(self, rng: jnp.ndarray):
    """Tests default initializer"""
    n, m, d = 200, 200, 2
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, batch_size=3)

    default_potential_a = linear_init.DefaultInitializer().init_dual_a(
        ot_problem, lse_mode=True
    )
    default_potential_b = linear_init.DefaultInitializer().init_dual_b(
        ot_problem, lse_mode=True
    )

    # check default is 0
    np.testing.assert_array_equal(0., default_potential_a)
    np.testing.assert_array_equal(0., default_potential_b)

  def test_gauss_pointcloud_geom(self, rng: jnp.ndarray):
    n, m, d = 200, 200, 2
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, batch_size=3)

    gaus_init = linear_init.GaussianInitializer()
    new_geom = geometry.Geometry(
        cost_matrix=ot_problem.geom.cost_matrix, epsilon=epsilon
    )
    ot_problem = linear_problem.LinearProblem(
        geom=new_geom, a=ot_problem.a, b=ot_problem.b
    )

    with pytest.raises(AssertionError, match=r"point cloud"):
      gaus_init.init_dual_a(ot_problem, lse_mode=True)

  @pytest.mark.parametrize('lse_mode', [True, False])
  @pytest.mark.parametrize("jit", [False, True])
  @pytest.mark.parametrize("initializer", ["sorting", "gaussian"])
  def test_initializer_n_iter(
      self, rng: jnp.ndarray, lse_mode: bool, jit: bool,
      initializer: Literal["sorting", "gaussian"]
  ):
    """Tests Gaussian initializer"""
    n, m, d = 200, 200, 2
    epsilon = 1e-2

    if initializer == "gaussian":
      ot_problem = create_ot_problem(
          rng, n, m, d, epsilon=epsilon, batch_size=3
      )
    else:
      ot_problem = create_sorting_problem(rng, n=n, epsilon=epsilon)

    if jit:
      run_fn = jax.jit(
          run_sinkhorn, static_argnames=["initializer", "lse_mode"]
      )
    else:
      run_fn = run_sinkhorn

    # run sinkhorn
    default_out = run_fn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        initializer="default",
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        lse_mode=lse_mode,
    )

    init_out = run_fn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        initializer=initializer,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        lse_mode=lse_mode
    )

    if lse_mode:
      assert default_out.converged
      assert init_out.converged
      assert default_out.n_iters > init_out.n_iters
    else:
      assert default_out.n_iters >= init_out.n_iters

  @pytest.mark.parametrize('lse_mode', [True, False])
  def test_meta_initializer(self, rng: jnp.ndarray, lse_mode: bool):
    """Tests Meta initializer"""
    n, m, d = 200, 200, 2
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, batch_size=3)
    a = ot_problem.a
    b = ot_problem.b
    geom = ot_problem.geom

    # run sinkhorn
    sink_out = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        initializer="default",
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        lse_mode=lse_mode
    )

    # overfit the initializer to the problem.
    meta_initializer = nn_init.MetaInitializer(geom)
    for _ in range(100):
      _, _, meta_initializer.state = meta_initializer.update(
          meta_initializer.state, a=a, b=b
      )

    prob = linear_problem.LinearProblem(geom, a, b)
    solver = sinkhorn.Sinkhorn(initializer=meta_initializer, lse_mode=lse_mode)
    meta_out = solver(prob)

    # check initializer is better
    if lse_mode:
      assert sink_out.converged
      assert meta_out.converged
      assert sink_out.n_iters > meta_out.n_iters
    else:
      assert sink_out.n_iters >= meta_out.n_iters
