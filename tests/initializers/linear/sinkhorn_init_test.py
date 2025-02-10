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

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import geometry, pointcloud
from ott.initializers.linear import initializers as linear_init
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def create_sorting_problem(
    rng: jax.Array,
    n: int,
    epsilon: float = 1e-2,
    batch_size: Optional[int] = None
) -> linear_problem.LinearProblem:
  # define ot problem
  x_init = jnp.array([-1.0, 0.0, 0.22])
  y_init = jnp.array([0.0, 0.0, 1.1])
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
    rng: jax.Array,
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


@pytest.mark.fast()
class TestSinkhornInitializers:

  @pytest.mark.parametrize(("vector_min", "lse_mode"), [(True, True),
                                                        (True, False),
                                                        (False, True)])
  def test_sorting_init(self, vector_min: bool, lse_mode: bool, rng: jax.Array):
    """Tests sorting dual initializer."""
    n, epsilon = 50, 1e-2

    ot_problem = create_sorting_problem(rng=rng, n=n, epsilon=epsilon)

    solver = sinkhorn.Sinkhorn(lse_mode=lse_mode)
    sink_out_base = jax.jit(solver)(ot_problem)

    solver = sinkhorn.Sinkhorn(
        lse_mode=lse_mode,
        initializer=linear_init.SortingInitializer(
            vectorized_update=vector_min
        )
    )
    sink_out_init = jax.jit(solver)(ot_problem)

    # check initializer is better or equal
    if lse_mode:
      assert sink_out_base.converged
      assert sink_out_init.converged
      assert sink_out_base.n_iters > sink_out_init.n_iters

  def test_sorting_init_online(self, rng: jax.Array):
    n = 10
    epsilon = 1e-2

    ot_problem = create_sorting_problem(
        rng=rng, n=n, epsilon=epsilon, batch_size=5
    )
    sort_init = linear_init.SortingInitializer(vectorized_update=True)
    with pytest.raises(AssertionError, match=r"online"):
      sort_init.init_fu(ot_problem, lse_mode=True)

  def test_sorting_init_square_cost(self, rng: jax.Array):
    n, m, d = 10, 15, 1
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon)
    sort_init = linear_init.SortingInitializer(vectorized_update=True)
    with pytest.raises(AssertionError, match=r"square"):
      sort_init.init_fu(ot_problem, lse_mode=True)

  def test_default_initializer(self, rng: jax.Array):
    """Tests default initializer"""
    n, m, d = 20, 20, 2
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, batch_size=3)

    f = linear_init.DefaultInitializer().init_fu(ot_problem, lse_mode=True)
    g = linear_init.DefaultInitializer().init_gv(ot_problem, lse_mode=True)

    # check default is 0
    np.testing.assert_array_equal(f, 0.0)
    np.testing.assert_array_equal(g, 0.0)

  def test_gauss_pointcloud_geom(self, rng: jax.Array):
    n, m, d = 20, 20, 2
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, batch_size=3)

    gaus_init = linear_init.GaussianInitializer()
    new_geom = geometry.Geometry(
        cost_matrix=ot_problem.geom.cost_matrix, epsilon=epsilon
    )
    ot_problem = linear_problem.LinearProblem(
        geom=new_geom, a=ot_problem.a, b=ot_problem.b
    )

    with pytest.raises(AssertionError, match=r"pointcloud"):
      gaus_init.init_fu(ot_problem, lse_mode=True)

  @pytest.mark.parametrize("lse_mode", [True, False])
  @pytest.mark.parametrize(
      "initializer", [
          linear_init.SortingInitializer(vectorized_update=True),
          linear_init.GaussianInitializer(),
          linear_init.SubsampleInitializer(10)
      ]
  )
  def test_initializer_n_iter(
      self,
      rng: jax.Array,
      lse_mode: bool,
      initializer: linear_init.SinkhornInitializer,
  ):
    """Tests Gaussian initializer"""
    n, m, d = 40, 40, 2
    epsilon = 5e-2

    # ot problem
    if isinstance(initializer, linear_init.SortingInitializer):
      ot_problem = create_sorting_problem(rng, n=n, epsilon=epsilon)
    else:
      ot_problem = create_ot_problem(
          rng, n, m, d, epsilon=epsilon, batch_size=3
      )

    solver = sinkhorn.Sinkhorn(lse_mode=lse_mode)
    default_out = jax.jit(solver)(ot_problem)

    solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, initializer=initializer)
    init_out = solver(ot_problem)

    if lse_mode:
      assert default_out.converged
      assert init_out.converged
      assert default_out.n_iters > init_out.n_iters
    else:
      assert default_out.n_iters >= init_out.n_iters
