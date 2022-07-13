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
"""Tests for the Gromov Wasserstein."""

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
from absl.testing import absltest, parameterized

from ott.core import initializers as init_lib
from ott.core import linear_problems
from ott.core.sinkhorn import sinkhorn
from ott.geometry import geometry, pointcloud


# define sinkhorn functions
@jax.jit
def run_sinkhorn_sort_init(x, y, a=None, b=None, epsilon=0.01, vector_min=True):
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  sort_init = init_lib.SortingInit(vector_min=vector_min)
  out = sinkhorn(geom, a=a, b=b, jit=True, potential_initializer=sort_init)
  return out


@jax.jit
def run_sinkhorn(x, y, a=None, b=None, epsilon=0.01):
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  out = sinkhorn(geom, a=a, b=b, jit=True)
  return out


@jax.jit
def run_sinkhorn_gaus_init(x, y, a=None, b=None, epsilon=0.01):
  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  out = sinkhorn(
      geom,
      a=a,
      b=b,
      jit=True,
      potential_initializer=init_lib.GaussianInitializer()
  )
  return out


class InitializerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def create_sorting_problem(self, n, epsilon=0.01):
    # definte ot problem
    x_init = jnp.array([-1., 0, .22])
    y_init = jnp.array([0., 0, 1.1])

    x = jnp.concatenate([
        x_init, 10 + jnp.abs(jax.random.normal(self.rng, (n,)))
    ]) * 5
    y = jnp.concatenate([
        y_init, 10 + jnp.abs(jax.random.normal(self.rng, (n,)))
    ]) * 5

    x = np.sort(x)
    y = np.sort(y)

    n = len(x)
    m = len(y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    geom = pointcloud.PointCloud(
        x.reshape(-1, 1), y.reshape(-1, 1), epsilon=epsilon
    )
    ot_problem = linear_problems.LinearProblem(geom=geom, a=a, b=b)

    return ot_problem

  def create_ot_problem(self, n, m, d, epsilon=0.01):
    # definte ot problem
    np.random.seed(0)

    mu_a = np.array([-1, 1]) * 5
    mu_b = np.array([0, 0])

    x = jax.random.normal(self.rng, (n, d)) + mu_a
    y = jax.random.normal(self.rng, (m, d)) + mu_b

    a = np.ones(n) / n
    b = np.ones(m) / m

    x_jnp, y_jnp = jnp.array(x), jnp.array(y)

    geom = pointcloud.PointCloud(x_jnp, y_jnp, epsilon=epsilon)

    ot_problem = linear_problems.LinearProblem(geom=geom, a=a, b=b)
    return ot_problem

  @parameterized.parameters([True], [False])
  def test_sorting_init(self, vector_min):
    """Tests sorting dual initializer."""

    n = 500
    epsilon = 0.01

    ot_problem = self.create_sorting_problem(n=n, epsilon=epsilon)
    # run sinkhorn
    sink_out = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon
    )
    base_num_iter = jnp.sum(sink_out.errors > -1)

    sink_out = run_sinkhorn_sort_init(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        vector_min=vector_min
    )
    sort_num_iter = jnp.sum(sink_out.errors > -1)

    # check initializer is better
    self.assertTrue(base_num_iter >= sort_num_iter)

  def test_default_initializer(self):
    """Tests default initializer"""
    n = 200
    m = 200
    d = 2
    epsilon = 0.01

    ot_problem = self.create_ot_problem(n, m, d)

    default_potential_a = init_lib._default_dual_a(
        ot_problem=ot_problem, lse_mode=True
    )
    default_potential_b = init_lib._default_dual_b(
        ot_problem=ot_problem, lse_mode=True
    )

    # check default is 0
    self.assertTrue((jnp.zeros(n) == default_potential_a).all())
    self.assertTrue((jnp.zeros(m) == default_potential_b).all())

    # check gausian init returns 0 for non point cloud geometry
    # init initializer
    gaus_init = init_lib.GaussianInitializer()
    new_geom = geometry.Geometry(
        cost_matrix=ot_problem.geom.cost_matrix, epsilon=epsilon
    )
    ot_problem = linear_problems.LinearProblem(
        geom=new_geom, a=ot_problem.a, b=ot_problem.b
    )
    init_potential_a = gaus_init.init_dual_a(
        ot_problem=ot_problem, lse_mode=True
    )
    init_potential_b = gaus_init.init_dual_b(
        ot_problem=ot_problem, lse_mode=True
    )

    self.assertTrue((jnp.zeros(n) == init_potential_a).all())
    self.assertTrue((jnp.zeros(m) == init_potential_b).all())

  def test_gaus_initializer(self):
    """Tests Gaussian initializer"""
    # definte ot problem
    n = 200
    m = 200
    d = 2
    epsilon = 0.01

    ot_problem = self.create_ot_problem(n, m, d)

    # run sinkhorn
    sink_out = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon
    )
    base_num_iter = jnp.sum(sink_out.errors > -1)

    sink_out = run_sinkhorn_gaus_init(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon
    )
    gaus_num_iter = jnp.sum(sink_out.errors > -1)

    # check initializer is better
    self.assertTrue(base_num_iter >= gaus_num_iter)


if __name__ == '__main__':
  absltest.main()
