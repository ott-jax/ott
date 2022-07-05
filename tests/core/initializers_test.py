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
from ott.core.linear_problems import LinearProblem
from ott.core.sinkhorn import sinkhorn
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud


class InitializerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def test_sorting_init(self):
    """Tests sorting dual initializer."""

    # init initializer
    sort_init = init_lib.SortingInit(vector_min=True)

    # define sinkhorn functions
    @jax.jit
    def run_sinkhorn_sort_init(x, y, a=None, b=None, init_dual_a=None):
      sink_kwargs = {
          'jit': True,
          'threshold': 0.001,
          'max_iterations': 10 ** 5,
          'potential_initializer': sort_init
      }
      geom_kwargs = {'epsilon': 0.01}
      geom = PointCloud(x, y, **geom_kwargs)
      out = sinkhorn(geom, a=a, b=b, init_dual_a=init_dual_a, **sink_kwargs)
      return out

    @jax.jit
    def run_sinkhorn(x, y, a=None, b=None, init_dual_a=None):
      sink_kwargs = {'jit': True, 'threshold': 0.001, 'max_iterations': 10 ** 5}
      geom_kwargs = {'epsilon': 0.01}
      geom = PointCloud(x, y, **geom_kwargs)
      out = sinkhorn(geom, a=a, b=b, init_dual_a=init_dual_a, **sink_kwargs)
      return out

    # definte ot problem
    x_init = np.array([-1., 0, .22])
    y_init = np.array([0., 0, 1.1])

    buf = 500
    np.random.seed(0)
    x = np.concatenate([x_init, 10 + np.abs(np.random.normal(size=buf))]) * 5
    y = np.concatenate([y_init, 10 + np.abs(np.random.normal(size=buf))]) * 5

    x = np.sort(x)
    y = np.sort(y)

    n = len(x)
    m = len(y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    x_jnp, y_jnp = jnp.array(x.reshape(-1, 1)), jnp.array(y.reshape(-1, 1))

    # run sinkhorn
    sink_out = run_sinkhorn(x=x_jnp, y=y_jnp, a=a, b=b)
    base_num_iter = jnp.sum(sink_out.errors > -1)

    sink_out = run_sinkhorn_sort_init(x=x_jnp, y=y_jnp, a=a, b=b)
    sort_num_iter = jnp.sum(sink_out.errors > -1)

    # check initializer is better
    self.assertTrue(base_num_iter >= sort_num_iter)

  def test_default_initializer(self):
    """Tests default initializer"""

    # definte ot problem
    np.random.seed(0)
    n, d = 1000, 2
    mu_a = np.array([-1, 1]) * 5
    mu_b = np.array([0, 0])

    x = np.random.normal(size=n * d).reshape(n, d) + mu_a
    y = np.random.normal(size=n * d).reshape(n, d) + mu_b

    n = len(x)
    m = len(y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    x_jnp, y_jnp = jnp.array(x), jnp.array(y)

    gaus_init = init_lib.GaussianInitializer()

    geom_kwargs = {'epsilon': 0.01}
    geom = PointCloud(x_jnp, y_jnp, **geom_kwargs)

    ot_problem = LinearProblem(geom=geom, a=a, b=b)
    default_potential_a = init_lib.default_dual_a(
        ot_problem=ot_problem, lse_mode=True
    )
    default_potential_b = init_lib.default_dual_b(
        ot_problem=ot_problem, lse_mode=True
    )

    # check default is 0
    self.assertTrue((jnp.zeros(n) == default_potential_a).all())
    self.assertTrue((jnp.zeros(m) == default_potential_b).all())

    # check gausian init returns 0 for non point cloud geometry
    new_geom = Geometry(cost_matrix=geom.cost_matrix, **geom_kwargs)
    ot_problem = LinearProblem(geom=new_geom, a=a, b=b)
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

    # init initializer
    gaus_init = init_lib.GaussianInitializer()

    @jax.jit
    def run_sinkhorn(x, y, a=None, b=None, init_dual_a=None):
      sink_kwargs = {'jit': True, 'threshold': 0.001, 'max_iterations': 10 ** 5}
      geom_kwargs = {'epsilon': 0.01}
      geom = PointCloud(x, y, **geom_kwargs)
      out = sinkhorn(geom, a=a, b=b, init_dual_a=init_dual_a, **sink_kwargs)
      return out

    @jax.jit
    def run_sinkhorn_gaus_init(x, y, a=None, b=None, init_dual_a=None):
      sink_kwargs = {
          'jit': True,
          'threshold': 0.001,
          'max_iterations': 10 ** 5,
          'potential_initializer': gaus_init
      }

      geom_kwargs = {'epsilon': 0.01}
      geom = PointCloud(x, y, **geom_kwargs)
      out = sinkhorn(geom, a=a, b=b, init_dual_a=init_dual_a, **sink_kwargs)
      return out

    # definte ot problem
    np.random.seed(0)
    n, d = 1000, 2
    mu_a = np.array([-1, 1]) * 5
    mu_b = np.array([0, 0])

    x = np.random.normal(size=n * d).reshape(n, d) + mu_a
    y = np.random.normal(size=n * d).reshape(n, d) + mu_b

    n = len(x)
    m = len(y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    x_jnp, y_jnp = jnp.array(x), jnp.array(y)

    # run sinkhorn
    sink_out = run_sinkhorn(x=x_jnp, y=y_jnp, a=a, b=b)
    base_num_iter = jnp.sum(sink_out.errors > -1)

    sink_out = run_sinkhorn_gaus_init(x=x_jnp, y=y_jnp, a=a, b=b)
    gaus_num_iter = jnp.sum(sink_out.errors > -1)

    # check initializer is better
    self.assertTrue(base_num_iter >= gaus_num_iter)


if __name__ == '__main__':
  absltest.main()
