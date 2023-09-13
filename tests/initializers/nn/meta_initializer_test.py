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

import jax
import jax.numpy as jnp
import pytest

_ = pytest.importorskip("flax")

from ott.geometry import pointcloud
from ott.initializers.linear import initializers as linear_init
from ott.initializers.nn import initializers as nn_init
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def create_ot_problem(
    rng: jax.random.PRNGKeyArray,
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
    initializer: linear_init.SinkhornInitializer,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    epsilon: float = 1e-2,
    lse_mode: bool = True,
) -> sinkhorn.SinkhornOutput:
  """Runs Sinkhorn algorithm with given initializer."""

  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  prob = linear_problem.LinearProblem(geom, a, b)
  solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, initializer=initializer)
  return solver(prob)


@pytest.mark.fast()
class TestMetaInitializer:

  @pytest.mark.parametrize("lse_mode", [True, False])
  def test_meta_initializer(self, rng: jax.random.PRNGKeyArray, lse_mode: bool):
    """Tests Meta initializer"""
    n, m, d = 20, 20, 2
    epsilon = 1e-2

    ot_problem = create_ot_problem(rng, n, m, d, epsilon=epsilon, batch_size=3)
    a = ot_problem.a
    b = ot_problem.b
    geom = ot_problem.geom

    # run sinkhorn
    sink_out = run_sinkhorn(
        x=ot_problem.geom.x,
        y=ot_problem.geom.y,
        initializer=linear_init.DefaultInitializer(),
        a=ot_problem.a,
        b=ot_problem.b,
        epsilon=epsilon,
        lse_mode=lse_mode
    )

    # overfit the initializer to the problem.
    meta_initializer = nn_init.MetaInitializer(geom)
    for _ in range(50):
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
