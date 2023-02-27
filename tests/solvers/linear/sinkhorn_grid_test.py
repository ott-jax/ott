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
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import grid, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


class TestSinkhornGrid:

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_separable_grid(self, rng: jax.random.PRNGKeyArray, lse_mode: bool):
    """Two histograms in a grid of size 5 x 6 x 7  in the hypercube^3."""
    grid_size = (5, 6, 7)
    rngs = jax.random.split(rng, 2)
    a = jax.random.uniform(rngs[0], grid_size)
    b = jax.random.uniform(rngs[1], grid_size)
    #  adding zero weights  to test proper handling, then ravel.
    a = a.at[0].set(0).ravel()
    a = a / jnp.sum(a)
    b = b.at[3].set(0).ravel()
    b = b / jnp.sum(b)

    threshold = 0.01
    geom = grid.Grid(grid_size=grid_size, epsilon=0.1)
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn(threshold=threshold, lse_mode=lse_mode)
    errors = solver(prob).errors
    err = errors[jnp.isfinite(errors)][-1]
    assert threshold > err

  @pytest.mark.fast.with_args("lse_mode", [False, True], only_fast=0)
  def test_grid_vs_euclidean(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool
  ):
    grid_size = (5, 6, 7)
    rngs = jax.random.split(rng, 2)
    a = jax.random.uniform(rngs[0], grid_size)
    b = jax.random.uniform(rngs[1], grid_size)
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)
    epsilon = 0.1
    geometry_grid = grid.Grid(grid_size=grid_size, epsilon=epsilon)
    x, y, z = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = jnp.stack([
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geometry_mat = pointcloud.PointCloud(xyz, xyz, epsilon=epsilon)
    out_mat = sinkhorn.solve(geometry_mat, a=a, b=b)
    out_grid = sinkhorn.solve(geometry_grid, a=a, b=b)
    np.testing.assert_allclose(
        out_mat.reg_ot_cost, out_grid.reg_ot_cost, rtol=1e-5, atol=1e-5
    )

  @pytest.mark.fast.with_args("lse_mode", [False, True], only_fast=1)
  def test_apply_transport_grid(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool
  ):
    grid_size = (5, 6, 7)
    rngs = jax.random.split(rng, 4)
    a = jax.random.uniform(rngs[0], grid_size)
    b = jax.random.uniform(rngs[1], grid_size)
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)
    geom_grid = grid.Grid(grid_size=grid_size, epsilon=0.1)
    x, y, z = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = jnp.stack([
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geom_mat = pointcloud.PointCloud(xyz, xyz, epsilon=0.1)
    sink_mat = sinkhorn.solve(geom_mat, a=a, b=b)
    sink_grid = sinkhorn.solve(geom_grid, a=a, b=b)

    batch_a = 3
    batch_b = 4
    vec_a = jax.random.normal(rngs[2], [batch_a, np.prod(np.array(grid_size))])
    vec_b = jax.random.normal(rngs[3], [batch_b, np.prod(grid_size)])

    vec_a = vec_a / jnp.sum(vec_a, axis=1)[:, jnp.newaxis]
    vec_b = vec_b / jnp.sum(vec_b, axis=1)[:, jnp.newaxis]

    mat_transport_t_vec_a = geom_mat.apply_transport_from_potentials(
        sink_mat.f, sink_mat.g, vec_a, axis=0
    )
    mat_transport_vec_b = geom_mat.apply_transport_from_potentials(
        sink_mat.f, sink_mat.g, vec_b, axis=1
    )

    grid_transport_t_vec_a = geom_grid.apply_transport_from_potentials(
        sink_grid.f, sink_grid.g, vec_a, axis=0
    )
    grid_transport_vec_b = geom_grid.apply_transport_from_potentials(
        sink_grid.f, sink_grid.g, vec_b, axis=1
    )

    np.testing.assert_allclose(
        mat_transport_t_vec_a, grid_transport_t_vec_a, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        mat_transport_vec_b, grid_transport_vec_b, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(jnp.isnan(mat_transport_t_vec_a), False)

  @pytest.mark.fast()
  def test_apply_cost(self, rng: jax.random.PRNGKeyArray):
    grid_size = (5, 6, 7)

    geom_grid = grid.Grid(grid_size=grid_size, epsilon=0.1)
    x, y, z = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = jnp.stack([
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geom_mat = pointcloud.PointCloud(xyz, xyz, epsilon=0.1)

    vec = jax.random.uniform(rng, grid_size).ravel()
    np.testing.assert_allclose(
        geom_mat.apply_cost(vec),
        geom_grid.apply_cost(vec),
        rtol=1e-4,
        atol=1e-4
    )

    np.testing.assert_allclose(
        geom_grid.apply_cost(vec)[:, 0],
        np.dot(geom_mat.cost_matrix.T, vec),
        rtol=1e-4,
        atol=1e-4
    )
