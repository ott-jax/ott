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
from typing import Callable, Optional

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import costs, geometry, grid, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers import linear
from ott.solvers.linear import sinkhorn


@jax.tree_util.register_pytree_node_class
class AsymmetricCost(costs.CostFn):

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    return jnp.sum((x - y) ** 2 + x)


class TestSinkhornGrid:

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_separable_grid(self, rng: jax.Array, lse_mode: bool):
    """Two histograms in a grid of size 5 x 6 x 7  in the hypercube^3."""
    grid_size = (5, 6, 7)
    rngs = jr.split(rng, 2)
    a = jr.uniform(rngs[0], grid_size)
    b = jr.uniform(rngs[1], grid_size)
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
  def test_grid_vs_euclidean(self, rng: jax.Array, lse_mode: bool):
    grid_size = (5, 6, 7)
    rngs = jr.split(rng, 2)
    a = jr.uniform(rngs[0], grid_size)
    b = jr.uniform(rngs[1], grid_size)
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
    out_mat = linear.solve(geometry_mat, a=a, b=b, lse_mode=lse_mode)
    out_grid = linear.solve(geometry_grid, a=a, b=b, lse_mode=lse_mode)
    np.testing.assert_allclose(
        out_mat.reg_ot_cost, out_grid.reg_ot_cost, rtol=1e-5, atol=1e-5
    )

  @pytest.mark.fast.with_args("lse_mode", [False, True], only_fast=1)
  def test_apply_transport_grid(self, rng: jax.Array, lse_mode: bool):
    grid_size = (5, 6, 7)
    rngs = jr.split(rng, 4)
    a = jr.uniform(rngs[0], grid_size)
    b = jr.uniform(rngs[1], grid_size)
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
    sink_mat = linear.solve(geom_mat, a=a, b=b, lse_mode=lse_mode)
    sink_grid = linear.solve(geom_grid, a=a, b=b, lse_mode=lse_mode)

    batch_a = 3
    batch_b = 4
    vec_a = jr.normal(rngs[2], [batch_a, np.prod(np.array(grid_size))])
    vec_b = jr.normal(rngs[3], [batch_b, np.prod(grid_size)])

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
  @pytest.mark.parametrize("axis", [0, 1])
  @pytest.mark.parametrize(
      "op", [
          "cost", "linear_cost", "fn_square_cost", "square_cost", "kernel",
          "lse_kernel"
      ]
  )
  def test_apply_asymmetric_cost(self, rng: jax.Array, op: str, axis: int):
    grid_size = (3, 4)
    geom_grid = grid.Grid(
        grid_size=grid_size, cost_fns=[AsymmetricCost()], epsilon=1.0
    )
    xs = np.meshgrid(*[np.linspace(0, 1, n) for n in grid_size], indexing="ij")
    xy = jnp.stack([jnp.array(x.ravel()) for x in xs], axis=1)
    geom_pc = pointcloud.PointCloud(xy, cost_fn=AsymmetricCost())
    geom_mat = geometry.Geometry(geom_pc.cost_matrix, epsilon=1.0)

    rng_vec, rng_f, rng_g = jr.split(rng, 3)
    n = geom_grid.shape[0]
    vec = jr.uniform(rng_vec, (n,))
    f, g = jr.normal(rng_f, (n,)), jr.normal(rng_g, (n,))

    def apply(geom: geometry.Geometry) -> jnp.ndarray:
      if op == "cost":
        return geom.apply_cost(vec, axis=axis)
      if op == "linear_cost":
        fn = lambda c: 2.0 * c
        return geom.apply_cost(vec, axis=axis, fn=fn, is_linear=True)
      if op == "fn_square_cost":
        return geom.apply_cost(vec, axis=axis, fn=lambda c: c ** 2)
      if op == "square_cost":
        return geom.apply_square_cost(vec, axis=axis)
      if op == "kernel":
        return geom.apply_kernel(vec, axis=axis)
      return geom.apply_lse_kernel(f, g, 1.0, axis=axis)[0]

    np.testing.assert_allclose(
        apply(geom_grid), apply(geom_mat), rtol=1e-4, atol=1e-4
    )

  @pytest.mark.fast()
  def test_apply_square_cost(self, rng: jax.Array):
    grid_size = (5, 6, 7)
    geom_grid = grid.Grid(grid_size=grid_size)
    xs = np.meshgrid(*[np.linspace(0, 1, n) for n in grid_size], indexing="ij")
    xyz = jnp.stack([jnp.array(x.ravel()) for x in xs], axis=1)
    geom_mat = pointcloud.PointCloud(xyz)

    arr = jr.uniform(rng, (geom_grid.shape[0], 2))
    np.testing.assert_allclose(
        geom_grid.apply_square_cost(arr),
        geom_mat.apply_square_cost(arr),
        rtol=1e-4,
        atol=1e-4
    )

  @pytest.mark.fast()
  @pytest.mark.parametrize(("fn", "is_linear"), [(None, False),
                                                 (lambda c: 2.0 * c, True),
                                                 (lambda c: c ** 2, False)],
                           ids=["none", "linear", "square"])
  def test_apply_cost(
      self, rng: jax.Array, fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
      is_linear: bool
  ):
    grid_size = (5, 6, 7)

    geom_grid = grid.Grid(grid_size=grid_size, epsilon=0.1)
    x, y, z = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
    xyz = jnp.stack([
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]).transpose()
    geom_mat = pointcloud.PointCloud(xyz, xyz, epsilon=0.1)

    vec = jr.uniform(rng, grid_size).ravel()
    cost_matrix = geom_mat.cost_matrix
    if fn is not None:
      cost_matrix = fn(cost_matrix)
    np.testing.assert_allclose(
        geom_mat.apply_cost(vec, fn=fn, is_linear=is_linear),
        geom_grid.apply_cost(vec, fn=fn, is_linear=is_linear),
        rtol=1e-4,
        atol=1e-4
    )

    np.testing.assert_allclose(
        geom_grid.apply_cost(vec, fn=fn, is_linear=is_linear),
        np.dot(cost_matrix.T, vec),
        rtol=1e-4,
        atol=1e-4
    )
