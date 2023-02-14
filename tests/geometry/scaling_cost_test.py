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
"""Tests for the option to scale the cost matrix."""
from typing import Optional, Union

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import geometry, low_rank, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr


class TestScaleCost:

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jnp.ndarray):
    self.dim = 4
    self.n = 7
    self.m = 9
    self.rng, *rngs = jax.random.split(rng, 8)
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    self.a = jax.random.uniform(rngs[2], (self.n,))
    self.b = jax.random.uniform(rngs[3], (self.m,))
    self.cost = ((self.x[:, None, :] - self.y[None, :, :]) ** 2).sum(-1)
    self.vec = jax.random.uniform(rngs[4], (self.m,))
    self.cost1 = jax.random.uniform(rngs[5], (self.n, 2))
    self.cost2 = jax.random.uniform(rngs[6], (self.m, 2))
    self.cost_lr = jnp.max(jnp.dot(self.cost1, self.cost2.T))
    self.eps = 5e-2

  @pytest.mark.fast.with_args(
      scale=["median", "mean", "max_cost", "max_norm", "max_bound", 100.],
      batch_size=[None, 4],
      only_fast=[0, -3],
  )
  def test_scale_cost_pointcloud(
      self, scale: Union[str, float], batch_size: Optional[int]
  ):
    """Test various scale cost options for pointcloud."""

    def apply_sinkhorn(
        x: jnp.ndarray, y: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray,
        scale_cost: Union[str, float]
    ):
      geom = pointcloud.PointCloud(
          x, y, epsilon=self.eps, scale_cost=scale_cost
      )
      prob = linear_problem.LinearProblem(geom, a, b)
      solver = sinkhorn.Sinkhorn()
      out = solver(prob)
      transport = geom.transport_from_potentials(out.f, out.g)
      return geom, out, transport

    if scale == "median" and batch_size is not None:
      pytest.skip("Median scaling for online is not implemented")

    geom0, _, _ = apply_sinkhorn(self.x, self.y, self.a, self.b, scale_cost=1.0)

    geom, out, transport = apply_sinkhorn(
        self.x, self.y, self.a, self.b, scale_cost=scale
    )

    apply_cost_vec = geom.apply_cost(self.vec, axis=1)
    apply_transport_vec = geom.apply_transport_from_potentials(
        out.f, out.g, self.vec, axis=1
    )

    np.testing.assert_allclose(
        jnp.matmul(transport, self.vec), apply_transport_vec, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom0.apply_cost(self.vec, axis=1) * geom.inv_scale_cost,
        apply_cost_vec,
        rtol=1e-4
    )

  @pytest.mark.parametrize(
      "scale", ["mean", "max_cost", "max_norm", "max_bound", 100.]
  )
  def test_online_matches_offline_pointcloud(self, scale: Union[str, float]):
    """Tests that the scale factors for online matches the ones without."""
    geom0 = pointcloud.PointCloud(
        self.x, self.y, epsilon=self.eps, scale_cost=scale, batch_size=4
    )
    geom1 = pointcloud.PointCloud(
        self.x, self.y, epsilon=self.eps, scale_cost=scale, batch_size=None
    )
    geom2 = pointcloud.PointCloud(
        self.x, self.y, epsilon=self.eps, scale_cost=scale, batch_size=1024
    )
    np.testing.assert_allclose(
        geom0.inv_scale_cost, geom1.inv_scale_cost, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom2.inv_scale_cost, geom1.inv_scale_cost, rtol=1e-4
    )
    if scale == 'mean':
      np.testing.assert_allclose(1.0, geom1.cost_matrix.mean(), rtol=1e-4)
    elif scale == 'max_cost':
      np.testing.assert_allclose(1.0, geom1.cost_matrix.max(), rtol=1e-4)

  @pytest.mark.fast.with_args(
      "scale", ["median", "mean", "max_cost", 100.], only_fast=1
  )
  def test_scale_cost_geometry(self, scale: Union[str, float]):
    """Test various scale cost options for geometry."""

    def apply_sinkhorn(
        cost: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray,
        scale_cost: Union[str, float]
    ):
      geom = geometry.Geometry(cost, epsilon=self.eps, scale_cost=scale_cost)
      prob = linear_problem.LinearProblem(geom, a, b)
      solver = sinkhorn.Sinkhorn()
      out = solver(prob)
      transport = geom.transport_from_potentials(out.f, out.g)
      return geom, out, transport

    geom0 = geometry.Geometry(self.cost, epsilon=1e-2, scale_cost=1.0)

    geom, out, transport = apply_sinkhorn(
        self.cost, self.a, self.b, scale_cost=scale
    )

    apply_cost_vec = geom.apply_cost(self.vec, axis=1)
    apply_transport_vec = geom.apply_transport_from_potentials(
        out.f, out.g, self.vec, axis=1
    )

    np.testing.assert_allclose(
        jnp.matmul(transport, self.vec), apply_transport_vec, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom0.apply_cost(self.vec, axis=1) * geom.inv_scale_cost,
        apply_cost_vec,
        rtol=1e-4
    )

  @pytest.mark.fast.with_args(
      "scale", ["mean", "max_bound", "max_cost", 100.], only_fast=2
  )
  def test_scale_cost_low_rank(self, scale: Union[str, float]):
    """Test various scale cost options for low rank."""

    def apply_sinkhorn(cost1, cost2, scale_cost):
      geom = low_rank.LRCGeometry(cost1, cost2, scale_cost=scale_cost)
      ot_prob = linear_problem.LinearProblem(geom, self.a, self.b)
      solver = sinkhorn_lr.LRSinkhorn(rank=5, threshold=1e-3)
      out = solver(ot_prob)
      return geom, out

    geom0 = low_rank.LRCGeometry(self.cost1, self.cost2, scale_cost=1.0)

    geom, out = jax.jit(
        apply_sinkhorn, static_argnums=2
    )(self.cost1, self.cost2, scale_cost=scale)

    apply_cost_vec = geom._apply_cost_to_vec(self.vec, axis=1)
    apply_transport_vec = out.apply(self.vec, axis=1)
    transport = out.matrix

    np.testing.assert_allclose(
        jnp.matmul(transport, self.vec), apply_transport_vec, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom0._apply_cost_to_vec(self.vec, axis=1) * geom.inv_scale_cost,
        apply_cost_vec,
        rtol=1e-4
    )

    if scale == 'mean':
      np.testing.assert_allclose(1.0, geom.cost_matrix.mean(), rtol=1e-4)
    if scale == 'max_cost':
      np.testing.assert_allclose(1.0, geom.cost_matrix.max(), rtol=1e-4)

  @pytest.mark.parametrize("batch_size", [5, 12])
  def test_max_scale_cost_low_rank_with_batch(self, batch_size: int):
    """Test max_cost options for low rank with batch_size fixed."""

    geom0 = low_rank.LRCGeometry(
        self.cost1, self.cost2, scale_cost='max_cost', batch_size=batch_size
    )

    np.testing.assert_allclose(
        geom0.inv_scale_cost, 1.0 / jnp.max(self.cost_lr), rtol=1e-4
    )

  def test_max_scale_cost_low_rank_large_array(self):
    """Test max_cost options for large matrices."""

    _, *keys = jax.random.split(self.rng, 3)
    cost1 = jax.random.uniform(keys[0], (10000, 2))
    cost2 = jax.random.uniform(keys[1], (11000, 2))
    max_cost_lr = jnp.max(jnp.dot(cost1, cost2.T))

    geom0 = low_rank.LRCGeometry(cost1, cost2, scale_cost='max_cost')

    np.testing.assert_allclose(
        geom0.inv_scale_cost, 1.0 / max_cost_lr, rtol=1e-4
    )
