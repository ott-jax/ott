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
import dataclasses
from typing import Optional, Union

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import geometry, low_rank, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from tests import _utils

EPS = 5e-2


@dataclasses.dataclass(frozen=True)
class ScaleCostData:
  """Inputs shared by every scale-cost test."""
  x: jnp.ndarray  # (n, dim)
  y: jnp.ndarray  # (m, dim)
  a: jnp.ndarray  # (n,), deliberately not normalized
  b: jnp.ndarray  # (m,), deliberately not normalized
  vec: jnp.ndarray  # (m,)
  cost1: jnp.ndarray  # (n, 2), low-rank cost factor
  cost2: jnp.ndarray  # (m, 2), low-rank cost factor

  @property
  def cost(self) -> jnp.ndarray:
    """Squared Euclidean cost matrix between :attr:`x` and :attr:`y`."""
    return ((self.x[:, None, :] - self.y[None, :, :]) ** 2).sum(-1)


@pytest.fixture(scope="module")
def data() -> ScaleCostData:
  """Inputs drawn exactly as this module always has."""
  n, m, dim = 7, 9, 4
  _, *rngs = jr.split(_utils.root_key(), 8)
  return ScaleCostData(
      x=jr.uniform(rngs[0], (n, dim)),
      y=jr.uniform(rngs[1], (m, dim)),
      a=jr.uniform(rngs[2], (n,)),
      b=jr.uniform(rngs[3], (m,)),
      vec=jr.uniform(rngs[4], (m,)),
      cost1=jr.uniform(rngs[5], (n, 2)),
      cost2=jr.uniform(rngs[6], (m, 2)),
  )


class TestScaleCost:

  @pytest.mark.fast.with_args(
      scale=["median", "mean", "max_cost", "max_norm", "max_bound", 100.0],
      batch_size=[None, 4],
      only_fast=[0, -3],
  )
  def test_scale_cost_pointcloud(
      self, data: ScaleCostData, scale: Union[str, float],
      batch_size: Optional[int]
  ):
    """Test various scale cost options for pointcloud."""

    def apply_sinkhorn(
        x: jnp.ndarray, y: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray,
        scale_cost: Union[str, float]
    ):
      geom = pointcloud.PointCloud(x, y, epsilon=EPS, scale_cost=scale_cost)
      prob = linear_problem.LinearProblem(geom, a, b)
      solver = sinkhorn.Sinkhorn()
      out = solver(prob)
      transport = geom.transport_from_potentials(out.f, out.g)
      return geom, out, transport

    if scale == "median" and batch_size is not None:
      pytest.skip("Median scaling for online is not implemented")

    geom0, _, _ = apply_sinkhorn(data.x, data.y, data.a, data.b, scale_cost=1.0)

    geom, out, transport = apply_sinkhorn(
        data.x, data.y, data.a, data.b, scale_cost=scale
    )

    apply_cost_vec = geom.apply_cost(data.vec, axis=1)
    apply_transport_vec = geom.apply_transport_from_potentials(
        out.f, out.g, data.vec, axis=1
    )

    np.testing.assert_allclose(
        jnp.matmul(transport, data.vec), apply_transport_vec, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom0.apply_cost(data.vec, axis=1) * geom.inv_scale_cost,
        apply_cost_vec,
        rtol=1e-4
    )

  @pytest.mark.parametrize(
      "scale", ["mean", "max_cost", "max_norm", "max_bound", 100.0]
  )
  def test_online_matches_offline_pointcloud(
      self, data: ScaleCostData, scale: Union[str, float]
  ):
    """Tests that the scale factors for online matches the ones without."""
    geom0 = pointcloud.PointCloud(
        data.x, data.y, epsilon=EPS, scale_cost=scale, batch_size=4
    )
    geom1 = pointcloud.PointCloud(
        data.x, data.y, epsilon=EPS, scale_cost=scale, batch_size=None
    )
    geom2 = pointcloud.PointCloud(
        data.x, data.y, epsilon=EPS, scale_cost=scale, batch_size=1024
    )
    np.testing.assert_allclose(
        geom0.inv_scale_cost, geom1.inv_scale_cost, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom2.inv_scale_cost, geom1.inv_scale_cost, rtol=1e-4
    )
    if scale == "mean":
      np.testing.assert_allclose(1.0, geom1.cost_matrix.mean(), rtol=1e-4)
    elif scale == "max_cost":
      np.testing.assert_allclose(1.0, geom1.cost_matrix.max(), rtol=1e-4)

  @pytest.mark.fast.with_args(
      "scale", ["median", "mean", "max_cost", 100.0], only_fast=1
  )
  def test_scale_cost_geometry(
      self, data: ScaleCostData, scale: Union[str, float]
  ):
    """Test various scale cost options for geometry."""

    def apply_sinkhorn(
        cost: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray,
        scale_cost: Union[str, float]
    ):
      geom = geometry.Geometry(cost, epsilon=EPS, scale_cost=scale_cost)
      prob = linear_problem.LinearProblem(geom, a, b)
      solver = sinkhorn.Sinkhorn()
      out = solver(prob)
      transport = geom.transport_from_potentials(out.f, out.g)
      return geom, out, transport

    geom0 = geometry.Geometry(data.cost, epsilon=1e-2, scale_cost=1.0)

    geom, out, transport = apply_sinkhorn(
        data.cost, data.a, data.b, scale_cost=scale
    )

    apply_cost_vec = geom.apply_cost(data.vec, axis=1)
    apply_transport_vec = geom.apply_transport_from_potentials(
        out.f, out.g, data.vec, axis=1
    )

    np.testing.assert_allclose(
        jnp.matmul(transport, data.vec), apply_transport_vec, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom0.apply_cost(data.vec, axis=1) * geom.inv_scale_cost,
        apply_cost_vec,
        rtol=1e-4
    )

  @pytest.mark.fast.with_args(
      "scale", ["mean", "max_bound", "max_cost", 100.0], only_fast=2
  )
  def test_scale_cost_low_rank(
      self, data: ScaleCostData, scale: Union[str, float]
  ):
    """Test various scale cost options for low rank."""

    def apply_sinkhorn(cost1, cost2, scale_cost):
      geom = low_rank.LRCGeometry(cost1, cost2, scale_cost=scale_cost)
      ot_prob = linear_problem.LinearProblem(geom, data.a, data.b)
      solver = sinkhorn_lr.LRSinkhorn(rank=5, threshold=1e-3)
      out = solver(ot_prob)
      return geom, out

    geom0 = low_rank.LRCGeometry(data.cost1, data.cost2, scale_cost=1.0)

    geom, out = jax.jit(
        apply_sinkhorn, static_argnums=2
    )(data.cost1, data.cost2, scale_cost=scale)

    apply_cost_vec = geom._apply_cost_to_vec(data.vec, axis=1)
    apply_transport_vec = out.apply(data.vec, axis=1)
    transport = out.matrix

    np.testing.assert_allclose(
        jnp.matmul(transport, data.vec), apply_transport_vec, rtol=1e-4
    )
    np.testing.assert_allclose(
        geom0._apply_cost_to_vec(data.vec, axis=1) * geom.inv_scale_cost,
        apply_cost_vec,
        rtol=1e-4
    )

    if scale == "mean":
      np.testing.assert_allclose(1.0, geom.cost_matrix.mean(), rtol=1e-4)
    if scale == "max_cost":
      np.testing.assert_allclose(1.0, geom.cost_matrix.max(), rtol=1e-4)

  def test_max_scale_cost_low_rank_large_array(self, rng: jax.Array):
    """Test max_cost options for large matrices."""

    _, *rngs = jr.split(rng, 3)
    cost1 = jr.uniform(rngs[0], (10000, 2))
    cost2 = jr.uniform(rngs[1], (11000, 2))
    max_cost_lr = jnp.max(jnp.dot(cost1, cost2.T))

    geom0 = low_rank.LRCGeometry(cost1, cost2, scale_cost="max_cost")

    np.testing.assert_allclose(
        geom0.inv_scale_cost, 1.0 / max_cost_lr, rtol=1e-4
    )
