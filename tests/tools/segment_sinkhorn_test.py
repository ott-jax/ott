# Copyright 2022 The OTT Authors.
#
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
"""Tests for Segmented Sinkhorn."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.geometry import costs, pointcloud
from ott.solvers.linear import sinkhorn
from ott.tools import segment_sinkhorn
from ott.tools.gaussian_mixture import gaussian_mixture


class TestSegmentSinkhorn:

  @pytest.fixture(autouse=True)
  def setUp(self, rng: jnp.ndarray):
    self._dim = 4
    self._num_points = 13, 17
    self._max_measure_size = 20
    self.rng, *rngs = jax.random.split(rng, 3)
    a = jax.random.uniform(rngs[0], (self._num_points[0],))
    b = jax.random.uniform(rngs[1], (self._num_points[1],))
    self._a = a / jnp.sum(a)
    self._b = b / jnp.sum(b)

  @pytest.mark.parametrize("shuffle", [False, True])
  def test_segment_sinkhorn_result(self, shuffle: bool):
    # Test that segmented sinkhorn gives the same results as run separately:
    rngs = jax.random.split(self.rng, 4)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    geom_kwargs = dict(epsilon=0.014)
    sinkhorn_kwargs = dict(threshold=.2e-2)
    true_regotcost = sinkhorn.sinkhorn(
        pointcloud.PointCloud(x, y, **geom_kwargs),
        a=self._a,
        b=self._b,
        **sinkhorn_kwargs
    ).reg_ot_cost

    if shuffle:
      # Now, shuffle the order of both arrays, but
      # still maintain the segment assignments:
      idx_x = jax.random.permutation(
          rngs[2], jnp.arange(x.shape[0] * 2), independent=True
      )
      idx_y = jax.random.permutation(
          rngs[3], jnp.arange(y.shape[0] * 2), independent=True
      )
    else:
      idx_x = jnp.arange(x.shape[0] * 2)
      idx_y = jnp.arange(y.shape[0] * 2)

    # Duplicate arrays:
    x_copied = jnp.concatenate((x, x))[idx_x]
    a_copied = jnp.concatenate((self._a, self._a))[idx_x]
    segment_ids_x = jnp.arange(2).repeat(x.shape[0])[idx_x]

    y_copied = jnp.concatenate((y, y))[idx_y]
    b_copied = jnp.concatenate((self._b, self._b))[idx_y]
    segment_ids_y = jnp.arange(2).repeat(y.shape[0])[idx_y]

    segmented_regotcost = segment_sinkhorn.segment_sinkhorn(
        x_copied,
        y_copied,
        num_segments=2,
        max_measure_size=self._max_measure_size,
        segment_ids_x=segment_ids_x,
        segment_ids_y=segment_ids_y,
        indices_are_sorted=False,
        weights_x=a_copied,
        weights_y=b_copied,
        sinkhorn_kwargs=sinkhorn_kwargs,
        **geom_kwargs
    )

    np.testing.assert_allclose(true_regotcost.repeat(2), segmented_regotcost)

  def test_segment_sinkhorn_different_segment_sizes(self):
    # Test other array sizes
    x1 = jnp.arange(10)[:, None].repeat(2, axis=1)
    y1 = jnp.arange(11)[:, None].repeat(2, axis=1) + 0.1

    # Should have larger divergence since further apart:
    x2 = jnp.arange(12)[:, None].repeat(2, axis=1)
    y2 = 2 * jnp.arange(13)[:, None].repeat(2, axis=1) + 0.1

    sink = jax.jit(
        segment_sinkhorn.segment_sinkhorn,
        static_argnames=['num_segments', 'max_measure_size'],
    )
    segmented_regotcost = sink(
        jnp.concatenate((x1, x2)),
        jnp.concatenate((y1, y2)),
        num_segments=2,
        max_measure_size=14,
        num_per_segment_x=(10, 12),
        num_per_segment_y=(11, 13),
        epsilon=0.01
    )

    assert segmented_regotcost.shape[0] == 2
    assert segmented_regotcost[1] > segmented_regotcost[0]

    true_regotcost = jnp.array([
        sinkhorn.sinkhorn(pointcloud.PointCloud(x, y,
                                                epsilon=0.01),).reg_ot_cost
        for x, y in zip((x1, x2), (y1, y2))
    ])
    np.testing.assert_allclose(
        segmented_regotcost, true_regotcost, atol=1e-4, rtol=1e-4
    )

  def test_sinkhorn_divergence_segment_custom_padding(self, rng):
    rngs = jax.random.split(rng, 4)
    dim = 3
    b_cost = costs.Bures(dim)

    num_per_segment_x = (5, 2)
    num_per_segment_y = (3, 5)
    ns = num_per_segment_x + num_per_segment_y

    means_and_covs_to_x = jax.vmap(
        costs.mean_and_cov_to_x, in_axes=[0, 0, None]
    )

    def g(rng, n):
      out = gaussian_mixture.GaussianMixture.from_random(
          rng, n_components=n, n_dimensions=dim
      )
      return means_and_covs_to_x(out.loc, out.covariance, dim)

    x1, x2, y1, y2 = (g(rngs[i], ns[i]) for i in range(4))

    true_regotcost = jnp.array([
        sinkhorn.sinkhorn(
            pointcloud.PointCloud(x, y, cost_fn=b_cost, epsilon=0.1)
        ).reg_ot_cost for x, y in zip((x1, x2), (y1, y2))
    ])

    x = jnp.vstack((x1, x2))
    y = jnp.vstack((y1, y2))

    segmented_regotcost = segment_sinkhorn.segment_sinkhorn(
        x,
        y,
        num_segments=2,
        max_measure_size=5,
        cost_fn=b_cost,
        num_per_segment_x=num_per_segment_x,
        num_per_segment_y=num_per_segment_y,
        sinkhorn_kwargs={'lse_mode': True},
        epsilon=0.1,
    )
    np.testing.assert_allclose(segmented_regotcost, true_regotcost)
