# coding=utf-8
# Copyright 2021 Google LLC.
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

# Lint as: python3
"""Tests for the Sinkhorn divergence."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.test_util
from ott.geometry import geometry
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence


class SinkhornDivergenceTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self._dim = 4
    self._num_points = 30, 37
    self.rng, *rngs = jax.random.split(self.rng, 3)
    a = jax.random.uniform(rngs[0], (self._num_points[0],))
    b = jax.random.uniform(rngs[1], (self._num_points[1],))
    self._a = a / jnp.sum(a)
    self._b = b / jnp.sum(b)

  def test_euclidean_point_cloud(self):
    rngs = jax.random.split(self.rng, 2)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    geometry_xx = pointcloud.PointCloud(x, x, epsilon=0.01)
    geometry_xy = pointcloud.PointCloud(x, y, epsilon=0.01)
    geometry_yy = pointcloud.PointCloud(y, y, epsilon=0.01)
    div = sinkhorn_divergence._sinkhorn_divergence(
        geometry_xy,
        geometry_xx,
        geometry_yy,
        self._a,
        self._b,
        threshold=1e-2)
    self.assertGreater(div.divergence, 0.0)
    self.assertLen(div.potentials, 3)

    # Test symmetric setting,
    # test that symmetric evaluation converges earlier/better.
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x, x, epsilon=1e-1,
        sinkhorn_kwargs={'inner_iterations': 1})
    self.assertAllClose(div.divergence, 0.0, rtol=1e-5, atol=1e-5)
    iters_xx = jnp.sum(div.errors[0] > 0)
    iters_xx_sym = jnp.sum(div.errors[1] > 0)
    self.assertGreater(iters_xx, iters_xx_sym)

  def test_euclidean_autoepsilon(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a, cloud_b,
        a=self._a, b=self._b,
        sinkhorn_kwargs=dict(threshold=1e-2))
    self.assertGreater(div.divergence, 0.0)
    self.assertLen(div.potentials, 3)
    self.assertLen(div.geoms, 3)
    print(div.geoms[0].epsilon, div.geoms[1].epsilon)
    print(div.geoms[0]._epsilon._scale, div.geoms[1]._epsilon._scale)
    self.assertAllClose(div.geoms[0].epsilon, div.geoms[1].epsilon)

  def test_euclidean_autoepsilon_not_share_epsilon(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a, cloud_b,
        a=self._a, b=self._b,
        sinkhorn_kwargs=dict(threshold=1e-2), share_epsilon=False)
    self.assertGreater(jnp.abs(div.geoms[0].epsilon - div.geoms[1].epsilon), 0)

  def test_euclidean_point_cloud_wrapper(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a, cloud_b, epsilon=0.1,
        a=self._a, b=self._b,
        sinkhorn_kwargs=dict(threshold=1e-2))
    self.assertGreater(div.divergence, 0.0)
    self.assertLen(div.potentials, 3)
    self.assertLen(div.geoms, 3)

  def test_euclidean_point_cloud_wrapper_no_weights(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a, cloud_b, epsilon=0.1,
        sinkhorn_kwargs=dict(threshold=1e-2))
    self.assertGreater(div.divergence, 0.0)
    self.assertLen(div.potentials, 3)
    self.assertLen(div.geoms, 3)

  def test_euclidean_point_cloud_unbalanced_wrapper(self):
    rngs = jax.random.split(self.rng, 2)
    cloud_a = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    cloud_b = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        cloud_a, cloud_b, epsilon=0.1,
        a=self._a +.001, b=self._b +.002,
        sinkhorn_kwargs=dict(threshold=1e-2, tau_a=0.8, tau_b=0.9))
    self.assertGreater(div.divergence, 0.0)
    self.assertLen(div.potentials, 3)
    self.assertLen(div.geoms, 3)

  def test_generic_point_cloud_wrapper(self):
    rngs = jax.random.split(self.rng, 2)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))

    # Tests with 3 cost matrices passed as args
    cxy = jnp.sum(jnp.abs(x[:, jnp.newaxis] - y[jnp.newaxis, :])**2, axis=2)
    cxx = jnp.sum(jnp.abs(x[:, jnp.newaxis] - x[jnp.newaxis, :])**2, axis=2)
    cyy = jnp.sum(jnp.abs(y[:, jnp.newaxis] - y[jnp.newaxis, :])**2, axis=2)
    div = sinkhorn_divergence.sinkhorn_divergence(
        geometry.Geometry,
        cxy, cxx, cyy, epsilon=0.1,
        a=self._a, b=self._b,
        sinkhorn_kwargs=dict(threshold=1e-2))
    self.assertIsNotNone(div.divergence)
    self.assertLen(div.potentials, 3)
    self.assertLen(div.geoms, 3)

    # Tests with 2 cost matrices passed as args
    div = sinkhorn_divergence.sinkhorn_divergence(
        geometry.Geometry,
        cxy, cxx, epsilon=0.1,
        a=self._a, b=self._b,
        sinkhorn_kwargs=dict(threshold=1e-2))
    self.assertIsNotNone(div.divergence)
    self.assertLen(div.potentials, 3)
    self.assertLen(div.geoms, 3)

    # Tests with 3 cost matrices passed as kwargs
    div = sinkhorn_divergence.sinkhorn_divergence(
        geometry.Geometry,
        cost_matrix=(cxy, cxx, cyy), epsilon=0.1,
        a=self._a, b=self._b,
        sinkhorn_kwargs=dict(threshold=1e-2))
    self.assertIsNotNone(div.divergence)
    self.assertLen(div.potentials, 3)
    self.assertLen(div.geoms, 3)

  def test_segment_sinkhorn_result(self):

    # Test that segmented sinkhorn gives the same results:
    rngs = jax.random.split(self.rng, 4)
    x = jax.random.uniform(rngs[0], (self._num_points[0], self._dim))
    y = jax.random.uniform(rngs[1], (self._num_points[1], self._dim))
    geom_kwargs = dict(epsilon=0.01)
    sinkhorn_kwargs = dict(threshold=1e-2)
    geometry = pointcloud.PointCloud
    true_divergence = sinkhorn_divergence.sinkhorn_divergence(
        geometry,
        x,
        y,
        a=self._a,
        b=self._b,
        sinkhorn_kwargs=sinkhorn_kwargs,
        **geom_kwargs).divergence

    for shuffle in [False, True]:
      if shuffle:
        # Now, shuffle the order of both arrays, but
        # still maintain the segment assignments:
        idx_x = jax.random.shuffle(rngs[2], jnp.arange(x.shape[0] * 2))
        idx_y = jax.random.shuffle(rngs[3], jnp.arange(y.shape[0] * 2))
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

      segmented_divergences = sinkhorn_divergence.segment_sinkhorn_divergence(
          x_copied,
          y_copied,
          segment_ids_x=segment_ids_x,
          segment_ids_y=segment_ids_y,
          indices_are_sorted=False,
          weights_x=a_copied,
          weights_y=b_copied,
          sinkhorn_kwargs=sinkhorn_kwargs,
          **geom_kwargs)

      self.assertArraysAllClose(
          true_divergence.repeat(2), segmented_divergences)

  def test_segment_sinkhorn_different_segment_sizes(self):

    # Test other array sizes
    x1 = jnp.arange(10)[:, None].repeat(2, axis=1)
    y1 = jnp.arange(11)[:, None].repeat(2, axis=1) + 0.1

    # Should have larger divergence since further apart:
    x2 = jnp.arange(12)[:, None].repeat(2, axis=1)
    y2 = 2 * jnp.arange(13)[:, None].repeat(2, axis=1) + 0.1

    segmented_divergences = sinkhorn_divergence.segment_sinkhorn_divergence(
        jnp.concatenate((x1, x2)),
        jnp.concatenate((y1, y2)),
        num_per_segment_x=jnp.array([10, 12]),
        num_per_segment_y=jnp.array([11, 13]),
        epsilon=0.01)

    self.assertEqual(segmented_divergences.shape[0], 2)
    self.assertGreater(segmented_divergences[1], segmented_divergences[0])

    true_divergences = jnp.array([
        sinkhorn_divergence.sinkhorn_divergence(
            pointcloud.PointCloud, x, y, epsilon=0.01).divergence
        for x, y in zip((x1, x2), (y1, y2))
    ])
    self.assertArraysAllClose(segmented_divergences, true_divergences)


if __name__ == '__main__':
  absltest.main()
