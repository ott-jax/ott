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
"""Tests for the Policy."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as np
import jax.test_util

from ott.core.ground_geometry import grid
from ott.core.ground_geometry import pointcloud
from ott.tools import discrete_barycenter as db


class DiscreteBarycenterTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.named_parameters(
      dict(
          testcase_name='lse-deb',
          lse_mode=True,
          debiased=True,
          epsilon=0.01),
      dict(
          testcase_name='scal-no-deb',
          lse_mode=False,
          debiased=False,
          epsilon=0.02))
  def test_discrete_barycenter_grid(self, lse_mode, debiased, epsilon):
    """Tests the discrete barycenters on a 5x5x5 grid.

    Puts two masses on opposing ends of the hypercube with small noise in
    between. Check that their W barycenter sits (mostly) at the middle of the
    hypercube (e.g. index (5x5x5-1)/2)

    Args:
      lse_mode: bool, lse or scaling computations.
      debiased: bool, use (or not) debiasing as proposed in
      https://arxiv.org/abs/2006.02575
      epsilon: float, regularization parameter
    """
    size = np.array([5, 5, 5])
    grid_3d = grid.Grid(grid_size=size, epsilon=epsilon)
    a = np.ones(size)
    b = np.ones(size)
    a = a.ravel()
    b = b.ravel()
    a = jax.ops.index_update(a, 0, 10000)
    b = jax.ops.index_update(b, -1, 10000)
    a = a / np.sum(a)
    b = b / np.sum(b)
    threshold = 1e-2
    _, _, bar, errors = db.discrete_barycenter(
        grid_3d, a=np.stack((a, b)), threshold=threshold,
        lse_mode=lse_mode,
        debiased=debiased)
    self.assertGreater(bar[(np.prod(size) - 1) // 2], 0.7)
    self.assertGreater(1, bar[(np.prod(size) - 1) // 2])
    err = errors[np.isfinite(errors)][-1]
    self.assertGreater(threshold, err)

  @parameterized.named_parameters(
      dict(
          testcase_name='lse',
          lse_mode=True,
          epsilon=0.001),
      dict(
          testcase_name='scal',
          lse_mode=False,
          epsilon=0.01))
  def test_discrete_barycenter_pointcloud(self, lse_mode, epsilon):
    """Tests the discrete barycenters on pointclouds.

    Two measures supported on the same set of points (a 1D grid), barycenter is
    evaluated on a different set of points (still in 1D).

    Args:
      lse_mode: bool, lse or scaling computations
      epsilon: float
    """
    n = 50
    ma = 0.2
    mb = 0.8
    # define two narrow Gaussian bumps in segment [0,1]
    a = np.exp(-(np.arange(0, n) / (n - 1) - ma)**2 / .01) + 1e-10
    b = np.exp(-(np.arange(0, n) / (n - 1) - mb)**2 / .01) + 1e-10
    a = a / np.sum(a)
    b = b / np.sum(b)

    # positions on the real line where weights are supported.
    x = np.atleast_2d(np.arange(0, n) / (n - 1)).T

    # choose a different support, half the size, for the barycenter.
    # note this is the reason why we do not use debiasing in this case.
    x_support_bar = np.atleast_2d((np.arange(0, (n / 2)) /
                                   (n / 2 - 1) - .5) * .9 + .5).T

    geom = pointcloud.PointCloudGeometry(x, x_support_bar, epsilon=epsilon)
    bar = db.discrete_barycenter(
        geom, a=np.stack((a, b)), lse_mode=lse_mode).histogram
    # check the barycenter has bump in the middle.
    self.assertGreater(bar[n // 4], 0.1)

if __name__ == '__main__':
  absltest.main()
