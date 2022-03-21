# coding=utf-8
# Copyright 2022 Google LLC.
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
"""Tests for the soft sort tools."""
import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from ott.tools import soft_sort


class SoftSortTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self._dim = 4
    self._num_points = 20

  @parameterized.parameters([(20,)], [(20, 1)])
  def test_sort_one_array(self, shape):
    x = jax.random.uniform(self.rng, shape)
    xs = soft_sort.sort(x, axis=0)
    self.assertEqual(x.shape, xs.shape)
    self.assertTrue(jnp.alltrue(jnp.diff(xs, axis=0) >= 0.0))

  def test_sort_array_squashing_momentum(self):
    shape = (33, 1)
    x = jax.random.uniform(self.rng, shape)
    xs_lin = soft_sort.sort(
        x,
        axis=0,
        squashing_fun=lambda x: x,
        epsilon=5e-4,
        chg_momentum_from=100)
    xs_sig = soft_sort.sort(
        x, axis=0, squashing_fun=None, epsilon=2e-4, chg_momentum_from=100)
    # Notice xs_lin and xs_sig have no reason to be equal, since they use
    # different squashing functions, but they should be similar.
    # One can recover "similar" looking outputs by tuning the regularization
    # parameter slightly higher for 'lin'
    np.testing.assert_allclose(xs_sig, xs_lin, rtol=0.05, atol=0.01)
    self.assertTrue(jnp.alltrue(jnp.diff(xs_lin, axis=0) >= -1e-8))
    self.assertTrue(jnp.alltrue(jnp.diff(xs_sig, axis=0) >= -1e-8))

  @parameterized.parameters(-1, 4, 100)
  def test_topk_one_array(self, k):
    n = 20
    x = jax.random.uniform(self.rng, (n,))
    axis = 0
    xs = soft_sort.sort(
        x, axis=axis, topk=k, epsilon=1e-3, squashing_fun=lambda x: x)
    outsize = k if 0 < k < n else n
    self.assertEqual(xs.shape, (outsize,))
    self.assertTrue(jnp.alltrue(jnp.diff(xs, axis=axis) >= 0.0))
    np.testing.assert_allclose(xs, jnp.sort(x, axis=axis)[-outsize:], atol=0.01)

  @parameterized.parameters([-1, 2, 5, 11])
  def test_sort_batch(self, topk):
    x = jax.random.uniform(self.rng, (32, 10, 6, 4))
    axis = 1
    xs = soft_sort.sort(x, axis=axis, topk=topk)
    expected_shape = list(x.shape)
    expected_shape[axis] = topk if (0 < topk < x.shape[axis]) else x.shape[axis]
    self.assertEqual(xs.shape, tuple(expected_shape))
    self.assertTrue(jnp.alltrue(jnp.diff(xs, axis=axis) >= 0.0))

  def test_rank_one_array(self):
    x = jax.random.uniform(self.rng, (20,))
    ranks = soft_sort.ranks(x, epsilon=0.001)
    self.assertEqual(x.shape, ranks.shape)
    expected_ranks = jnp.argsort(jnp.argsort(x, axis=0), axis=0).astype(float)
    np.testing.assert_allclose(ranks, expected_ranks, atol=0.9, rtol=0.1)

  @parameterized.parameters([0.2, 0.9])
  def test_quantile(self, level):
    x = jnp.linspace(0.0, 1.0, 100)
    q = soft_sort.quantile(
        x, level=level, weight=0.05, epsilon=1e-3, lse_mode=True)
    self.assertAlmostEqual(q, level, places=1)

  def test_quantile_on_several_axes(self):
    batch, height, width, channels = 16, 100, 100, 3
    x = jax.random.uniform(self.rng, shape=(batch, height, width, channels))
    q = soft_sort.quantile(
        x, axis=(1, 2), level=0.5, weight=0.05, epsilon=1e-3, lse_mode=True)
    self.assertEqual(q.shape, (batch, 1, channels))
    np.testing.assert_allclose(q, 0.5 * np.ones((batch, 1, channels)), atol=3e-2)

  def test_soft_quantile_normalization(self):
    rngs = jax.random.split(self.rng, 2)
    x = jax.random.uniform(rngs[0], shape=(100,))
    mu, sigma = 2.0, 1.2
    y = mu + sigma * jax.random.normal(self.rng, shape=(48,))
    mu_target, sigma_target = y.mean(), y.std()
    qn = soft_sort.quantile_normalization(x, jnp.sort(y), epsilon=1e-4)
    mu_transform, sigma_transform = qn.mean(), qn.std()
    np.testing.assert_allclose([mu_transform, sigma_transform],
                        [mu_target, sigma_target],
                        rtol=0.05)

  def test_sort_with(self):
    n, d = 20, 4
    inputs = jax.random.uniform(self.rng, shape=(n, d))
    criterion = jnp.linspace(0.1, 1.2, n)
    output = soft_sort.sort_with(inputs, criterion, epsilon=1e-4)
    self.assertEqual(output.shape, inputs.shape)
    np.testing.assert_allclose(output, inputs, atol=0.05)

    k = 4
    # investigate why epsilon=1e-4 fails
    output = soft_sort.sort_with(inputs, criterion, topk=k, epsilon=1e-3)
    self.assertEqual(output.shape, (k, d))
    np.testing.assert_allclose(output, inputs[-k:], atol=0.05)

  def test_quantize(self):
    n = 100
    inputs = jnp.linspace(0.0, 1.0, n)[..., None]
    q = soft_sort.quantize(inputs, num_levels=4, axis=0, epsilon=1e-4)
    delta = jnp.abs(q - jnp.array([0.12, 0.34, 0.64, 0.86]))
    min_distances = jnp.min(delta, axis=1)
    np.testing.assert_allclose(min_distances, jnp.zeros_like(min_distances), atol=0.05)

  @parameterized.parameters([True, False])
  def test_soft_sort_jacobian(self, implicit: bool):
    b, n = 10, 40
    idx_column = 5
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 3)
    z = jax.random.uniform(rngs[0], ((b, n)))
    random_dir = jax.random.normal(rngs[1], (b,)) / b

    def loss_fn(logits):
      ranks_fn = functools.partial(
          soft_sort.ranks,
          axis=-1,
          num_targets=167,
          implicit_differentiation=implicit)
      return jnp.sum(ranks_fn(logits)[:, idx_column] * random_dir)

    _, grad = jax.jit(jax.value_and_grad(loss_fn))(z)
    delta = jax.random.uniform(rngs[2], z.shape) - 0.5
    eps = 1e-3
    val_peps = loss_fn(z + eps * delta)
    val_meps = loss_fn(z - eps * delta)
    np.testing.assert_allclose((val_peps - val_meps) / (2 * eps),
                        jnp.sum(grad * delta),
                        atol=0.01, rtol=0.1)


if __name__ == '__main__':
  absltest.main()
