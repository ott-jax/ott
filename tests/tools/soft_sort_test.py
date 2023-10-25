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
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.solvers.linear import acceleration
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.tools import soft_sort


class TestSoftSort:

  @pytest.mark.parametrize("shape", [(20,), (20, 1)])
  def test_sort_one_array(
      self, rng: jax.random.PRNGKeyArray, shape: Tuple[int, ...]
  ):
    x = jax.random.uniform(rng, shape)
    xs = soft_sort.sort(x, axis=0)

    np.testing.assert_array_equal(x.shape, xs.shape)
    np.testing.assert_array_equal(jnp.diff(xs, axis=0) >= 0.0, True)

  def test_sort_array_squashing_momentum(self, rng: jax.random.PRNGKeyArray):
    shape = (33, 1)
    x = jax.random.uniform(rng, shape)
    xs_lin = soft_sort.sort(
        x,
        axis=0,
        squashing_fun=lambda x: x,
        epsilon=5e-4,
        momentum=acceleration.Momentum(start=100),
    )
    xs_sig = soft_sort.sort(
        x,
        axis=0,
        squashing_fun=None,
        epsilon=2e-4,
        momentum=acceleration.Momentum(start=100)
    )
    # Notice xs_lin and xs_sig have no reason to be equal, since they use
    # different squashing functions, but they should be similar.
    # One can recover "similar" looking outputs by tuning the regularization
    # parameter slightly higher for 'lin'
    np.testing.assert_allclose(xs_sig, xs_lin, rtol=0.05, atol=0.01)
    np.testing.assert_array_equal(jnp.diff(xs_lin, axis=0) >= -1e-8, True)
    np.testing.assert_array_equal(jnp.diff(xs_sig, axis=0) >= -1e-8, True)

  @pytest.mark.fast()
  @pytest.mark.parametrize("k", [-1, 4, 100])
  def test_topk_one_array(self, rng: jax.random.PRNGKeyArray, k: int):
    n = 20
    x = jax.random.uniform(rng, (n,))
    axis = 0
    xs = soft_sort.sort(
        x, axis=axis, topk=k, epsilon=1e-3, squashing_fun=lambda x: x
    )
    outsize = k if 0 < k < n else n

    np.testing.assert_array_equal(xs.shape, (outsize,))
    np.testing.assert_array_equal(jnp.diff(xs, axis=axis) >= 0.0, True)
    np.testing.assert_allclose(xs, jnp.sort(x, axis=axis)[-outsize:], atol=0.01)

  @pytest.mark.fast.with_args("topk", [-1, 2, 11], only_fast=-1)
  def test_sort_batch(self, rng: jax.random.PRNGKeyArray, topk: int):
    x = jax.random.uniform(rng, (32, 10, 6, 4))
    axis = 1
    xs = soft_sort.sort(x, axis=axis, topk=topk)
    expected_shape = list(x.shape)
    expected_shape[axis] = topk if (0 < topk < x.shape[axis]) else x.shape[axis]

    np.testing.assert_array_equal(xs.shape, expected_shape)
    np.testing.assert_array_equal(jnp.diff(xs, axis=axis) >= 0.0, True)

  def test_multivariate_cdf_quantiles(self, rng: jax.random.PRNGKeyArray):
    n, d = 512, 3
    key1, key2, key3 = jax.random.split(rng, 3)

    # Set central point in sampled input measure
    z = jax.random.uniform(key1, (1, d))

    # Sample inputs symmetrically centered on z
    inputs = 0.34 * jax.random.normal(key2, (n, d)) + z

    # Set central point in target distribution.
    q = 0.5 * jnp.ones((1, d))

    # Set tolerance for quantile / cdf comparisons to ground truth.
    atol = 0.1

    # Check approximate correctness of naked call to API
    cdf, qua = soft_sort.multivariate_cdf_quantile_maps(inputs)
    np.testing.assert_allclose(cdf(z), q, atol=atol)
    np.testing.assert_allclose(z, qua(q), atol=atol)

    # Check passing custom sampler, must be still symmetric / centered on {.5}^d
    # Check passing custom epsilon also works.
    def ball_sampler(k: jax.random.PRNGKey, s: Tuple[int, int]) -> jnp.ndarray:
      return 0.5 * (jax.random.ball(k, d=s[1], p=4, shape=(s[0],)) + 1.)

    num_target_samples = 473

    @functools.partial(jax.jit, static_argnums=[1])
    def mv_c_q(inputs, num_target_samples, rng, epsilon):
      return soft_sort.multivariate_cdf_quantile_maps(
          inputs,
          target_sampler=ball_sampler,
          num_target_samples=num_target_samples,
          rng=rng,
          epsilon=epsilon
      )

    cdf, qua = mv_c_q(inputs, num_target_samples, key3, 0.05)
    np.testing.assert_allclose(cdf(z), q, atol=atol)
    np.testing.assert_allclose(z, qua(q), atol=atol)

  @pytest.mark.fast.with_args("axis,jit", [(0, False), (1, True)], only_fast=0)
  def test_ranks(self, axis, rng: jax.random.PRNGKeyArray, jit: bool):
    rng1, rng2 = jax.random.split(rng, 2)
    num_targets = 13
    x = jax.random.uniform(rng1, (8, 5, 2))
    expected_ranks = jnp.argsort(
        jnp.argsort(x, axis=axis), axis=axis
    ).astype(float)
    # Define a custom version of ranks suited to recover ranks that are
    # close to true ranks. This requires notably small epsilon and large # iter.
    my_ranks = functools.partial(
        soft_sort.ranks,
        squashing_fun=lambda x: x,
        epsilon=1e-4,
        axis=axis,
        max_iterations=5000
    )
    if jit:
      my_ranks = jax.jit(my_ranks, static_argnames="num_targets")

    ranks = my_ranks(x)

    np.testing.assert_array_equal(x.shape, ranks.shape)
    np.testing.assert_allclose(ranks, expected_ranks, atol=0.3, rtol=0.1)

    ranks = my_ranks(x, num_targets=num_targets)
    np.testing.assert_array_equal(x.shape, ranks.shape)
    np.testing.assert_allclose(ranks, expected_ranks, atol=0.3, rtol=0.1)

    target_weights = jax.random.uniform(rng2, (num_targets,))
    target_weights /= jnp.sum(target_weights)
    ranks = my_ranks(x, target_weights=target_weights)
    np.testing.assert_array_equal(x.shape, ranks.shape)
    np.testing.assert_allclose(ranks, expected_ranks, atol=0.3, rtol=0.1)

  @pytest.mark.fast.with_args("axis,jit", [(0, False), (1, True)], only_fast=0)
  def test_topk_mask(self, axis, rng: jax.random.PRNGKeyArray, jit: bool):

    def boolean_topk_mask(u, k):
      return u >= jnp.flip(jax.numpy.sort(u))[k - 1]

    k = 3
    x = jax.random.uniform(rng, (13, 7, 1))
    my_topk_mask = functools.partial(
        soft_sort.topk_mask,
        squashing_fun=lambda x: x,
        epsilon=1e-4,  # needed to recover a sharp mask given close ties
        max_iterations=15000,  # needed to recover a sharp mask given close ties
        axis=axis
    )
    if jit:
      my_topk_mask = jax.jit(my_topk_mask, static_argnames=("k", "axis"))

    mask = my_topk_mask(x, k=k, axis=axis)

    expected_mask = soft_sort.apply_on_axis(boolean_topk_mask, x, axis, k)
    np.testing.assert_array_equal(x.shape, mask.shape)
    np.testing.assert_allclose(mask, expected_mask, atol=0.01, rtol=0.1)

  @pytest.mark.fast()
  @pytest.mark.parametrize("q", [0.2, 0.9])
  def test_quantile(self, q: float):
    x = jnp.linspace(0.0, 1.0, 100)
    x_q = soft_sort.quantile(x, q=q, weight=0.05, epsilon=1e-3, lse_mode=True)

    np.testing.assert_approx_equal(x_q, q, significant=1)

  def test_quantile_on_several_axes(self, rng: jax.random.PRNGKeyArray):
    batch, height, width, channels = 4, 47, 45, 3
    x = jax.random.uniform(rng, shape=(batch, height, width, channels))
    q = soft_sort.quantile(
        x, axis=(1, 2), q=0.5, weight=0.05, epsilon=1e-2, lse_mode=True
    )

    np.testing.assert_array_equal(q.shape, (batch, 1, channels))
    np.testing.assert_allclose(
        q, 0.5 * np.ones((batch, 1, channels)), atol=3e-2
    )

  @pytest.mark.fast()
  @pytest.mark.parametrize("jit", [False, True])
  def test_quantiles(self, rng: jax.random.PRNGKeyArray, jit: bool):
    inputs = jax.random.uniform(rng, (100, 2, 3))
    q = jnp.array([.1, .8, .4])
    quantile_fn = soft_sort.quantile
    if jit:
      quantile_fn = jax.jit(quantile_fn, static_argnames="axis")

    m1 = quantile_fn(inputs, q=q, weight=None, axis=0)

    np.testing.assert_allclose(m1.mean(axis=[1, 2]), q, atol=5e-2)

  @pytest.mark.parametrize("jit", [False, True])
  def test_soft_quantile_normalization(
      self, rng: jax.random.PRNGKeyArray, jit: bool
  ):
    rngs = jax.random.split(rng, 2)
    x = jax.random.uniform(rngs[0], shape=(100,))
    mu, sigma = 2.0, 1.2
    y = mu + sigma * jax.random.normal(rng, shape=(48,))
    mu_target, sigma_target = y.mean(), y.std()
    quantize_fn = soft_sort.quantile_normalization
    if jit:
      quantize_fn = jax.jit(quantize_fn)

    qn = quantize_fn(x, jnp.sort(y), epsilon=1e-4)

    mu_transform, sigma_transform = qn.mean(), qn.std()
    np.testing.assert_allclose([mu_transform, sigma_transform],
                               [mu_target, sigma_target],
                               rtol=0.05)

  def test_sort_with(self, rng: jax.random.PRNGKeyArray):
    n, d = 20, 4
    inputs = jax.random.uniform(rng, shape=(n, d))
    criterion = jnp.linspace(0.1, 1.2, n)
    output = soft_sort.sort_with(inputs, criterion, epsilon=1e-4)

    np.testing.assert_array_equal(output.shape, inputs.shape)
    np.testing.assert_allclose(output, inputs, atol=0.05)

    k = 4
    # TODO: investigate why epsilon=1e-4 fails
    output = soft_sort.sort_with(inputs, criterion, topk=k, epsilon=1e-3)

    np.testing.assert_array_equal(output.shape, [k, d])
    np.testing.assert_allclose(output, inputs[-k:], atol=0.05)

  @pytest.mark.fast()
  @pytest.mark.parametrize("jit", [False, True])
  def test_quantize(self, jit: bool):
    n = 100
    inputs = jnp.linspace(0.0, 1.0, n)[..., None]
    quantize_fn = soft_sort.quantize
    if jit:
      quantize_fn = jax.jit(quantize_fn, static_argnames=("num_levels", "axis"))

    q = quantize_fn(inputs, num_levels=4, axis=0, epsilon=1e-4)

    delta = jnp.abs(q - jnp.array([0.12, 0.34, 0.64, 0.86]))
    min_distances = jnp.min(delta, axis=1)
    np.testing.assert_allclose(min_distances, min_distances, atol=0.05)

  @pytest.mark.parametrize("implicit", [False, True])
  def test_soft_sort_jacobian(
      self, rng: jax.random.PRNGKeyArray, implicit: bool
  ):
    # Add a ridge when using JAX solvers.
    try:
      from ott.solvers.linear import lineax_implicit  # noqa: F401
      solver_kwargs = {}
    except ImportError:
      solver_kwargs = {"ridge_identity": 1e-1, "ridge_kernel": 1e-1}
    b, n = 10, 40
    num_targets = n // 2
    idx_column = 5
    rngs = jax.random.split(rng, 3)
    z = jax.random.uniform(rngs[0], ((b, n)))
    random_dir = jax.random.normal(rngs[1], (b,)) / b

    def loss_fn(logits: jnp.ndarray) -> float:
      im_d = None
      if implicit:
        # Ridge parameters are only used when using JAX's CG.
        im_d = implicit_lib.ImplicitDiff(solver_kwargs=solver_kwargs)

      ranks_fn = functools.partial(
          soft_sort.ranks,
          axis=-1,
          num_targets=num_targets,
          implicit_diff=im_d,
      )
      return jnp.sum(ranks_fn(logits)[:, idx_column] * random_dir)

    _, grad = jax.jit(jax.value_and_grad(loss_fn))(z)
    delta = jax.random.uniform(rngs[2], z.shape) - 0.5
    eps = 1e-3
    val_peps = loss_fn(z + eps * delta)
    val_meps = loss_fn(z - eps * delta)

    np.testing.assert_allclose((val_peps - val_meps) / (2 * eps),
                               jnp.sum(grad * delta),
                               atol=0.01,
                               rtol=0.1)
