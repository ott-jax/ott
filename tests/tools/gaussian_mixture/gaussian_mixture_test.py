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
from ott.tools.gaussian_mixture import gaussian_mixture, linalg


@pytest.mark.fast()
class TestGaussianMixture:

  def test_get_summary_stats_from_points_and_assignment_probs(
      self, rng: jax.random.PRNGKeyArray
  ):
    n = 50
    rng, subrng0, subrng1 = jax.random.split(rng, num=3)
    points0 = jax.random.normal(key=subrng0, shape=(n, 2))
    points1 = (
        2. * jax.random.normal(key=subrng1, shape=(n, 2)) + jnp.array([6., 8.])
    )
    points = jnp.concatenate([points0, points1], axis=0)
    rng, subrng0, subrng1 = jax.random.split(rng, num=3)
    weights0 = jax.random.uniform(key=subrng0, shape=(n,))
    weights1 = jax.random.uniform(key=subrng1, shape=(n,))
    weights = jnp.concatenate([weights0, weights1], axis=0)
    aprobs0 = jnp.stack([jnp.ones((n,)), jnp.zeros((n,))], axis=-1)
    aprobs1 = jnp.stack([jnp.zeros((n,)), jnp.ones((n,))], axis=-1)
    aprobs = jnp.concatenate([aprobs0, aprobs1], axis=0)
    mean, cov, comp_wt = (
        gaussian_mixture.get_summary_stats_from_points_and_assignment_probs(
            points=points, point_weights=weights, assignment_probs=aprobs
        )
    )
    mean0, cov0 = linalg.get_mean_and_cov(points0, weights=weights0)
    mean1, cov1 = linalg.get_mean_and_cov(points1, weights=weights1)
    expected_mean = jnp.stack([mean0, mean1], axis=0)
    expected_cov = jnp.stack([cov0, cov1], axis=0)
    expected_wt = (
        jnp.array([jnp.sum(weights0), jnp.sum(weights1)]) / jnp.sum(weights)
    )

    np.testing.assert_allclose(expected_mean, mean, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(expected_cov, cov, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(expected_wt, comp_wt, atol=1e-4, rtol=1e-4)

  def test_from_random(self, rng: jax.random.PRNGKeyArray):
    gmm = gaussian_mixture.GaussianMixture.from_random(
        rng=rng, n_components=3, n_dimensions=2
    )
    np.testing.assert_array_equal([gmm.n_components, gmm.n_dimensions], (3, 2))

  def test_from_mean_cov_component_weights(self,):
    mean = jnp.array([[1., 2], [3., 4.], [5, 6.]])
    cov = jnp.array([[[1., 0.], [0., 2.]], [[3., 0.], [0., 4.]],
                     [[5., 0.], [0., 6.]]])
    comp_wts = jnp.array([0.2, 0.3, 0.5])
    gmm = gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
        mean=mean, cov=cov, component_weights=comp_wts
    )
    np.testing.assert_array_equal(mean, gmm.loc)
    for i, component in enumerate(gmm.components()):
      np.testing.assert_allclose(
          cov[i], component.covariance(), atol=1e-4, rtol=1e-4
      )
    np.testing.assert_allclose(
        comp_wts, gmm.component_weights, atol=1e-4, rtol=1e-4
    )

  def test_covariance(self, rng: jax.random.PRNGKeyArray):
    gmm = gaussian_mixture.GaussianMixture.from_random(
        rng=rng, n_components=3, n_dimensions=2
    )
    cov = gmm.covariance
    for i, component in enumerate(gmm.components()):
      np.testing.assert_allclose(
          cov[i], component.covariance(), atol=1e-4, rtol=1e-4
      )

  def test_sample(self, rng: jax.random.PRNGKeyArray):
    gmm = gaussian_mixture.GaussianMixture.from_mean_cov_component_weights(
        mean=jnp.array([[-1., 0.], [1., 0.]]),
        cov=jnp.array([[[0.01, 0.], [0., 0.01]], [[0.01, 0.], [0., 0.01]]]),
        component_weights=jnp.array([0.2, 0.8])
    )
    samples = gmm.sample(rng=rng, size=10000)
    frac_pos = jnp.mean(samples[:, 0] > 0.)

    np.testing.assert_array_equal(samples.shape, (10000, 2))
    np.testing.assert_allclose(frac_pos, 0.8, atol=0.015)
    np.testing.assert_allclose(
        jnp.mean(samples[samples[:, 0] > 0.], axis=0),
        jnp.array([1., 0.]),
        atol=1.e-2
    )
    np.testing.assert_allclose(
        jnp.mean(samples[samples[:, 0] < 0.], axis=0),
        jnp.array([-1., 0.]),
        atol=1.e-1
    )

  def test_log_prob(self, rng: jax.random.PRNGKeyArray):
    n_components = 3
    size = 100
    subrng0, subrng1 = jax.random.split(rng, num=2)
    gmm = gaussian_mixture.GaussianMixture.from_random(
        rng=subrng0,
        n_components=3,
        n_dimensions=2,
        stdev_mean=1.,
        stdev_cov=1.,
        stdev_weights=1
    )
    x = gmm.sample(rng=subrng1, size=size)
    actual = gmm.log_prob(x)

    prob = jnp.zeros(size)
    for i in range(n_components):
      prob += (
          gmm.component_weights[i] * jnp.exp(gmm.components()[i].log_prob(x))
      )
    expected = jnp.log(prob)

    np.testing.assert_allclose(expected, actual, atol=1e-4, rtol=1e-4)

  def test_log_component_posterior(self, rng: jax.random.PRNGKeyArray):
    gmm = gaussian_mixture.GaussianMixture.from_random(
        rng=rng, n_components=3, n_dimensions=2
    )
    x = jnp.zeros(shape=(1, 2))
    px_c = jnp.exp(gmm.conditional_log_prob(x))
    pc = gmm.component_weights
    posterior = (px_c * pc) / jnp.sum(px_c * pc)
    expected = jnp.log(posterior)

    np.testing.assert_allclose(
        expected, gmm.get_log_component_posterior(x), atol=1e-4, rtol=1e-4
    )

  def test_flatten_unflatten(self, rng: jax.random.PRNGKeyArray):
    gmm = gaussian_mixture.GaussianMixture.from_random(
        rng=rng, n_components=3, n_dimensions=2
    )
    children, aux_data = jax.tree_util.tree_flatten(gmm)
    gmm_new = jax.tree_util.tree_unflatten(aux_data, children)

    assert gmm == gmm_new

  def test_pytree_mapping(self, rng: jax.random.PRNGKeyArray):
    gmm = gaussian_mixture.GaussianMixture.from_random(
        rng=rng, n_components=3, n_dimensions=2
    )
    gmm_x_2 = jax.tree_map(lambda x: 2 * x, gmm)
    np.testing.assert_allclose(2. * gmm.loc, gmm_x_2.loc, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        2. * gmm.scale_params, gmm_x_2.scale_params, atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        2. * gmm.component_weight_ob.params,
        gmm_x_2.component_weight_ob.params,
        atol=1e-4,
        rtol=1e-4
    )
