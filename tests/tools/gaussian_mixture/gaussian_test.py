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
from ott.tools.gaussian_mixture import gaussian, scale_tril


@pytest.mark.fast()
class TestGaussian:

  def test_from_random(self, rng: jax.random.PRNGKeyArray):
    g = gaussian.Gaussian.from_random(rng=rng, n_dimensions=3)

    np.testing.assert_array_equal(g.loc.shape, (3,))
    np.testing.assert_array_equal(g.covariance().shape, (3, 3))

  def test_from_mean_and_cov(self):
    mean = jnp.array([1., 2., 3.])
    cov = jnp.diag(jnp.array([4., 5., 6.]))
    g = gaussian.Gaussian.from_mean_and_cov(mean=mean, cov=cov)

    np.testing.assert_array_equal(mean, g.loc)
    np.testing.assert_allclose(cov, g.covariance(), atol=1e-4, rtol=1e-4)

  def test_to_z(self, rng: jax.random.PRNGKeyArray):
    g = gaussian.Gaussian(
        loc=jnp.array([1., 2.]),
        scale=scale_tril.ScaleTriL(
            params=jnp.array([0., 0.25, jnp.log(0.5)]), size=2
        )
    )
    samples = g.sample(rng=rng, size=1000)
    z = g.to_z(samples)
    sample_mean = jnp.mean(z, axis=0)
    sample_cov = jnp.cov(z, rowvar=False)

    np.testing.assert_array_equal(z.shape, (1000, 2))
    np.testing.assert_allclose(sample_mean, jnp.zeros(2), atol=0.1)
    np.testing.assert_allclose(sample_cov, jnp.eye(2), atol=0.1)

  def test_from_z(self, rng: jax.random.PRNGKeyArray):
    g = gaussian.Gaussian(
        loc=jnp.array([0., 0.]),
        scale=scale_tril.ScaleTriL(
            params=jnp.array([jnp.log(2.), 0., 0.]), size=2
        )
    )
    x = g.sample(rng=rng, size=100)
    z = g.to_z(x)
    xnew = g.from_z(z)
    np.testing.assert_allclose(x, xnew, atol=1e-4, rtol=1e-4)

  def test_log_prob(self, rng: jax.random.PRNGKeyArray):
    g = gaussian.Gaussian(
        loc=jnp.array([0., 0.]),
        scale=scale_tril.ScaleTriL(
            params=jnp.array([jnp.log(2.), 0., 0.]), size=2
        )
    )
    x = g.sample(rng=rng, size=100)
    actual = g.log_prob(x)
    expected = jnp.log(
        jax.scipy.stats.multivariate_normal.pdf(x, g.loc, g.covariance())
    )
    np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)

  def test_sample(self, rng: jax.random.PRNGKeyArray):
    mean = jnp.array([1., 2.])
    cov = jnp.diag(jnp.array([1., 4.]))
    g = gaussian.Gaussian.from_mean_and_cov(mean, cov)
    samples = g.sample(rng=rng, size=10000)
    sample_mean = jnp.mean(samples, axis=0)
    sample_cov = jnp.cov(samples, rowvar=False)

    np.testing.assert_allclose(sample_mean, mean, atol=3. * 2. / 100.)
    np.testing.assert_allclose(sample_cov, cov, atol=2e-1)

  def test_w2_dist(self, rng: jax.random.PRNGKeyArray):
    # make sure distance between a random normal and itself is 0
    rng, subrng = jax.random.split(rng)
    n = gaussian.Gaussian.from_random(rng=subrng, n_dimensions=3)
    w2 = n.w2_dist(n)
    np.testing.assert_almost_equal(w2, 0., decimal=5)

    # When covariances commute (e.g. if covariance is diagonal), have
    # distance between covariances = frobenius norm^2 of (delta cholesky), see
    # https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/  # noqa: E501
    size = 4
    rng, subrng0, subrng1 = jax.random.split(rng, num=3)
    loc0 = jax.random.normal(key=subrng0, shape=(size,))
    loc1 = jax.random.normal(key=subrng1, shape=(size,))
    rng, subrng0, subrng1 = jax.random.split(rng, num=3)
    diag0 = jnp.exp(jax.random.normal(key=subrng0, shape=(size,)))
    diag1 = jnp.exp(jax.random.normal(key=subrng1, shape=(size,)))
    g0 = gaussian.Gaussian(
        loc=loc0, scale=scale_tril.ScaleTriL.from_covariance(jnp.diag(diag0))
    )
    g1 = gaussian.Gaussian(
        loc=loc1, scale=scale_tril.ScaleTriL.from_covariance(jnp.diag(diag1))
    )
    w2 = g0.w2_dist(g1)
    delta_mean = jnp.sum((loc1 - loc0) ** 2., axis=-1)
    delta_sigma = jnp.sum((jnp.sqrt(diag0) - jnp.sqrt(diag1)) ** 2.)
    expected = delta_mean + delta_sigma
    np.testing.assert_allclose(expected, w2, rtol=1e-6, atol=1e-6)

  def test_transport(self, rng: jax.random.PRNGKeyArray):
    diag0 = jnp.array([1.])
    diag1 = jnp.array([4.])
    g0 = gaussian.Gaussian(
        loc=jnp.array([0.]),
        scale=scale_tril.ScaleTriL.from_covariance(jnp.diag(diag0))
    )
    g1 = gaussian.Gaussian(
        loc=jnp.array([1.]),
        scale=scale_tril.ScaleTriL.from_covariance(jnp.diag(diag1))
    )
    points = jax.random.normal(key=rng, shape=(10, 1))
    actual = g0.transport(dest=g1, points=points)
    expected = 2. * points + 1.
    np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)

  def test_flatten_unflatten(self, rng: jax.random.PRNGKeyArray):
    g = gaussian.Gaussian.from_random(rng, n_dimensions=3)
    children, aux_data = jax.tree_util.tree_flatten(g)
    g_new = jax.tree_util.tree_unflatten(aux_data, children)

    assert g == g_new

  def test_pytree_mapping(self, rng: jax.random.PRNGKeyArray):
    g = gaussian.Gaussian.from_random(rng, n_dimensions=3)
    g_x_2 = jax.tree_map(lambda x: 2 * x, g)

    np.testing.assert_allclose(2. * g.loc, g_x_2.loc)
    np.testing.assert_allclose(2. * g.scale.params, g_x_2.scale.params)
