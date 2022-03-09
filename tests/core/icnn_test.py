# coding=utf-8
# Lint as: python3
"""Tests for ICNN network architecture."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from ott.core.icnn import ICNN


class ICNNTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters({'n_samples': 10, 'n_features': 2})
  def test_icnn_convexity(self, n_samples, n_features, dim_hidden=[64, 64]):
    """Tests convexity of ICNN."""

    # define icnn model
    icnn = ICNN(dim_hidden)

    # initialize model
    params = icnn.init(self.rng, jnp.ones(n_features))['params']

    # check convexity
    x = jax.random.normal(self.rng, (n_samples, n_features)) * 0.1
    y = jax.random.normal(self.rng, (n_samples, n_features))

    out_x = icnn.apply({'params': params}, x)
    out_y = icnn.apply({'params': params}, y)

    out = list()
    for t in jnp.linspace(0, 1):
      out_xy = icnn.apply({'params': params}, t * x + (1 - t) * y)
      out.append((t * out_x + (1 - t) * out_y) - out_xy)

    self.assertTrue((jnp.array(out) >= 0).all())

  @parameterized.parameters({'n_samples': 10})
  def test_icnn_hessian(self, n_samples, dim_hidden=[64, 64]):
    """Tests if Hessian of ICNN is positive-semidefinite."""

    # define icnn model
    icnn = ICNN(dim_hidden)

    # initialize model
    params = icnn.init(self.rng, jnp.ones(n_samples, ))['params']

    # check if Hessian is positive-semidefinite via eigenvalues
    data = jax.random.normal(self.rng, (n_samples, ))

    # compute Hessian
    hessian = jax.jacfwd(jax.jacrev(icnn.apply, argnums=1), argnums=1)
    icnn_hess = hessian({'params': params}, data)

    # compute eigenvalues
    w, _ = jnp.linalg.eig(icnn_hess)

    self.assertTrue((w >= 0).all())


if __name__ == '__main__':
  absltest.main()
