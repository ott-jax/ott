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
"""Tests for ICNN network architecture."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
from ott.core.icnn import ICNN


class ICNNTest(jax.test_util.JaxTestCase):

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
