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
from ott.solvers.nn import models


@pytest.mark.fast()
class TestICNN:

  def test_icnn_convexity(self, rng: jax.random.PRNGKeyArray):
    """Tests convexity of ICNN."""
    n_samples, n_features = 10, 2
    dim_hidden = (64, 64)

    # define icnn model
    model = models.ICNN(n_features, dim_hidden=dim_hidden)

    # initialize model
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    params = model.init(rng1, jnp.ones(n_features))["params"]

    # check convexity
    x = jax.random.normal(rng1, (n_samples, n_features)) * 0.1
    y = jax.random.normal(rng2, (n_samples, n_features))

    out_x = model.apply({"params": params}, x)
    out_y = model.apply({"params": params}, y)

    out = []
    for t in jnp.linspace(0, 1):
      out_xy = model.apply({"params": params}, t * x + (1 - t) * y)
      out.append((t * out_x + (1 - t) * out_y) - out_xy)

    np.testing.assert_array_equal(jnp.asarray(out) >= 0, True)

  def test_icnn_hessian(self, rng: jax.random.PRNGKeyArray):
    """Tests if Hessian of ICNN is positive-semidefinite."""

    # define icnn model
    n_features = 2
    dim_hidden = (64, 64)
    model = models.ICNN(n_features, dim_hidden=dim_hidden)

    # initialize model
    rng1, rng2 = jax.random.split(rng)
    params = model.init(rng1, jnp.ones(n_features))["params"]

    # check if Hessian is positive-semidefinite via eigenvalues
    data = jax.random.normal(rng2, (n_features,))

    # compute Hessian
    hessian = jax.hessian(model.apply, argnums=1)({"params": params}, data)

    # compute eigenvalues
    w = jnp.linalg.eigvalsh((hessian + hessian.T) / 2.0)

    np.testing.assert_array_equal(w >= 0, True)
