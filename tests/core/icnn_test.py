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
"""Tests for ICNN network architecture."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core.icnn import ICNN


@pytest.mark.fast
class TestICNN:

  def test_icnn_convexity(self, rng: jnp.ndarray):
    """Tests convexity of ICNN."""
    n_samples, n_features = 10, 2
    dim_hidden = (64, 64)

    # define icnn model
    icnn = ICNN(dim_hidden)

    # initialize model
    key1, key2, key3 = jax.random.split(rng, 3)
    params = icnn.init(key1, jnp.ones(n_features))['params']

    # check convexity
    x = jax.random.normal(key1, (n_samples, n_features)) * 0.1
    y = jax.random.normal(key2, (n_samples, n_features))

    out_x = icnn.apply({'params': params}, x)
    out_y = icnn.apply({'params': params}, y)

    out = list()
    for t in jnp.linspace(0, 1):
      out_xy = icnn.apply({'params': params}, t * x + (1 - t) * y)
      out.append((t * out_x + (1 - t) * out_y) - out_xy)

    np.testing.assert_array_equal(jnp.asarray(out) >= 0, True)

  def test_icnn_hessian(self, rng: jnp.ndarray):
    """Tests if Hessian of ICNN is positive-semidefinite."""

    # define icnn model
    n_samples = 2
    dim_hidden = (64, 64)
    icnn = ICNN(dim_hidden)

    # initialize model
    key1, key2 = jax.random.split(rng)
    params = icnn.init(key1, jnp.ones(n_samples))['params']

    # check if Hessian is positive-semidefinite via eigenvalues
    data = jax.random.normal(key2, (n_samples,))

    # compute Hessian
    hessian = jax.jacfwd(jax.jacrev(icnn.apply, argnums=1), argnums=1)
    icnn_hess = hessian({'params': params}, data)

    # compute eigenvalues
    w, _ = jnp.linalg.eig(icnn_hess)

    np.testing.assert_array_equal(w >= 0, True)
