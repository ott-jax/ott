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
import numpy as np
import pytest
from ott.geometry import costs
from ott.solvers.nn import losses


@pytest.mark.fast()
class TestMongeGap:

  def test_monge_gap_non_negativity(self, rng: jax.random.PRNGKey):
    """Tests non-negativity of the Monge gap."""

    # generate data
    n_samples, n_features = 10, 2
    rng1, rng2 = jax.random.split(rng, 2)
    source = jax.random.normal(rng1, (n_samples, n_features))
    target = jax.random.normal(rng2, (n_samples, n_features)) * .1 + 3.

    # compute the Monge gap
    monge_gap_value = losses.monge_gap(source=source, target=target)
    np.testing.assert_array_equal(monge_gap_value >= 0, True)

  def test_monge_gap_different_cost(self, rng: jax.random.PRNGKey):
    """Tests that Monge gaps intantiated for different costs
    provide different values.
    """

    # generate data
    n_samples, n_features = 10, 2
    rng1, rng2 = jax.random.split(rng, 2)
    source = jax.random.normal(rng1, (n_samples, n_features))
    target = jax.random.normal(rng2, (n_samples, n_features)) * .1 + 3.

    # compute the Monge gaps for different costs
    monge_gap_value_sq_eucl = losses.monge_gap(
        source=source, target=target, cost_fn=costs.SqEuclidean()
    )
    monge_gap_value_eucl = losses.monge_gap(
        source=source, target=target, cost_fn=costs.Euclidean()
    )
    np.testing.assert_array_equal(
        monge_gap_value_sq_eucl == monge_gap_value_eucl, False
    )

  def test_monge_gap_jit(self, rng: jax.random.PRNGKey):
    """Tests if the Monge gap can be jitted
    w.r.t. the data points.
    """

    # generate data
    n_samples, n_features = 10, 2
    rng1, rng2 = jax.random.split(rng, 2)
    source = jax.random.normal(rng1, (n_samples, n_features))
    target = jax.random.normal(rng2, (n_samples, n_features)) * .1 + 3.

    # define jitted monge gap
    jit_monge_gap = jax.jit(
        lambda source, target: losses.monge_gap(source, target)
    )

    # compute the Monge gaps for different costs
    monge_gap_value = losses.monge_gap(
        source=source,
        target=target,
    )
    jit_monge_gap_value = jit_monge_gap(
        source=source,
        target=target,
    )
    np.testing.assert_allclose(monge_gap_value, jit_monge_gap_value, rtol=1e-3)
