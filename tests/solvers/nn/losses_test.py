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

_ = pytest.importorskip("flax")

from ott.geometry import costs
from ott.solvers.nn import losses, models


@pytest.mark.fast()
class TestMongeGap:

  @pytest.mark.parametrize("n_samples", [5, 25])
  @pytest.mark.parametrize("n_features", [10, 50, 100])
  def test_monge_gap_non_negativity(
      self, rng: jax.random.PRNGKey, n_samples: int, n_features: int
  ):

    # generate data
    rng1, rng2 = jax.random.split(rng, 2)
    reference_points = jax.random.normal(rng1, (n_samples, n_features))

    model = models.MLP(dim_hidden=[8, 8], is_potential=False)
    params = model.init(rng2, x=reference_points[0])
    target = model.apply(params, reference_points)

    # compute the Monge gap based on samples
    monge_gap_from_samples_value = losses.monge_gap_from_samples(
        source=reference_points, target=target
    )
    np.testing.assert_array_equal(monge_gap_from_samples_value >= 0, True)

    # Compute the Monge gap using model directly
    monge_gap_value = losses.monge_gap(
        map_fn=lambda x: model.apply(params, x),
        reference_points=reference_points
    )
    np.testing.assert_array_equal(monge_gap_value >= 0, True)

    np.testing.assert_array_equal(monge_gap_value, monge_gap_from_samples_value)

  def test_monge_gap_jit(self, rng: jax.random.PRNGKey):
    n_samples, n_features = 31, 17
    # generate data
    rng1, rng2 = jax.random.split(rng, 2)
    source = jax.random.normal(rng1, (n_samples, n_features))
    target = jax.random.normal(rng2, (n_samples, n_features))
    # define jitted monge gap
    jit_monge_gap = jax.jit(losses.monge_gap_from_samples)

    # compute the Monge gaps for different costs
    monge_gap_value = losses.monge_gap_from_samples(
        source=source, target=target
    )
    jit_monge_gap_value = jit_monge_gap(source, target)
    np.testing.assert_allclose(monge_gap_value, jit_monge_gap_value, rtol=1e-3)

  @pytest.mark.parametrize(
      ("cost_fn", "n_samples", "n_features"),
      [
          (costs.SqEuclidean(), 13, 5),
          (costs.PNormP(p=1), 20, 3),
          (costs.ElasticL1(scaling_reg=2.0), 100, 30),
          (costs.ElasticSTVS(scaling_reg=2.0), 7, 10),
      ],
      ids=[
          "squared-euclidean",
          "p-norm-p1",
          "elasticnet-gam2",
          "stvs-gam2",
      ],
  )
  def test_monge_gap_from_samples_different_cost(
      self, rng: jax.random.PRNGKeyArray, cost_fn: costs.CostFn, n_samples: int,
      n_features: int
  ):
    """Test that the Monge gap for different costs.

    We use the Monge gap for the Euclidean cost as a reference,
    and we compute the Monge gap for several other costs and
    verify that we obtain a different value.
    """

    # generate data
    rng1, rng2 = jax.random.split(rng, 2)
    source = jax.random.normal(rng1, (n_samples, n_features))
    target = jax.random.normal(rng2, (n_samples, n_features)) * .1 + 3.

    # compute the Monge gaps for the euclidean cost
    monge_gap_from_samples_value_eucl = losses.monge_gap_from_samples(
        source=source, target=target, cost_fn=costs.Euclidean()
    )
    monge_gap_from_samples_value_cost_fn = losses.monge_gap_from_samples(
        source=source, target=target, cost_fn=cost_fn
    )

    with pytest.raises(AssertionError, match=r"tolerance"):
      np.testing.assert_allclose(
          monge_gap_from_samples_value_eucl,
          monge_gap_from_samples_value_cost_fn,
          rtol=1e-1,
          atol=1e-1
      )

    np.testing.assert_array_equal(
        np.isfinite(monge_gap_from_samples_value_eucl), True
    )
    np.testing.assert_array_equal(
        np.isfinite(monge_gap_from_samples_value_cost_fn), True
    )
