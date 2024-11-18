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
from typing import Optional

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott import datasets
from ott.geometry import costs, regularizers
from ott.neural.methods import monge_gap
from ott.neural.networks import potentials
from ott.tools import sinkhorn_divergence


@pytest.mark.fast()
class TestMongeGap:

  @pytest.mark.parametrize("n_samples", [5, 25])
  @pytest.mark.parametrize("n_features", [10, 50, 100])
  def test_monge_gap_non_negativity(
      self, rng: jax.Array, n_samples: int, n_features: int
  ):

    # generate data
    rng1, rng2 = jax.random.split(rng, 2)
    reference_points = jax.random.normal(rng1, (n_samples, n_features))

    model = potentials.PotentialMLP(dim_hidden=[8, 8], is_potential=False)
    params = model.init(rng2, x=reference_points[0])
    target = model.apply(params, reference_points)

    # compute the Monge gap based on samples
    monge_gap_from_samples_value = monge_gap.monge_gap_from_samples(
        source=reference_points, target=target
    )
    np.testing.assert_array_equal(monge_gap_from_samples_value >= 0, True)

    # Compute the Monge gap using model directly
    monge_gap_value = monge_gap.monge_gap(
        map_fn=lambda x: model.apply(params, x),
        reference_points=reference_points
    )
    np.testing.assert_array_equal(monge_gap_value >= 0, True)

    np.testing.assert_array_equal(monge_gap_value, monge_gap_from_samples_value)

  def test_monge_gap_jit(self, rng: jax.Array):
    n_samples, n_features = 31, 17
    # generate data
    rng1, rng2 = jax.random.split(rng, 2)
    source = jax.random.normal(rng1, (n_samples, n_features))
    target = jax.random.normal(rng2, (n_samples, n_features))
    # define jitted monge gap
    jit_monge_gap = jax.jit(monge_gap.monge_gap_from_samples)

    # compute the Monge gaps for different costs
    monge_gap_value = monge_gap.monge_gap_from_samples(
        source=source, target=target
    )
    jit_monge_gap_value = jit_monge_gap(source, target)
    np.testing.assert_allclose(monge_gap_value, jit_monge_gap_value, rtol=1e-3)

  @pytest.mark.parametrize(
      ("cost_fn", "n_samples", "n_features"),
      [
          (costs.SqEuclidean(), 13, 5),
          (costs.PNormP(p=1), 20, 3),
          (costs.RegTICost(regularizers.L1(), lam=2.0), 100, 30),
          (costs.RegTICost(regularizers.STVS(gamma=3.0), lam=1.0), 7, 10),
      ],
      ids=[
          "sqeucl",
          "pnorm-1",
          "l1-lam2",
          "stvs-lam2",
      ],
  )
  def test_monge_gap_from_samples_different_cost(
      self, rng: jax.Array, cost_fn: costs.CostFn, n_samples: int,
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
    target = jax.random.normal(rng2, (n_samples, n_features)) * 0.1 + 3.0

    # compute the Monge gaps for the euclidean cost
    monge_gap_from_samples_value_eucl = monge_gap.monge_gap_from_samples(
        source=source, target=target, cost_fn=costs.Euclidean()
    )
    monge_gap_from_samples_value_cost_fn = monge_gap.monge_gap_from_samples(
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


@pytest.mark.fast()
class TestMongeGapEstimator:

  def test_map_estimator_convergence(self):
    """Tests convergence of a simple
    map estimator with Sinkhorn divergence fitting loss
    and Monge (coupling) gap regularizer.
    """

    # define the fitting loss and the regularizer
    def fitting_loss(
        samples: jnp.ndarray,
        mapped_samples: jnp.ndarray,
    ) -> Optional[float]:
      r"""Sinkhorn divergence fitting loss."""
      div, _ = sinkhorn_divergence.sinkdiv(
          x=samples,
          y=mapped_samples,
      )
      return div, None

    def regularizer(x, y):
      gap, out = monge_gap.monge_gap_from_samples(x, y, return_output=True)
      return gap, out.n_iters

    # define the model
    model = potentials.PotentialMLP(dim_hidden=[16, 8], is_potential=False)

    # generate data
    train_dataset, valid_dataset, dim_data = (
        datasets.create_gaussian_mixture_samplers(
            name_source="simple",
            name_target="circle",
            train_batch_size=30,
            valid_batch_size=30,
        )
    )

    # fit the map
    solver = monge_gap.MongeGapEstimator(
        dim_data=dim_data,
        fitting_loss=fitting_loss,
        regularizer=regularizer,
        model=model,
        regularizer_strength=1.0,
        num_train_iters=15,
        logging=True,
        valid_freq=5,
    )
    neural_state, logs = solver.train_map_estimator(
        *train_dataset, *valid_dataset
    )

    # check if the loss has decreased during training
    assert logs["train"]["total_loss"][0] > logs["train"]["total_loss"][-1]

    # check dimensionality of the mapped source
    source = next(train_dataset.source_iter)
    mapped_source = neural_state.apply_fn({"params": neural_state.params},
                                          source)
    assert mapped_source.shape[1] == dim_data
