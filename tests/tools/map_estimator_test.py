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

import jax.numpy as jnp
import pytest

_ = pytest.importorskip("flax")

from ott.geometry import pointcloud
from ott.problems.nn import dataset
from ott.solvers.nn import losses, models
from ott.tools import map_estimator, sinkhorn_divergence


@pytest.mark.fast()
class TestMapEstimator:

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
      div = sinkhorn_divergence.sinkhorn_divergence(
          pointcloud.PointCloud,
          x=samples,
          y=mapped_samples,
      ).divergence
      return (div, None)

    def regularizer(x, y):
      gap, out = losses.monge_gap_from_samples(x, y, return_output=True)
      return gap, out.n_iters

    # define the model
    model = models.MLP(dim_hidden=[16, 8], is_potential=False)

    # generate data
    train_dataset, valid_dataset, dim_data = (
        dataset.create_gaussian_mixture_samplers(
            name_source="simple",
            name_target="circle",
            train_batch_size=30,
            valid_batch_size=30,
        )
    )

    # fit the map
    solver = map_estimator.MapEstimator(
        dim_data=dim_data,
        fitting_loss=fitting_loss,
        regularizer=regularizer,
        model=model,
        regularizer_strength=1.,
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
