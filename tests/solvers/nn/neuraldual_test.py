#
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
"""Tests for implementation of ICNN-based Kantorovich dual by Makkuva+(2020)."""
from typing import Sequence, Tuple

import pytest

import jax
import numpy as np

from ott.problems.nn import dataset
from ott.solvers.nn import models, neuraldual

ModelPair_t = Tuple[models.ModelBase, models.ModelBase]
DatasetPair_t = Tuple[dataset.Dataset, dataset.Dataset]


@pytest.fixture(params=[("simple", "circle")])
def datasets(request: Tuple[str, str]) -> DatasetPair_t:
  train_dataset, valid_dataset, _ = dataset.create_gaussian_mixture_samplers(
      request.param[0], request.param[1]
  )
  return (train_dataset, valid_dataset)


@pytest.fixture(params=["icnns", "mlps", "mlps-grad"])
def neural_models(request: str) -> ModelPair_t:
  if request.param == 'icnns':
    return (
        models.ICNN(dim_data=2, dim_hidden=[128]),
        models.ICNN(dim_data=2, dim_hidden=[128])
    )
  elif request.param == 'mlps':
    return (models.MLP(dim_hidden=[128]), models.MLP(dim_hidden=[128]))
  elif request.param == 'mlps-grad':
    return (
        models.MLP(dim_hidden=[128]),
        models.MLP(is_potential=False, dim_hidden=[128])
    )
  else:
    raise ValueError(f'Invalid request: {request.param}')


class TestNeuralDual:

  @pytest.mark.fast.with_args("back_and_forth", [True, False])
  def test_neural_dual_convergence(
      self, datasets: DatasetPair_t, neural_models: ModelPair_t,
      back_and_forth: bool
  ):
    """Tests convergence of learning the Kantorovich dual using ICNNs."""

    def increasing(losses: Sequence[float]) -> bool:
      return all(x <= y for x, y in zip(losses, losses[1:]))

    def decreasing(losses: Sequence[float]) -> bool:
      return all(x >= y for x, y in zip(losses, losses[1:]))

    num_train_iters, log_freq = 100, 100
    neural_f, neural_g = neural_models

    # initialize neural dual
    neural_dual_solver = neuraldual.W2NeuralDual(
        dim_data=2,
        neural_f=neural_f,
        neural_g=neural_g,
        num_train_iters=num_train_iters,
        logging=True,
        log_freq=log_freq,
        back_and_forth=back_and_forth,
    )
    train_dataset, valid_dataset = datasets
    neural_dual, logs = neural_dual_solver(*train_dataset, *valid_dataset)

    # check if training loss of f is increasing and g is decreasing
    assert increasing(logs['train_logs']['loss_f'])
    assert decreasing(logs['train_logs']['loss_g'])

  def test_neural_dual_jit(self, datasets: DatasetPair_t):
    num_train_iters = 10
    # initialize neural dual
    neural_dual_solver = neuraldual.W2NeuralDual(
        dim_data=2, num_train_iters=num_train_iters
    )
    train_dataset, valid_dataset = datasets
    neural_dual = neural_dual_solver(*train_dataset, *valid_dataset)

    data_source = next(train_dataset.source_iter)
    pred_target = neural_dual.transport(data_source)

    compute_transport = jax.jit(
        lambda data_source: neural_dual.transport(data_source)
    )
    pred_target_jit = compute_transport(data_source)

    # ensure epsilon and optimal f's are a scale^2 apart (^2 comes from ^2 cost)
    np.testing.assert_allclose(
        pred_target, pred_target_jit, rtol=1e-3, atol=1e-3
    )
