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

# Lint as: python3
"""Tests for implementation of ICNN-based Kantorovich dual by Makkuva+(2020)."""
from typing import Iterator, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing_extensions import Literal

from ott.core.neuraldual import NeuralDualSolver


class ToyDataset:

  def __init__(
      self, name: Literal['simple', 'circle', 'square_four', 'square_five']
  ):
    self.name = name

  def __iter__(self) -> Iterator[jnp.ndarray]:
    yield from self.create_sample_generators()

  def create_sample_generators(self, scale: float = 5.0, variance: float = 0.5):
    # given name of dataset, select centers
    if self.name == "simple":
      centers = np.array([0, 0])
    elif self.name == "circle":
      centers = np.array([
          (1, 0),
          (-1, 0),
          (0, 1),
          (0, -1),
          (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
          (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
          (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
          (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
      ])
    elif self.name == "square_five":
      centers = np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
    elif self.name == "square_four":
      centers = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    else:
      raise NotImplementedError(self.name)

    # create generator which randomly picks center and adds noise
    centers = scale * centers
    while True:
      center = centers[np.random.choice(len(centers))]
      point = center + variance ** 2 * np.random.randn(2)

      yield np.expand_dims(point, 0)

  @property
  def ndim(self) -> int:
    return 2


@pytest.fixture(params=[("simple", "circle")])
def toy_dataset(request) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
  src = iter(ToyDataset(request.param[0]))
  tgt = iter(ToyDataset(request.param[1]))
  return src, tgt


class TestNeuralDual:

  @pytest.mark.fast
  def test_neural_dual_convergence(
      self, toy_dataset: Tuple[ToyDataset, ToyDataset]
  ):
    """Tests convergence of learning the Kantorovich dual using ICNNs."""

    def increasing(losses: Sequence[float]) -> bool:
      return all(x <= y for x, y in zip(losses, losses[1:]))

    def decreasing(losses: Sequence[float]) -> bool:
      return all(x >= y for x, y in zip(losses, losses[1:]))

    num_train_iters, log_freq = 100, 100
    dataloader_source, dataloader_target = toy_dataset

    # initialize neural dual
    neural_dual_solver = NeuralDualSolver(
        input_dim=2,
        num_train_iters=num_train_iters,
        logging=True,
        log_freq=log_freq
    )
    neural_dual, logs = neural_dual_solver(
        dataloader_source, dataloader_target, dataloader_source,
        dataloader_target
    )

    # check if training loss of f is increasing and g is decreasing
    assert increasing(logs['train_logs']['train_loss_f'])
    assert decreasing(logs['train_logs']['train_loss_g'])

  def test_neural_dual_jit(self, toy_dataset: Tuple[ToyDataset, ToyDataset]):
    num_train_iters = 10
    dataloader_source, dataloader_target = toy_dataset
    # initialize neural dual
    neural_dual_solver = NeuralDualSolver(
        input_dim=2, num_train_iters=num_train_iters
    )
    neural_dual = neural_dual_solver(
        dataloader_source, dataloader_target, dataloader_source,
        dataloader_target
    )

    data_source = next(dataloader_source)
    pred_target = neural_dual.transport(data_source)

    compute_transport = jax.jit(
        lambda data_source: neural_dual.transport(data_source)
    )
    pred_target_jit = compute_transport(data_source)

    # ensure epsilon and optimal f's are a scale^2 apart (^2 comes from ^2 cost)
    np.testing.assert_allclose(
        pred_target, pred_target_jit, rtol=1e-3, atol=1e-3
    )
