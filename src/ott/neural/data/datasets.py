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
from typing import Dict, Iterable, Optional

import jax.tree_util as jtu
import numpy as np

__all__ = ["OTDataset", "ConditionalOTDataset"]


class OTDataset:
  """Dataset for Optimal transport problems.

  Args:
    lin: Linear part of the measure.
    quad: Quadratic part of the measure.
    conditions: Conditions of the source measure.
  """

  def __init__(
      self,
      lin: Optional[np.ndarray] = None,
      quad: Optional[np.ndarray] = None,
      conditions: Optional[np.ndarray] = None,
  ):
    self.data = {}
    if lin is not None:
      self.data["lin"] = lin
    if quad is not None:
      self.data["quad"] = quad
    if conditions is not None:
      self.data["conditions"] = conditions
    self._check_sizes()

  def _check_sizes(self) -> None:
    sizes = {k: len(v) for k, v in self.data.items()}
    if not len(set(sizes.values())) == 1:
      raise ValueError(f"Not all arrays have the same size: {sizes}.")

  def __getitem__(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
    return jtu.tree_map(lambda x: x[idx], self.data)

  def __len__(self) -> int:
    for v in self.data.values():
      return len(v)
    return 0


# TODO(michalk8): rename
class ConditionalOTDataset:
  """Dataset for OT problems with conditions.

  This data loader wraps several data loaders and samples from them.

  Args:
    datasets: Datasets to sample from.
    weights: TODO.
    seed: Random seed.
  """

  def __init__(
      self,
      # TODO(michalk8): generalize the type
      datasets: Iterable[OTDataset],
      weights: Iterable[float] = None,
      seed: Optional[int] = None,
  ):
    self.datasets = tuple(datasets)

    if weights is None:
      weights = np.ones(len(self.datasets))
    weights = np.asarray(weights)
    self.weights = weights / np.sum(weights)
    assert len(self.weights) == len(self.datasets), "TODO"

    self._rng = np.random.default_rng(seed)
    self._iterators = ()

  def __next__(self) -> Dict[str, np.ndarray]:
    idx = self._rng.choice(len(self._iterators), p=self.weights)
    return next(self._iterators[idx])

  def __iter__(self) -> "ConditionalOTDataset":
    self._iterators = tuple(iter(ds) for ds in self.datasets)
    return self
