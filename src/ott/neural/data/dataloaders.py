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
from typing import Any, List, Mapping, Optional

import numpy as np
from jax import tree_util

__all__ = ["OTDataSet", "ConditionalOTDataLoader"]


class OTDataSet:
  """Data set for OT problems.

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
    if lin is not None:
      if quad is not None:
        assert len(lin) == len(quad)
        self.n_samples = len(lin)
      else:
        self.n_samples = len(lin)
    else:
      self.n_samples = len(quad)
    if conditions is not None:
      assert len(conditions) == self.n_samples

    self.lin = lin
    self.quad = quad
    self.conditions = conditions
    self._tree = {}
    if lin is not None:
      self._tree["lin"] = lin
    if quad is not None:
      self._tree["quad"] = quad
    if conditions is not None:
      self._tree["conditions"] = conditions

  def __getitem__(self, idx: np.ndarray) -> Mapping[str, np.ndarray]:
    return tree_util.tree_map(lambda x: x[idx], self._tree)

  def __len__(self):
    return self.n_samples


class ConditionalOTDataLoader:
  """Data loader for OT problems with conditions.

  This data loader wraps several data loaders and samples from them.

  Args:
    dataloaders: List of data loaders.
    seed: Random seed.
  """

  def __init__(
      self,
      dataloaders: List[Any],
      seed: int = 0  # dataloader should subclass torch dataloader
  ):
    super().__init__()
    self.dataloaders = dataloaders
    self.conditions = list(dataloaders)
    self.rng = np.random.default_rng(seed=seed)

  def __next__(self) -> Mapping[str, np.ndarray]:
    idx = self.rng.choice(len(self.conditions))
    return next(iter(self.dataloaders[idx]))

  def __iter__(self) -> "ConditionalOTDataLoader":
    return self
