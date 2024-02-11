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

__all__ = ["OTDataSet", "ConditionalOTDataLoader"]


class OTDataSet:
  """Data set for OT problems.

  Args:
    source_lin: Linear part of the source measure.
    source_quad: Quadratic part of the source measure.
    target_lin: Linear part of the target measure.
    target_quad: Quadratic part of the target measure.
    source_conditions: Conditions of the source measure.
    target_conditions: Conditions of the target measure.
  """

  def __init__(
      self,
      source_lin: Optional[np.ndarray] = None,
      source_quad: Optional[np.ndarray] = None,
      target_lin: Optional[np.ndarray] = None,
      target_quad: Optional[np.ndarray] = None,
      source_conditions: Optional[np.ndarray] = None,
      target_conditions: Optional[np.ndarray] = None,
  ):
    if source_lin is not None:
      if source_quad is not None:
        assert len(source_lin) == len(source_quad)
        self.n_source = len(source_lin)
      else:
        self.n_source = len(source_lin)
    else:
      self.n_source = len(source_quad)
    if source_conditions is not None:
      assert len(source_conditions) == self.n_source
    if target_lin is not None:
      if target_quad is not None:
        assert len(target_lin) == len(target_quad)
        self.n_target = len(target_lin)
      else:
        self.n_target = len(target_lin)
    else:
      self.n_target = len(target_quad)
    if target_conditions is not None:
      assert len(target_conditions) == self.n_target

    self.source_lin = source_lin
    self.target_lin = target_lin
    self.source_quad = source_quad
    self.target_quad = target_quad
    self.source_conditions = source_conditions
    self.target_conditions = target_conditions

  def __getitem__(self, idx: np.ndarray) -> Mapping[str, np.ndarray]:
    return {
        "source_lin":
            self.source_lin[idx] if self.source_lin is not None else [],
        "source_quad":
            self.source_quad[idx] if self.source_quad is not None else [],
        "target_lin":
            self.target_lin[idx] if self.target_lin is not None else [],
        "target_quad":
            self.target_quad[idx] if self.target_quad is not None else [],
        "source_conditions":
            self.source_conditions[idx]
            if self.source_conditions is not None else [],
        "target_conditions":
            self.target_conditions[idx]
            if self.target_conditions is not None else [],
    }

  def __len__(self):
    return len(self.source_lin
              ) if self.source_lin is not None else len(self.source_quad)


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
