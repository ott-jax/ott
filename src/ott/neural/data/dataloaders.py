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
from typing import Dict, Iterator, Mapping, Optional

import numpy as np

__all__ = ["OTDataLoader", "ConditionalDataLoader"]


class OTDataLoader:
  """Data loader for OT problems.

  Args:
    batch_size: Number of samples per batch.
    source_lin: Linear part of the source measure.
    source_quad: Quadratic part of the source measure.
    target_lin: Linear part of the target measure.
    target_quad: Quadratic part of the target measure.
    source_conditions: Conditions of the source measure.
    target_conditions: Conditions of the target measure.
    seed: Random seed.
  """

  def __init__(
      self,
      batch_size: int = 64,
      source_lin: Optional[np.ndarray] = None,
      source_quad: Optional[np.ndarray] = None,
      target_lin: Optional[np.ndarray] = None,
      target_quad: Optional[np.ndarray] = None,
      source_conditions: Optional[np.ndarray] = None,
      target_conditions: Optional[np.ndarray] = None,
      seed: int = 0,
  ):
    super().__init__()
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
    self.batch_size = batch_size
    self.rng = np.random.default_rng(seed=seed)

  def __next__(self) -> Mapping[str, np.ndarray]:
    inds_source = self.rng.choice(self.n_source, size=[self.batch_size])
    inds_target = self.rng.choice(self.n_target, size=[self.batch_size])
    return {
        "source_lin":
            self.source_lin[inds_source, :]
            if self.source_lin is not None else None,
        "source_quad":
            self.source_quad[inds_source, :]
            if self.source_quad is not None else None,
        "target_lin":
            self.target_lin[inds_target, :]
            if self.target_lin is not None else None,
        "target_quad":
            self.target_quad[inds_target, :]
            if self.target_quad is not None else None,
        "source_conditions":
            self.source_conditions[inds_source, :]
            if self.source_conditions is not None else None,
        "target_conditions":
            self.target_conditions[inds_target, :]
            if self.target_conditions is not None else None,
    }


class ConditionalDataLoader:
  """Data loader for OT problems with conditions.

  This data loader wraps several data loaders and samples from them according
  to their conditions.

  Args:
    dataloaders: Dictionary of data loaders with keys corresponding to
      conditions.
    p: Probability of sampling from each data loader.
    seed: Random seed.
  """

  def __init__(
      self, dataloaders: Dict[str, Iterator], p: np.ndarray, seed: int = 0
  ):
    super().__init__()
    self.dataloaders = dataloaders
    self.conditions = list(dataloaders.keys())
    self.p = p
    self.rng = np.random.default_rng(seed=seed)

  def __next__(self, cond: str = None) -> Mapping[str, np.ndarray]:
    if cond is not None:
      if cond not in self.conditions:
        raise ValueError(f"Condition {cond} not in {self.conditions}")
      return next(self.dataloaders[cond])
    idx = self.rng.choice(len(self.conditions), p=self.p)
    return next(self.dataloaders[self.conditions[idx]])
