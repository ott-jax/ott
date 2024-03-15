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
import collections
import dataclasses
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

__all__ = ["OTData", "OTDataset", "ConditionalLoader"]

Item_t = Dict[str, np.ndarray]


@dataclasses.dataclass(repr=False, frozen=True)
class OTData:
  """TODO."""
  lin: Optional[np.ndarray] = None
  quad: Optional[np.ndarray] = None
  condition: Optional[np.ndarray] = None

  def __getitem__(self, ix: int) -> Item_t:
    return {k: v[ix] for k, v in self.__dict__.items() if v is not None}

  def __len__(self) -> int:
    if self.lin is not None:
      return len(self.lin)
    if self.quad is not None:
      return len(self.quad)
    return 0


class OTDataset:
  """TODO."""
  SRC_PREFIX = "src"
  TGT_PREFIX = "tgt"

  def __init__(
      self,
      src_data: OTData,
      tgt_data: OTData,
      src_conditions: Optional[Sequence[Any]] = None,
      tgt_conditions: Optional[Sequence[Any]] = None,
      is_aligned: bool = False,
      seed: Optional[int] = None
  ):
    self.src_data = src_data
    self.tgt_data = tgt_data

    if src_conditions is None:
      src_conditions = [None] * len(src_data)
    self.src_conditions = list(src_conditions)
    if tgt_conditions is None:
      tgt_conditions = [None] * len(tgt_data)
    self.tgt_conditions = list(tgt_conditions)

    self._tgt_cond_to_ix = collections.defaultdict(list)
    for ix, cond in enumerate(tgt_conditions):
      self._tgt_cond_to_ix[cond].append(ix)

    self.is_aligned = is_aligned
    self._rng = np.random.default_rng(seed)

    self._verify_integriy()

  def _verify_integriy(self) -> None:
    assert len(self.src_data) == len(self.src_conditions)
    assert len(self.tgt_data) == len(self.tgt_conditions)

    if self.is_aligned:
      assert len(self.src_data) == len(self.tgt_data)
      assert self.src_conditions == self.tgt_conditions
    else:
      sym_diff = set(self.src_conditions
                    ).symmetric_difference(self.tgt_conditions)
      assert not sym_diff, sym_diff

  def _sample_from_target(self, src_ix: int) -> Item_t:
    src_cond = self.src_conditions[src_ix]
    tgt_ixs = self._tgt_cond_to_ix[src_cond]
    ix = self._rng.choice(tgt_ixs)
    return self.src_data[ix]

  def __getitem__(self, ix: int) -> Item_t:
    src = self.src_data[ix]
    src = {f"{self.SRC_PREFIX}_{k}": v for k, v in src.items()}

    tgt = self.src_data[ix] if self.is_aligned else self._sample_from_target(ix)
    tgt = {f"{self.TGT_PREFIX}_{k}": v for k, v in tgt.items()}

    return {**src, **tgt}

  def __len__(self) -> int:
    return len(self.src_data)


class ConditionalLoader:
  """Dataset for OT problems with conditions.

  This data loader wraps several data loaders and samples from them.

  Args:
    datasets: Datasets to sample from.
    seed: Random seed.
  """

  def __init__(
      self,
      datasets: Iterable[OTDataset],
      seed: Optional[int] = None,
  ):
    self.datasets = tuple(datasets)
    self._rng = np.random.default_rng(seed)
    self._iterators = []
    self._it = 0

  def __next__(self) -> Item_t:
    if self._it == len(self):
      raise StopIteration

    ix = self._rng.choice(len(self._iterators))
    iterator = self._iterators[ix]
    try:
      data = next(iterator)
      # TODO(michalk8): improve the logic a bit
      self._it += 1
      return data
    except StopIteration:
      self._iterators[ix] = iter(self.datasets[ix])
      if not self._iterators:
        raise

  def __iter__(self) -> "ConditionalLoader":
    self._iterators = [iter(ds) for ds in self.datasets]
    self._it = 0
    return self

  def __len__(self) -> int:
    return max((len(ds) for ds in self.datasets), default=0)