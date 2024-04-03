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
from collections import defaultdict
from typing import Dict, NamedTuple, Optional, Union

import pytest

import jax.numpy as jnp
import numpy as np

from ott.neural import datasets


class SimpleDataLoader:

  def __init__(
      self,
      dataset: datasets.OTDataset,
      batch_size: int,
      seed: Optional[int] = None
  ):
    self.dataset = dataset
    self.batch_size = batch_size
    self.seed = seed

  def __iter__(self):
    self._rng = np.random.default_rng(self.seed)
    return self

  def __next__(self) -> Dict[str, jnp.ndarray]:
    data = defaultdict(list)
    for _ in range(self.batch_size):
      ix = self._rng.integers(0, len(self.dataset))
      for k, v in self.dataset[ix].items():
        data[k].append(v)

    return {k: jnp.vstack(v) for k, v in data.items()}


class OTLoader(NamedTuple):
  loader: SimpleDataLoader
  lin_dim: int = 0
  quad_src_dim: int = 0
  quad_tgt_dim: int = 0
  cond_dim: Optional[int] = None


def _ot_data(
    rng: np.random.Generator,
    *,
    n: int = 100,
    lin_dim: Optional[int] = None,
    quad_dim: Optional[int] = None,
    condition: Optional[Union[float, np.ndarray]] = None,
    cond_dim: Optional[int] = None,
    offset: float = 0.0
) -> datasets.OTData:
  assert lin_dim or quad_dim, \
    "Either linear or quadratic dimension has to be specified."

  lin_data = None if lin_dim is None else (
      rng.normal(size=(n, lin_dim)) + offset
  )
  quad_data = None if quad_dim is None else (
      rng.normal(size=(n, quad_dim)) + offset
  )

  if isinstance(condition, float):
    _dim = lin_dim if lin_dim is not None else quad_dim
    cond_dim = _dim if cond_dim is None else cond_dim
    condition = np.full((n, cond_dim), fill_value=condition)

  return datasets.OTData(lin=lin_data, quad=quad_data, condition=condition)


@pytest.fixture()
def lin_dl() -> OTLoader:
  n, d = 128, 2
  rng = np.random.default_rng(0)

  src = _ot_data(rng, n=n, lin_dim=d)
  tgt = _ot_data(rng, n=n, lin_dim=d, offset=1.0)
  ds = datasets.OTDataset(src, tgt)

  return OTLoader(
      SimpleDataLoader(ds, batch_size=13),
      lin_dim=d,
  )


@pytest.fixture()
def lin_cond_dl() -> OTLoader:
  n, d, cond_dim = 128, 2, 3
  rng = np.random.default_rng(13)

  src_cond = rng.normal(size=(n, cond_dim))
  tgt_cond = rng.normal(size=(n, cond_dim))
  src = _ot_data(rng, n=n, lin_dim=d, condition=src_cond)
  tgt = _ot_data(rng, n=n, lin_dim=d, condition=tgt_cond)

  ds = datasets.OTDataset(src, tgt)
  return OTLoader(
      SimpleDataLoader(ds, batch_size=14),
      lin_dim=d,
      cond_dim=cond_dim,
  )


@pytest.fixture()
def quad_dl() -> OTLoader:
  n, quad_src_dim, quad_tgt_dim = 128, 2, 4
  rng = np.random.default_rng(11)

  src = _ot_data(rng, n=n, quad_dim=quad_src_dim)
  tgt = _ot_data(rng, n=n, quad_dim=quad_tgt_dim, offset=1.0)
  ds = datasets.OTDataset(src, tgt)

  return OTLoader(
      SimpleDataLoader(ds, batch_size=15),
      quad_src_dim=quad_src_dim,
      quad_tgt_dim=quad_tgt_dim,
  )


@pytest.fixture()
def quad_cond_dl() -> OTLoader:
  n, quad_src_dim, quad_tgt_dim, cond_dim = 128, 2, 4, 5
  rng = np.random.default_rng(414)

  src_cond = rng.normal(size=(n, cond_dim))
  tgt_cond = rng.normal(size=(n, cond_dim))
  src = _ot_data(rng, n=n, quad_dim=quad_src_dim, condition=src_cond)
  tgt = _ot_data(rng, n=n, quad_dim=quad_tgt_dim, offset=1.0, cond_dim=tgt_cond)
  ds = datasets.OTDataset(src, tgt)

  return OTLoader(
      SimpleDataLoader(ds, batch_size=16),
      quad_src_dim=quad_src_dim,
      quad_tgt_dim=quad_tgt_dim,
      cond_dim=cond_dim,
  )


@pytest.fixture()
def fused_dl() -> OTLoader:
  n, lin_dim, quad_src_dim, quad_tgt_dim = 128, 6, 2, 4
  rng = np.random.default_rng(11)

  src = _ot_data(rng, n=n, lin_dim=lin_dim, quad_dim=quad_src_dim)
  tgt = _ot_data(rng, n=n, lin_dim=lin_dim, quad_dim=quad_tgt_dim, offset=1.0)
  ds = datasets.OTDataset(src, tgt)

  return OTLoader(
      SimpleDataLoader(ds, batch_size=17),
      lin_dim=lin_dim,
      quad_src_dim=quad_src_dim,
      quad_tgt_dim=quad_tgt_dim,
  )


@pytest.fixture()
def fused_cond_dl() -> OTLoader:
  n, lin_dim, quad_src_dim, quad_tgt_dim, cond_dim = 128, 6, 2, 4, 7
  rng = np.random.default_rng(11)

  src_cond = rng.normal(size=(n, cond_dim))
  tgt_cond = rng.normal(size=(n, cond_dim))
  src = _ot_data(
      rng, n=n, lin_dim=lin_dim, quad_dim=quad_src_dim, condition=src_cond
  )
  tgt = _ot_data(
      rng,
      n=n,
      lin_dim=lin_dim,
      quad_dim=quad_tgt_dim,
      offset=1.0,
      condition=tgt_cond
  )
  ds = datasets.OTDataset(src, tgt)

  return OTLoader(
      SimpleDataLoader(ds, batch_size=18),
      lin_dim=lin_dim,
      quad_src_dim=quad_src_dim,
      quad_tgt_dim=quad_tgt_dim,
  )
