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
from typing import Optional, Union

import pytest

import numpy as np
from torch.utils.data import DataLoader

from ott.neural.data import datasets


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
  assert lin_dim or quad_dim, "TODO"

  lin_data = None if lin_dim is None else (
      rng.normal(size=(n, lin_dim)) + offset
  )
  quad_data = None if quad_dim is None else (
      rng.normal(size=(n, quad_dim)) + offset
  )

  if isinstance(condition, float):
    cond_dim = lin_dim if cond_dim is None else cond_dim
    condition = np.full((n, cond_dim), fill_value=condition)

  return datasets.OTData(lin=lin_data, quad=quad_data, condition=condition)


@pytest.fixture()
def lin_dl() -> DataLoader:
  """Returns a data loader for a simple Gaussian mixture."""
  n, d = 128, 2
  rng = np.random.default_rng(0)

  src = _ot_data(rng, n=n, lin_dim=d)
  tgt = _ot_data(rng, n=n, lin_dim=d, offset=1.0)
  ds = datasets.OTDataset(src, tgt)

  return DataLoader(ds, batch_size=16, shuffle=True)


@pytest.fixture()
def lin_dl_with_conds() -> DataLoader:
  n, d, cond_dim = 128, 2, 3
  rng = np.random.default_rng(13)

  src_cond = rng.normal(size=(n, cond_dim))
  tgt_cond = rng.normal(size=(n, cond_dim))
  src = _ot_data(rng, n=n, lin_dim=d, condition=src_cond)
  tgt = _ot_data(rng, n=n, lin_dim=d, condition=tgt_cond)

  ds = datasets.OTDataset(src, tgt)
  return DataLoader(ds, batch_size=16, shuffle=True)


@pytest.fixture()
def conditional_lin_dl() -> datasets.ConditionalLoader:
  d, cond_dim = 2, 4
  rng = np.random.default_rng(42)

  src0 = _ot_data(rng, condition=0.0, lin_dim=d, cond_dim=cond_dim)
  tgt0 = _ot_data(rng, offset=2.0)
  src1 = _ot_data(rng, condition=1.0, lin_dim=d, cond_dim=cond_dim)
  tgt1 = _ot_data(rng, offset=-2.0)

  src_ds = datasets.OTDataset(src0, tgt0)
  tgt_ds = datasets.OTDataset(src1, tgt1)

  src_dl = DataLoader(src_ds, batch_size=16, shuffle=True)
  tgt_dl = DataLoader(tgt_ds, batch_size=16, shuffle=True)

  return datasets.ConditionalLoader([src_dl, tgt_dl])


@pytest.fixture()
def quad_dl():
  n, d = 128, 2
  rng = np.random.default_rng(11)

  src = _ot_data(rng, n=n, quad_dim=d)
  tgt = _ot_data(rng, n=n, quad_dim=d + 2, offset=1.0)
  ds = datasets.OTDataset(src, tgt)

  return DataLoader(ds, batch_size=16, shuffle=True)


@pytest.fixture()
def quad_dl_with_conds():
  pass
