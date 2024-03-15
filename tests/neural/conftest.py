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
import torch
from torch.utils.data import DataLoader

from ott.neural.data import datasets


def _ot_data(
    rng: np.random.Generator,
    *,
    n: int = 100,
    dim: int = 2,
    condition: Optional[Union[float, np.ndarray]] = None,
    cond_dim: Optional[int] = None,
    offset: float = 0.0
) -> datasets.OTData:
  data = rng.normal(size=(n, dim)) + offset

  if isinstance(condition, float):
    cond_dim = dim if cond_dim is None else cond_dim
    condition = np.full((n, cond_dim), fill_value=condition)

  return datasets.OTData(lin=data, condition=condition)


@pytest.fixture()
def lin_dl() -> DataLoader:
  """Returns a data loader for a simple Gaussian mixture."""
  n, d = 100, 2
  rng = np.random.default_rng(0)
  src, tgt = _ot_data(rng, n=n, dim=d), _ot_data(rng, n=n, dim=d, offset=1.0)
  ds = datasets.OTDataset(src, tgt)
  return DataLoader(ds, batch_size=16, shuffle=True)


@pytest.fixture()
def lin_dl_with_conds() -> DataLoader:
  n, d, cond_dim = 100, 2, 3
  rng = np.random.default_rng(13)

  src_cond = rng.normal(size=(n, cond_dim))
  tgt_cond = rng.normal(size=(n, cond_dim))
  src = _ot_data(rng, n=n, dim=d, condition=src_cond)
  tgt = _ot_data(rng, n=n, dim=d, condition=tgt_cond)

  ds = datasets.OTDataset(src, tgt)
  return DataLoader(ds, batch_size=16, shuffle=True)


@pytest.fixture()
def conditional_lin_dl() -> datasets.ConditionalLoader:
  cond_dim = 4
  rng = np.random.default_rng(42)

  src0 = _ot_data(rng, condition=0.0, cond_dim=cond_dim)
  tgt0 = _ot_data(rng, offset=2.0)
  src1 = _ot_data(rng, condition=1.0, cond_dim=cond_dim)
  tgt1 = _ot_data(rng, offset=-2.0)

  src_ds = datasets.OTDataset(src0, tgt0)
  tgt_ds = datasets.OTDataset(src1, tgt1)

  src_dl = DataLoader(src_ds, batch_size=16, shuffle=True)
  tgt_dl = DataLoader(tgt_ds, batch_size=16, shuffle=True)

  return datasets.ConditionalLoader([src_dl, tgt_dl])


# TODO(michalk8): refactor the below for GENOT


@pytest.fixture(scope="module")
def genot_data_loader_linear():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  src = rng.normal(size=(100, 2))
  tgt = rng.normal(size=(100, 2)) + 1.0
  dataset = datasets.OTDataset(lin=src, tgt_lin=tgt)
  return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture(scope="module")
def genot_data_loader_linear_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  src_0 = rng.normal(size=(100, 2))
  tgt_0 = rng.normal(size=(100, 2)) + 1.0
  src_1 = rng.normal(size=(100, 2))
  tgt_1 = rng.normal(size=(100, 2)) + 1.0
  ds0 = datasets.OTDataset(
      lin=src_0, tgt_lin=tgt_0, conditions=np.zeros_like(src_0) * 0.0
  )
  ds1 = datasets.OTDataset(
      lin=src_1, tgt_lin=tgt_1, conditions=np.ones_like(src_1) * 1.0
  )
  sampler0 = torch.utils.data.RandomSampler(ds0, replacement=True)
  sampler1 = torch.utils.data.RandomSampler(ds1, replacement=True)
  dl0 = DataLoader(ds0, batch_size=16, sampler=sampler0)
  dl1 = DataLoader(ds1, batch_size=16, sampler=sampler1)

  return datasets.ConditionalLoader((dl0, dl1))


@pytest.fixture(scope="module")
def genot_data_loader_quad():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  src = rng.normal(size=(100, 2))
  tgt = rng.normal(size=(100, 1)) + 1.0
  dataset = datasets.OTDataset(quad=src, tgt_quad=tgt)
  return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture(scope="module")
def genot_data_loader_quad_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  src_0 = rng.normal(size=(100, 2))
  tgt_0 = rng.normal(size=(100, 1)) + 1.0
  src_1 = rng.normal(size=(100, 2))
  tgt_1 = rng.normal(size=(100, 1)) + 1.0
  ds0 = datasets.OTDataset(
      quad=src_0, tgt_quad=tgt_0, conditions=np.zeros_like(src_0) * 0.0
  )
  ds1 = datasets.OTDataset(
      quad=src_1, tgt_quad=tgt_1, conditions=np.ones_like(src_1) * 1.0
  )
  sampler0 = torch.utils.data.RandomSampler(ds0, replacement=True)
  sampler1 = torch.utils.data.RandomSampler(ds1, replacement=True)
  dl0 = DataLoader(ds0, batch_size=16, sampler=sampler0)
  dl1 = DataLoader(ds1, batch_size=16, sampler=sampler1)

  return datasets.ConditionalLoader((dl0, dl1))


@pytest.fixture(scope="module")
def genot_data_loader_fused():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  src_q = rng.normal(size=(100, 2))
  tgt_q = rng.normal(size=(100, 1)) + 1.0
  src_lin = rng.normal(size=(100, 2))
  tgt_lin = rng.normal(size=(100, 2)) + 1.0
  dataset = datasets.OTDataset(
      lin=src_lin, quad=src_q, tgt_lin=tgt_lin, tgt_quad=tgt_q
  )
  return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture(scope="module")
def genot_data_loader_fused_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  src_q_0 = rng.normal(size=(100, 2))
  tgt_q_0 = rng.normal(size=(100, 1)) + 1.0
  src_lin_0 = rng.normal(size=(100, 2))
  tgt_lin_0 = rng.normal(size=(100, 2)) + 1.0

  src_q_1 = 2 * rng.normal(size=(100, 2))
  tgt_q_1 = 2 * rng.normal(size=(100, 1)) + 1.0
  src_lin_1 = 2 * rng.normal(size=(100, 2))
  tgt_lin_1 = 2 * rng.normal(size=(100, 2)) + 1.0

  ds0 = datasets.OTDataset(
      lin=src_lin_0,
      tgt_lin=tgt_lin_0,
      quad=src_q_0,
      tgt_quad=tgt_q_0,
      conditions=np.zeros_like(src_lin_0) * 0.0
  )
  ds1 = datasets.OTDataset(
      lin=src_lin_1,
      tgt_lin=tgt_lin_1,
      quad=src_q_1,
      tgt_quad=tgt_q_1,
      conditions=np.ones_like(src_lin_1) * 1.0
  )
  sampler0 = torch.utils.data.RandomSampler(ds0, replacement=True)
  sampler1 = torch.utils.data.RandomSampler(ds1, replacement=True)
  dl0 = DataLoader(ds0, batch_size=16, sampler=sampler0)
  dl1 = DataLoader(ds1, batch_size=16, sampler=sampler1)
  return datasets.ConditionalLoader((dl0, dl1))
