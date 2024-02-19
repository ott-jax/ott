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
from typing import Tuple

import pytest

import numpy as np
import torch
from torch.utils.data import DataLoader

from ott.neural.data import datasets


@pytest.fixture(scope="module")
def data_loaders_gaussian() -> Tuple[DataLoader, DataLoader]:
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  src_dataset = datasets.OTDataset(lin=source)
  tgt_dataset = datasets.OTDataset(lin=target)
  loader_src = DataLoader(src_dataset, batch_size=16, shuffle=True)
  loader_tgt = DataLoader(tgt_dataset, batch_size=16, shuffle=True)
  return loader_src, loader_tgt


@pytest.fixture(scope="module")
def data_loader_gaussian_conditional():
  """Returns a data loader for Gaussian mixtures with conditions."""
  rng = np.random.default_rng(seed=0)
  source_0 = rng.normal(size=(100, 2))
  target_0 = rng.normal(size=(100, 2)) + 2.0

  source_1 = rng.normal(size=(100, 2))
  target_1 = rng.normal(size=(100, 2)) - 2.0
  ds0 = datasets.OTDataset(
      lin=source_0,
      target_lin=target_0,
      conditions=np.zeros_like(source_0) * 0.0
  )
  ds1 = datasets.OTDataset(
      lin=source_1,
      target_lin=target_1,
      conditions=np.ones_like(source_1) * 1.0
  )
  sampler0 = torch.utils.data.RandomSampler(ds0, replacement=True)
  sampler1 = torch.utils.data.RandomSampler(ds1, replacement=True)
  dl0 = DataLoader(ds0, batch_size=16, sampler=sampler0)
  dl1 = DataLoader(ds1, batch_size=16, sampler=sampler1)

  return datasets.ConditionalOTDataset((dl0, dl1))


@pytest.fixture(scope="module")
def data_loader_gaussian_with_conditions():
  """Returns a data loader for a simple Gaussian mixture with conditions."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  source_conditions = rng.normal(size=(100, 1))
  target_conditions = rng.normal(size=(100, 1)) - 1.0

  dataset = datasets.OTDataset(
      lin=source,
      target_lin=target,
      conditions=source_conditions,
      target_conditions=target_conditions
  )
  return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture(scope="module")
def genot_data_loader_linear():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  dataset = datasets.OTDataset(lin=source, target_lin=target)
  return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture(scope="module")
def genot_data_loader_linear_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source_0 = rng.normal(size=(100, 2))
  target_0 = rng.normal(size=(100, 2)) + 1.0
  source_1 = rng.normal(size=(100, 2))
  target_1 = rng.normal(size=(100, 2)) + 1.0
  ds0 = datasets.OTDataset(
      lin=source_0,
      target_lin=target_0,
      conditions=np.zeros_like(source_0) * 0.0
  )
  ds1 = datasets.OTDataset(
      lin=source_1,
      target_lin=target_1,
      conditions=np.ones_like(source_1) * 1.0
  )
  sampler0 = torch.utils.data.RandomSampler(ds0, replacement=True)
  sampler1 = torch.utils.data.RandomSampler(ds1, replacement=True)
  dl0 = DataLoader(ds0, batch_size=16, sampler=sampler0)
  dl1 = DataLoader(ds1, batch_size=16, sampler=sampler1)

  return datasets.ConditionalOTDataset((dl0, dl1))


@pytest.fixture(scope="module")
def genot_data_loader_quad():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 1)) + 1.0
  dataset = datasets.OTDataset(quad=source, target_quad=target)
  return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture(scope="module")
def genot_data_loader_quad_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source_0 = rng.normal(size=(100, 2))
  target_0 = rng.normal(size=(100, 1)) + 1.0
  source_1 = rng.normal(size=(100, 2))
  target_1 = rng.normal(size=(100, 1)) + 1.0
  ds0 = datasets.OTDataset(
      quad=source_0,
      target_quad=target_0,
      conditions=np.zeros_like(source_0) * 0.0
  )
  ds1 = datasets.OTDataset(
      quad=source_1,
      target_quad=target_1,
      conditions=np.ones_like(source_1) * 1.0
  )
  sampler0 = torch.utils.data.RandomSampler(ds0, replacement=True)
  sampler1 = torch.utils.data.RandomSampler(ds1, replacement=True)
  dl0 = DataLoader(ds0, batch_size=16, sampler=sampler0)
  dl1 = DataLoader(ds1, batch_size=16, sampler=sampler1)

  return datasets.ConditionalOTDataset((dl0, dl1))


@pytest.fixture(scope="module")
def genot_data_loader_fused():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source_q = rng.normal(size=(100, 2))
  target_q = rng.normal(size=(100, 1)) + 1.0
  source_lin = rng.normal(size=(100, 2))
  target_lin = rng.normal(size=(100, 2)) + 1.0
  dataset = datasets.OTDataset(
      lin=source_lin,
      quad=source_q,
      target_lin=target_lin,
      target_quad=target_q
  )
  return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture(scope="module")
def genot_data_loader_fused_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source_q_0 = rng.normal(size=(100, 2))
  target_q_0 = rng.normal(size=(100, 1)) + 1.0
  source_lin_0 = rng.normal(size=(100, 2))
  target_lin_0 = rng.normal(size=(100, 2)) + 1.0

  source_q_1 = 2 * rng.normal(size=(100, 2))
  target_q_1 = 2 * rng.normal(size=(100, 1)) + 1.0
  source_lin_1 = 2 * rng.normal(size=(100, 2))
  target_lin_1 = 2 * rng.normal(size=(100, 2)) + 1.0

  ds0 = datasets.OTDataset(
      lin=source_lin_0,
      target_lin=target_lin_0,
      quad=source_q_0,
      target_quad=target_q_0,
      conditions=np.zeros_like(source_lin_0) * 0.0
  )
  ds1 = datasets.OTDataset(
      lin=source_lin_1,
      target_lin=target_lin_1,
      quad=source_q_1,
      target_quad=target_q_1,
      conditions=np.ones_like(source_lin_1) * 1.0
  )
  sampler0 = torch.utils.data.RandomSampler(ds0, replacement=True)
  sampler1 = torch.utils.data.RandomSampler(ds1, replacement=True)
  dl0 = DataLoader(ds0, batch_size=16, sampler=sampler0)
  dl1 = DataLoader(ds1, batch_size=16, sampler=sampler1)
  return datasets.ConditionalOTDataset((dl0, dl1))
