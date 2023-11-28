from typing import Dict, Iterator, Mapping, Optional

import numpy as np
import pytest


class DataLoader:

  def __init__(
      self,
      source_data: np.ndarray,
      target_data: np.ndarray,
      batch_size: int = 64,
      source_conditions: Optional[np.ndarray] = None,
      target_conditions: Optional[np.ndarray] = None,
  ) -> None:
    super().__init__()
    self.source_data = source_data
    self.target_data = target_data
    self.source_conditions = source_conditions
    self.target_conditions = target_conditions
    self.batch_size = batch_size
    self.rng = np.random.default_rng(seed=0)

  def __next__(self) -> Mapping[str, np.ndarray]:
    inds_source = self.rng.choice(len(self.source_data), size=[self.batch_size])
    inds_target = self.rng.choice(len(self.target_data), size=[self.batch_size])
    return {
        "source_lin":
            self.source_data[inds_source, :],
        "target_lin":
            self.target_data[inds_target, :],
        "source_conditions":
            self.source_conditions[inds_source, :]
            if self.source_conditions is not None else None,
        "target_conditions":
            self.target_conditions[inds_target, :]
            if self.target_conditions is not None else None,
    }


class ConditionalDataLoader:

  def __init__(self, dataloaders: Dict[str, Iterator], p: np.ndarray) -> None:
    super().__init__()
    self.dataloaders = dataloaders
    self.conditions = list(dataloaders.keys())
    self.p = p
    self.rng = np.random.default_rng(seed=0)

  def __next__(self, cond: str = None) -> Mapping[str, np.ndarray]:
    if cond is not None:
      if cond not in self.conditions:
        raise ValueError(f"Condition {cond} not in {self.conditions}")
      return next(self.dataloaders[cond])
    idx = self.rng.choice(len(self.conditions), p=self.p)
    return next(self.dataloaders[self.conditions[idx]])


@pytest.fixture(scope="module")
def data_loader_gaussian():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  return DataLoader(source, target, 16)


@pytest.fixture(scope="module")
def data_loader_gaussian_conditional():
  """Returns a data loader for Gaussian mixtures with conditions."""
  rng = np.random.default_rng(seed=0)
  source_0 = rng.normal(size=(100, 2))
  target_0 = rng.normal(size=(100, 2)) + 2.0

  source_1 = rng.normal(size=(100, 2))
  target_1 = rng.normal(size=(100, 2)) - 2.0
  dl0 = DataLoader(
      source_0, target_0, 16, source_conditions=np.zeros_like(source_0) * 0.0
  )
  dl1 = DataLoader(
      source_1, target_1, 16, source_conditions=np.ones_like(source_1) * 1.0
  )

  return ConditionalDataLoader({"0": dl0, "1": dl1}, np.array([0.5, 0.5]))


@pytest.fixture(scope="module")
def data_loader_gaussian_with_conditions():
  """Returns a data loader for a simple Gaussian mixture with conditions."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  source_conditions = rng.normal(size=(100, 1))
  target_conditions = rng.normal(size=(100, 1)) - 1.0
  return DataLoader(source, target, 16, source_conditions, target_conditions)


class GENOTDataLoader:

  def __init__(
      self,
      batch_size: int = 64,
      source_lin: Optional[np.ndarray] = None,
      source_quad: Optional[np.ndarray] = None,
      target_lin: Optional[np.ndarray] = None,
      target_quad: Optional[np.ndarray] = None,
      source_conditions: Optional[np.ndarray] = None,
      target_conditions: Optional[np.ndarray] = None,
  ) -> None:
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
    self.rng = np.random.default_rng(seed=0)

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


@pytest.fixture(scope="module")
def genot_data_loader_linear():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  return GENOTDataLoader(16, source_lin=source, target_lin=target)


@pytest.fixture(scope="module")
def genot_data_loader_linear_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  conditions_source = rng.normal(size=(100, 4))
  conditions_target = rng.normal(size=(100, 4)) - 1.0
  return GENOTDataLoader(
      16,
      source_lin=source,
      target_lin=target,
      conditions_source=conditions_source,
      conditions_target=conditions_target
  )


@pytest.fixture(scope="module")
def genot_data_loader_quad():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 1)) + 1.0
  return GENOTDataLoader(16, source_quad=source, target_quad=target)


@pytest.fixture(scope="module")
def genot_data_loader_quad_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 1)) + 1.0
  conditions = rng.normal(size=(100, 7))
  return GENOTDataLoader(
      16,
      source_quad=source,
      target_quad=target,
      source_conditions=conditions,
      target_conditions=conditions
  )


@pytest.fixture(scope="module")
def genot_data_loader_fused():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source_q = rng.normal(size=(100, 2))
  target_q = rng.normal(size=(100, 1)) + 1.0
  source_lin = rng.normal(size=(100, 2))
  target_lin = rng.normal(size=(100, 2)) + 1.0
  return GENOTDataLoader(
      16,
      source_lin=source_lin,
      source_quad=source_q,
      target_lin=target_lin,
      target_quad=target_q
  )


@pytest.fixture(scope="module")
def genot_data_loader_fused_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source_q = rng.normal(size=(100, 2))
  target_q = rng.normal(size=(100, 1)) + 1.0
  source_lin = rng.normal(size=(100, 2))
  target_lin = rng.normal(size=(100, 2)) + 1.0
  conditions = rng.normal(size=(100, 7))
  return GENOTDataLoader(
      16,
      source_lin=source_lin,
      source_quad=source_q,
      target_lin=target_lin,
      target_quad=target_q,
      source_conditions=conditions,
      target_conditions=conditions
  )
