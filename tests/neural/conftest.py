import numpy as np
import pytest

from ott.neural.data.dataloaders import ConditionalDataLoader, OTDataLoader


@pytest.fixture(scope="module")
def data_loader_gaussian():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  return OTDataLoader(source, target, 16)


@pytest.fixture(scope="module")
def data_loader_gaussian_conditional():
  """Returns a data loader for Gaussian mixtures with conditions."""
  rng = np.random.default_rng(seed=0)
  source_0 = rng.normal(size=(100, 2))
  target_0 = rng.normal(size=(100, 2)) + 2.0

  source_1 = rng.normal(size=(100, 2))
  target_1 = rng.normal(size=(100, 2)) - 2.0
  dl0 = OTDataLoader(
      source_0, target_0, 16, source_conditions=np.zeros_like(source_0) * 0.0
  )
  dl1 = OTDataLoader(
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
  return OTDataLoader(source, target, 16, source_conditions, target_conditions)


@pytest.fixture(scope="module")
def genot_data_loader_linear():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  return OTDataLoader(16, source_lin=source, target_lin=target)


@pytest.fixture(scope="module")
def genot_data_loader_linear_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 2)) + 1.0
  source_conditions = rng.normal(size=(100, 4))
  return OTDataLoader(
      16,
      source_lin=source,
      target_lin=target,
      source_conditions=source_conditions,
  )


@pytest.fixture(scope="module")
def genot_data_loader_quad():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 1)) + 1.0
  return OTDataLoader(16, source_quad=source, target_quad=target)


@pytest.fixture(scope="module")
def genot_data_loader_quad_conditional():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source = rng.normal(size=(100, 2))
  target = rng.normal(size=(100, 1)) + 1.0
  source_conditions = rng.normal(size=(100, 7))
  return OTDataLoader(
      16,
      source_quad=source,
      target_quad=target,
      source_conditions=source_conditions,
  )


@pytest.fixture(scope="module")
def genot_data_loader_fused():
  """Returns a data loader for a simple Gaussian mixture."""
  rng = np.random.default_rng(seed=0)
  source_q = rng.normal(size=(100, 2))
  target_q = rng.normal(size=(100, 1)) + 1.0
  source_lin = rng.normal(size=(100, 2))
  target_lin = rng.normal(size=(100, 2)) + 1.0
  return OTDataLoader(
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
  source_conditions = rng.normal(size=(100, 7))
  return OTDataLoader(
      16,
      source_lin=source_lin,
      source_quad=source_q,
      target_lin=target_lin,
      target_quad=target_q,
      source_conditions=source_conditions,
  )
