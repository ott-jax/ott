from typing import Dict, Iterator, Optional

import jax
import jax.numpy as jnp
import pytest


class DataLoader:

  def __init__(
      self,
      source_data: jax.Array,
      target_data: jax.Array,
      conditions: Optional[jax.Array],
      batch_size: int = 64
  ) -> None:
    super().__init__()
    self.source_data = source_data
    self.target_data = target_data
    self.conditions = conditions
    self.batch_size = batch_size
    self.key = jax.random.PRNGKey(0)

  def __next__(self) -> jax.Array:
    key, self.key = jax.random.split(self.key)
    inds_source = jax.random.choice(
        key, len(self.source_data), shape=[self.batch_size]
    )
    inds_target = jax.random.choice(
        key, len(self.target_data), shape=[self.batch_size]
    )
    return self.source_data[inds_source, :], self.target_data[
        inds_target, :], self.conditions[
            inds_source, :] if self.conditions is not None else None


class ConditionalDataLoader:

  def __init__(
      self, rng: jax.random.KeyArray, dataloaders: Dict[str, Iterator],
      p: jax.Array
  ) -> None:
    super().__init__()
    self.rng = rng
    self.dataloaders = dataloaders
    self.conditions = list(dataloaders.keys())
    self.p = p

  def __next__(self) -> jnp.ndarray:
    self.rng, rng = jax.random.split(self.rng, 2)
    idx = jax.random.choice(rng, len(self.conditions), p=self.p)
    return next(self.dataloaders[self.conditions[idx]])


@pytest.fixture(scope="module")
def data_loader_gaussian():
  """Returns a data loader for a simple Gaussian mixture."""
  source = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
  target = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2)) + 1.0
  return DataLoader(source, target, None, 16)


@pytest.fixture(scope="module")
def data_loader_gaussian_conditional():
  """Returns a data loader for Gaussian mixtures with conditions."""
  source_0 = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
  target_0 = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2)) + 2.0

  source_1 = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
  target_1 = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2)) - 2.0
  dl0 = DataLoader(source_0, target_0, jnp.zeros_like(source_0) * 0.0, 16)
  dl1 = DataLoader(source_1, target_1, jnp.ones_like(source_1) * 1.0, 16)

  return ConditionalDataLoader(
      jax.random.PRNGKey(0), {
          "0": dl0,
          "1": dl1
      }, jnp.array([0.5, 0.5])
  )


@pytest.fixture(scope="module")
def data_loader_gaussian_with_conditions():
  """Returns a data loader for a simple Gaussian mixture with conditions."""
  source = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
  conditions = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 1))
  target = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2)) + 1.0
  return DataLoader(source, target, conditions, 16)
