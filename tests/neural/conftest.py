from typing import Optional

import jax
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


@pytest.fixture(scope="module")
def data_loader_gaussian():
  """Returns a data loader for a simple Gaussian mixture."""
  source = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
  target = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
  return DataLoader(source, target, None, 16)
