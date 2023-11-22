from typing import Iterator

import pytest

from ott import datasets


class UnconditionalDataLoader:

  def __init__(self, iter: Iterator):
    self.iter = iter

  def __next__(self):
    return next(self.iter), None


@pytest.fixture(scope="module")
def data_loader_gaussian_1():
  """Returns a data loader for a simple Gaussian mixture."""
  loader = datasets.create_gaussian_mixture_samplers(
      name_source="simple",
      name_target="circle",
      train_batch_size=30,
      valid_batch_size=30,
  )
  return UnconditionalDataLoader(loader[0])


@pytest.fixture(scope="module")
def data_loader_gaussian_2():
  """Returns a data loader for a simple Gaussian mixture."""
  loader = datasets.create_gaussian_mixture_samplers(
      name_source="simple",
      name_target="circle",
      train_batch_size=30,
      valid_batch_size=30,
  )
  return UnconditionalDataLoader(loader[0] + 1)
