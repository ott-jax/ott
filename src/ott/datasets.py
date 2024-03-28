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
import dataclasses
from typing import Iterator, Literal, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["create_gaussian_mixture_samplers", "Dataset", "GaussianMixture"]

from ott import utils

Name_t = Literal["simple", "circle", "square_five", "square_four"]


class Dataset(NamedTuple):
  r"""Samplers from source and target measures.

  Args:
    source_iter: loader for the source measure
    target_iter: loader for the target measure
  """
  source_iter: Iterator[jnp.ndarray]
  target_iter: Iterator[jnp.ndarray]


@dataclasses.dataclass
class GaussianMixture:
  """A mixture of Gaussians.

  Args:
    name: the name specifying the centers of the mixture components:

      - ``simple`` - data clustered in one center,
      - ``circle`` - two-dimensional Gaussians arranged on a circle,
      - ``square_five`` - two-dimensional Gaussians on a square with
        one Gaussian in the center, and
      - ``square_four`` - two-dimensional Gaussians in the corners of a
        rectangle

    batch_size: batch size of the samples
    rng: initial PRNG key
    scale: scale of the Gaussian means
    std: the standard deviation of the individual Gaussian samples
  """
  name: Name_t
  batch_size: int
  rng: jax.Array
  scale: float = 5.0
  std: float = 0.5

  def __post_init__(self) -> None:
    gaussian_centers = {
        "simple":
            np.array([[0, 0]]),
        "circle":
            np.array([
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            ]),
        "square_five":
            np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]),
        "square_four":
            np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    }
    if self.name not in gaussian_centers:
      raise ValueError(
          f"{self.name} is not a valid dataset for GaussianMixture"
      )
    self.centers = gaussian_centers[self.name]

  def __iter__(self) -> Iterator[jnp.array]:
    """Random sample generator from Gaussian mixture.

    Returns:
      A generator of samples from the Gaussian mixture.
    """
    return self._create_sample_generators()

  def _create_sample_generators(self) -> Iterator[jnp.array]:
    rng = self.rng
    while True:
      rng1, rng2, rng = jax.random.split(rng, 3)
      means = jax.random.choice(rng1, self.centers, (self.batch_size,))
      normal_samples = jax.random.normal(rng2, (self.batch_size, 2))
      samples = self.scale * means + (self.std ** 2) * normal_samples
      yield samples


def create_gaussian_mixture_samplers(
    name_source: Name_t,
    name_target: Name_t,
    train_batch_size: int = 2048,
    valid_batch_size: int = 2048,
    rng: Optional[jax.Array] = None,
) -> Tuple[Dataset, Dataset, int]:
  """Gaussian samplers.

  Args:
    name_source: name of the source sampler
    name_target: name of the target sampler
    train_batch_size: the training batch size
    valid_batch_size: the validation batch size
    rng: initial PRNG key

  Returns:
    The dataset and dimension of the data.
  """
  rng = utils.default_prng_key(rng)
  rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
  train_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(name_source, batch_size=train_batch_size, rng=rng1)
      ),
      target_iter=iter(
          GaussianMixture(name_target, batch_size=train_batch_size, rng=rng2)
      )
  )
  valid_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(name_source, batch_size=valid_batch_size, rng=rng3)
      ),
      target_iter=iter(
          GaussianMixture(name_target, batch_size=valid_batch_size, rng=rng4)
      )
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data
