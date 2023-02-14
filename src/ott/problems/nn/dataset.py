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
"""Toy datasets for neural OT."""

import dataclasses
from typing import Iterable, Iterator, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ['create_gaussian_mixture_samplers', 'Dataset', 'GaussianMixture']

Name_t = Literal['simple', 'circle', 'square_five', 'square_four']


class Dataset(NamedTuple):
  r"""Samplers from source and target measures.

  Args:
    source_iter: loader for the source measure
    target_iter: loader for the target measure
  """
  source_iter: Iterable[jnp.ndarray]
  target_iter: Iterable[jnp.ndarray]


@dataclasses.dataclass
class GaussianMixture:
  """A mixture of Gaussians.

  Args:
    name: the name specifying the centers of the mixture components:

        - ``simple`` (data clustered in one center),
        - ``circle`` (two-dimensional Gaussians arranged on a circle),
        - ``square_five`` (two-dimensional Gaussians on a square with
          one Gaussian in the center), and
        - ``square_four`` (two-dimensional Gaussians in the corners of a rectangle)

    batch_size: batch size of the samples
    init_key: initial PRNG key
    scale: scale of the individual Gaussian samples
    variance: the variance of the individual Gaussian samples
  """
  name: Name_t
  batch_size: int
  init_key: jax.random.PRNGKey
  scale: float = 5.0
  variance: float = 0.5

  def __post_init__(self):
    gaussian_centers = {
        'simple':
            np.array([[0, 0]]),
        'circle':
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
        'square_five':
            np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]),
        'square_four':
            np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    }
    if self.name not in gaussian_centers:
      raise ValueError(
          f'{self.name} is not a valid dataset for GaussianMixture'
      )
    self.centers = gaussian_centers[self.name]

  def __iter__(self) -> Iterator[jnp.array]:
    return self.create_sample_generators()

  def create_sample_generators(self) -> Iterator[jnp.array]:
    # create generator which randomly picks center and adds noise
    key = self.init_key
    while True:
      k1, k2, key = jax.random.split(key, 3)
      means = jax.random.choice(k1, self.centers, [self.batch_size])
      normal_samples = jax.random.normal(k2, [self.batch_size, 2])
      samples = self.scale * means + self.variance ** 2 * normal_samples
      yield samples


def create_gaussian_mixture_samplers(
    name_source: Name_t,
    name_target: Name_t,
    train_batch_size: int = 2048,
    valid_batch_size: int = 2048,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> Tuple[Dataset, Dataset, int]:
  """Creates Gaussian samplers for :class:`~ott.solvers.nn.neuraldual.W2NeuralDual`.

  Args:
    name_source: name of the source sampler
    name_target: name of the target sampler
    train_batch_size: the training batch size
    valid_batch_size: the validation batch size
    key: initial PRNG key

  Returns:
    The dataset and dimension of the data.
  """
  k1, k2, k3, k4 = jax.random.split(key, 4)
  train_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(
              name_source, batch_size=train_batch_size, init_key=k1
          )
      ),
      target_iter=iter(
          GaussianMixture(
              name_target, batch_size=train_batch_size, init_key=k2
          )
      )
  )
  valid_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(
              name_source, batch_size=valid_batch_size, init_key=k3
          )
      ),
      target_iter=iter(
          GaussianMixture(
              name_target, batch_size=valid_batch_size, init_key=k4
          )
      )
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data
