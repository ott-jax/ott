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
"""Neural OT datasets."""

import dataclasses
from typing import Any, Iterable, Iterator, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ['create', 'Dataloaders', 'GaussianMixture']

Name_t = Literal['gaussian_simple', 'gaussian_circle', 'gaussian_square_five',
                 'gaussian_square_four']


class Dataloaders(NamedTuple):
  r"""Train and validation dataloaders for the source and target measures.

  Args:
    trainloader_source: Training dataset, source measure
    trainloader_target: Training dataset, target measure
    validloader_source: Valid dataset, source measure
    validloader_target: Valid dataset, target measure
  """
  trainloader_source: Iterable[jnp.ndarray]
  trainloader_target: Iterable[jnp.ndarray]
  validloader_source: Iterable[jnp.ndarray]
  validloader_target: Iterable[jnp.ndarray]


GAUSSIAN_CENTERS = {
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


@dataclasses.dataclass
class GaussianMixture:
  """A mixture of Gaussians.

  Args:
    name: the name specifying the centers of the mixture components;
      `simple` (data clustered in one center),
      `circle` (two-dimensional Gaussians arranged on a circle),
      `square_five` (two-dimensional Gaussians on a square with one Gaussian
      in the center), and `square_four` (two-dimensional Gaussians in the
      corners of a rectangle)
    batch_size: batch size of the samples
    init_key: initial PRNG key
    scale: scale of the individual Gaussian samples
    variance the variance of the individual Gaussian samples
  """
  name: str
  batch_size: int
  init_key: jax.random.PRNGKey
  scale: float = 5.0
  variance: float = 0.5

  def __post_init__(self):
    if self.name not in GAUSSIAN_CENTERS:
      raise ValueError(
          f'{self.name} is not a valid dataset for GaussianMixture'
      )
    self.centers = GAUSSIAN_CENTERS[self.name]

  def __iter__(self):
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


def get_sampler(name: str, **kwargs: Any) -> Iterable[jnp.ndarray]:
  """Returns a sampler.

  Args:
    name: the name of the dataset
    kwargs: additional arguments passed into the sampler

  Returns:
    The sampler
  """
  if name.startswith('gaussian'):
    name = name[9:]
    return iter(GaussianMixture(name, **kwargs))
  else:
    raise ValueError('Only Gaussian datasets are supported')


def create(
    name_source: Name_t,
    name_target: Name_t,
    train_batch_size: int = 1024,
    valid_batch_size: int = 1000,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> Tuple[Dataloaders, int]:
  """Provides the dataloaders.

  Args:
    name_source: name of the source measure
    name_target: name of the target measure
    train_batch_size: the training batch size
    valid_batch_size: the validation batch size
    key: initial PRNG key
  """
  k1, k2, k3, k4 = jax.random.split(key, 4)
  dataloaders = Dataloaders(
      trainloader_source=get_sampler(
          name_source, batch_size=train_batch_size, init_key=k1
      ),
      trainloader_target=get_sampler(
          name_target, batch_size=valid_batch_size, init_key=k2
      ),
      validloader_source=get_sampler(
          name_source, batch_size=train_batch_size, init_key=k3
      ),
      validloader_target=get_sampler(
          name_target, batch_size=valid_batch_size, init_key=k4
      ),
  )
  dim_data = 2
  return dataloaders, dim_data
