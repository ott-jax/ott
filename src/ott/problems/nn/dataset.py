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

__all__ = [
    'gaussian_mixture_samplers', 'uniform_mixture_samplers', 'Dataset',
    'GaussianMixture'
]

Arrangement_t = Literal['simple', 'circle', 'square_five', 'square_four']
Position_t = Literal['top', 'bottom']


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
    rng: initial PRNG key
    scale: scale of the individual Gaussian samples
    variance: the variance of the individual Gaussian samples
  """
  name: Arrangement_t
  batch_size: int
  rng: jax.random.PRNGKeyArray
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

  def __iter__(self) -> Iterator[jnp.ndarray]:
    return self.create_sample_generators()

  def create_sample_generators(self) -> Iterator[jnp.ndarray]:
    # create generator which randomly picks center and adds noise
    key = self.rng
    
    @jax.jit
    def sample(key : jax.random.PRNGKeyArray):
      """Jitted gaussian sample function."""
      k1, k2, key = jax.random.split(key, 3)
      means = jax.random.choice(k1, self.centers, [self.batch_size])
      normal_samples = jax.random.normal(k2, [self.batch_size, 2])
      samples = self.scale * means + self.variance ** 2 * normal_samples
      return samples, key
    
    while True:
      samples, key = sample(key)
      yield samples


def gaussian_mixture_samplers(
    name_source: Arrangement_t,
    name_target: Arrangement_t,
    train_batch_size: int = 2048,
    valid_batch_size: int = 2048,
    key: jax.random.PRNGKeyArray = jax.random.PRNGKey(0),
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
          GaussianMixture(name_source, batch_size=train_batch_size, rng=k1)
      ),
      target_iter=iter(
          GaussianMixture(name_target, batch_size=train_batch_size, rng=k2)
      )
  )
  valid_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(name_source, batch_size=valid_batch_size, rng=k3)
      ),
      target_iter=iter(
          GaussianMixture(name_target, batch_size=valid_batch_size, rng=k4)
      )
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data


@dataclasses.dataclass
class UniformMixture:
  """A mixture of uniform distributions.

  Args:
    name: the name specifying position of mixture of Gaussian:

        - ``top`` (uniform distributions on top),
        - ``bottom`` (uniform distributions at the bottom),

    batch_size: batch size of the samples
    rng: initial PRNG key
    mixture_weights: mixture weights between two uniform distributions
  """
  name: Position_t
  batch_size: int
  rng: jax.random.PRNGKeyArray
  mixture_weights: Tuple[float, float]

  def __post_init__(self):
    uniform_anchors = {
        'bottom': [[0.0, 0.0], [5.0, 0.0]],
        'top': [[0.0, 2.0], [5.0, 2.0]],
    }
    if self.name not in uniform_anchors:
      raise ValueError(f'{self.name} is not a valid dataset for UniformMixture')
    self.anchors = uniform_anchors[self.name]

  def __iter__(self) -> Iterator[jnp.ndarray]:
    return self.create_sample_generators()

  def create_sample_generators(self) -> Iterator[jnp.ndarray]:
    # create generator which randomly picks center and adds noise
    key = self.rng

    @jax.jit
    def sample(key : jax.random.PRNGKeyArray):
      """Jitted uniform sample function."""
      k1, k2, key = jax.random.split(key, 3)
      samples_1 = jax.random.uniform(
          k1,
          shape=(int(self.batch_size * self.mixture_weights[0]), 2),
          minval=jnp.array(self.anchors[0]) - 0.5,
          maxval=jnp.array(self.anchors[0]) + 0.5
      )
      samples_2 = jax.random.uniform(
          k2,
          shape=(int(self.batch_size * self.mixture_weights[1]), 2),
          minval=jnp.array(self.anchors[1]) - 0.5,
          maxval=jnp.array(self.anchors[1]) + 0.5
      )
      return jnp.vstack((samples_1, samples_2)), key
    
    while True:
      samples, key = sample(key)
      yield samples


def uniform_mixture_samplers(
    name_source: Position_t,
    name_target: Position_t,
    mixture_weights_source: Tuple[float, float],
    mixture_weights_target: Tuple[float, float],
    train_batch_size: int = 2048,
    valid_batch_size: int = 2048,
    key: jax.random.PRNGKeyArray = jax.random.PRNGKey(0),
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
          UniformMixture(
              name_source,
              batch_size=train_batch_size,
              rng=k1,
              mixture_weights=mixture_weights_source
          )
      ),
      target_iter=iter(
          UniformMixture(
              name_target,
              batch_size=train_batch_size,
              rng=k2,
              mixture_weights=mixture_weights_target
          )
      )
  )
  valid_dataset = Dataset(
      source_iter=iter(
          UniformMixture(
              name_source,
              batch_size=valid_batch_size,
              rng=k3,
              mixture_weights=mixture_weights_source
          )
      ),
      target_iter=iter(
          UniformMixture(
              name_target,
              batch_size=valid_batch_size,
              rng=k4,
              mixture_weights=mixture_weights_target
          )
      )
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data
