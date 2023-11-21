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
from types import MappingProxyType
from typing import Any, Iterator, Literal, Mapping, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import sklearn.datasets

__all__ = [
    "Dataset", "create_gaussian_mixture_samplers", "GaussianMixture",
    "create_sklearn_samplers", "SklearnDistribution"
]

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
    init_rng: initial PRNG key
    scale: scale of the Gaussian means
    std: the standard deviation of the individual Gaussian samples
  """
  name: Name_t
  batch_size: int
  init_rng: jax.Array
  scale: float = 5.0
  std: float = 0.5

  def __post_init__(self):
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
    rng = self.init_rng
    while True:
      rng1, rng2, rng = jax.random.split(rng, 3)
      means = jax.random.choice(rng1, self.centers, (self.batch_size,))
      normal_samples = jax.random.normal(rng2, (self.batch_size, 2))
      samples = self.scale * means + (self.std ** 2) * normal_samples
      yield samples


@dataclasses.dataclass
class SklearnDistribution:
  """A class to define toy probability 2-dimensional distributions.

  Args:
    name: the name specifying the centers of the mixture components:

      - ``moon`` - lower moon Sklearn dataset,
      - ``s_curve`` - s curve Sklearn dataset.

    init_rng: initial PRNG key
    theta_rotation: angle to define the rotation matrix to rotate samples
    offset: offset added to the Sklearn samples
    scale: scaling factor to scale the Sklearn samples
    std_noise: standard deviation of the gaussian additive noise
    batch_size: batch size of the samples
  """

  name: Literal["moon", "s_curve"]
  init_rng: jax.Array
  theta_rotation: float = 0.0
  offset: Optional[jnp.ndarray] = None
  scale: float = 1.0
  std_noise: float = 0.01
  batch_size: int = 1024

  def __iter__(self) -> Iterator[jnp.ndarray]:
    """Random sample generator from a Sklearn distribution.

    Returns:
    A generator of samples from the Sklearn distribution.
    """
    return self._create_sample_generators()

  def __post_init__(self):
    # define rotation matrix to rotate samples
    self.rotation = jnp.array([
        [jnp.cos(self.theta_rotation), -jnp.sin(self.theta_rotation)],
        [jnp.sin(self.theta_rotation),
         jnp.cos(self.theta_rotation)],
    ])

  def _create_sample_generators(self) -> Iterator[jnp.ndarray]:
    rng = jax.random.PRNGKey(0) if self.init_rng is None else self.init_rng

    while True:
      rng, _ = jax.random.split(rng)
      seed = jax.random.randint(rng, [], minval=0, maxval=1e5).item()
      if self.name == "moon":
        samples, _ = sklearn.datasets.make_moons(
            n_samples=(self.batch_size, 0),
            random_state=seed,
            noise=self.std_noise,
        )
      elif self.name == "s_curve":
        x, _ = sklearn.datasets.make_s_curve(
            n_samples=self.batch_size,
            random_state=seed,
            noise=self.std_noise,
        )
        samples = x[:, [2, 0]]
      else:
        raise NotImplementedError(
            f"SklearnDistribution `{self.name}` not implemented."
        )

      samples = jnp.asarray(samples, dtype=jnp.float32)
      samples = jnp.squeeze(jnp.matmul(self.rotation[None, :], samples.T).T)
      offset = jnp.zeros(2) if self.offset is None else self.offset
      samples = offset + self.scale * samples
      yield samples


def create_gaussian_mixture_samplers(
    name_source: Name_t,
    name_target: Name_t,
    train_batch_size: int = 2048,
    valid_batch_size: int = 2048,
    rng: Optional[jax.Array] = None,
) -> Tuple[Dataset, Dataset, int]:
  """Gaussian samplers for :class:`~ott.solvers.nn.neuraldual.W2NeuralDual`.

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
          GaussianMixture(
              name_source, batch_size=train_batch_size, init_rng=rng1
          )
      ),
      target_iter=iter(
          GaussianMixture(
              name_target, batch_size=train_batch_size, init_rng=rng2
          )
      )
  )
  valid_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(
              name_source, batch_size=valid_batch_size, init_rng=rng3
          )
      ),
      target_iter=iter(
          GaussianMixture(
              name_target, batch_size=valid_batch_size, init_rng=rng4
          )
      )
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data


def create_sklearn_samplers(
    source_kwargs: Mapping[str, Any] = MappingProxyType({}),
    target_kwargs: Mapping[str, Any] = MappingProxyType({}),
    train_batch_size: int = 256,
    valid_batch_size: int = 256,
    rng: Optional[jax.Array] = None,
) -> Tuple[Dataset, Dataset, int]:
  """Sklearn samplers for :class:`~ott.solvers.nn.neuraldual.W2NeuralDual`.

  Args:
  source_kwargs: kwargs to initialize source sampler
  target_kwargs: kwargs to initialize source sampler
  train_batch_size: the training batch size
  valid_batch_size: the validation batch size
  rng: initial PRNG key

  Returns:
  The dataset and dimension of the data.
  """
  rng = jax.random.PRNGKey(0) if rng is None else rng
  rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
  train_dataset = Dataset(
      source_iter=iter(
          SklearnDistribution(
              init_rng=rng1, batch_size=train_batch_size, **source_kwargs
          )
      ),
      target_iter=iter(
          SklearnDistribution(
              init_rng=rng2, batch_size=train_batch_size, **target_kwargs
          )
      ),
  )
  valid_dataset = Dataset(
      source_iter=iter(
          SklearnDistribution(
              init_rng=rng3, batch_size=valid_batch_size, **source_kwargs
          )
      ),
      target_iter=iter(
          SklearnDistribution(
              init_rng=rng4, batch_size=valid_batch_size, **target_kwargs
          )
      ),
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data
