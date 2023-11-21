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
  """A class to define 2-dimensional Gaussian mixtures.

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
      - ``swiss`` - swiss roll Sklearn dataset

    init_rng: initial PRNG key
    dim_data: data dimensionality
    theta_rotation: angle to define the rotation matrix to rotate samples
    offset: offset added to the Sklearn samples
    scale: scaling factor to scale the Sklearn samples
    std_noise: standard deviation of the gaussian additive noise
    batch_size: batch size of the samples
  """

  name: Literal["moon", "s_curve", "swiss"]
  init_rng: jax.Array
  dim_data: int = 2
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

    # check dimension consistency with dataset
    if self.dim_data > 2:
      assert self.name != "moon", (
          "Moon disribution only supported in dimension 2."
      )
      assert self.theta_rotation == 0., (
          "Samples rotation only supported in dimension 2."
      )

    # define offset
    self._offset = jnp.zeros(
        self.dim_data
    ) if self.offset is None else self.offset

    # define rotation matrix to rotate samples
    if self.theta_rotation == 0.:
      self.rotation_fn = lambda x: x
    else:
      rotation_marix = jnp.array([
          [jnp.cos(self.theta_rotation), -jnp.sin(self.theta_rotation)],
          [jnp.sin(self.theta_rotation),
           jnp.cos(self.theta_rotation)],
      ])
      self.rotation_fn = lambda x: jnp.matmul(rotation_marix, x)

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
        samples = x[:, [2, 0]] if self.dim_data == 2 else x
      elif self.name == "swiss":
        x, _ = sklearn.datasets.make_swiss_roll(
            n_samples=self.batch_size,
            random_state=seed,
            noise=self.std_noise,
        )
        samples = x[:, [2, 0]] if self.dim_data == 2 else x
      else:
        raise NotImplementedError(
            f"SklearnDistribution `{self.name}` not implemented."
        )

      samples = jnp.asarray(samples, dtype=jnp.float32)
      samples = jax.vmap(self.rotation_fn)(samples)
      samples = self._offset + self.scale * samples
      yield samples


@dataclasses.dataclass
class SortedSprial:
  """A class to define 2 or 3-dimensional ordered spiral distributions.

  The spiral is sorted in the sense that the indices of the points in each
  generated batch follow the progression of the spiral, i.e. the angles to draw
  the points are linearly increasing between ``min_angle`` and ``max_angle``.
  Afterwards, the first point of a batch point always has a norm close to
  ``min_radius``, while the last point of the batch always has a norm close
  to ``max_radius``, and the intermediate points have a linearly increasing
  norm between ``min_radius`` and ``max_raidus``.

  Args:
    init_rng: initial PRNG key
    dim_data: data dimensionality
    min_angle: angle we start from to draw the spiral
    max_angle: angle of the spiral's ending point
    std_noise: standard deviation of the gaussian additive noise
    batch_size: batch size of the samples
  """

  init_rng: jax.Array
  dim_data: int = 2
  min_radius: float = 3.
  max_radius: float = 10.
  min_angle: float = 0.
  max_angle: float = 10.
  std_noise: float = 0.01
  batch_size: int = 1024

  def __iter__(self) -> Iterator[jnp.ndarray]:
    """Random sample generator from an ordered spiral distribution.

    Returns:
    A generator of samples from the ordered spiral distribution.
    """
    return self._create_sample_generators()

  def _create_sample_generators(self) -> Iterator[jnp.ndarray]:
    rng = jax.random.PRNGKey(0) if self.init_rng is None else self.init_rng

    while True:
      rng, rng1, rng2 = jax.random.split(rng, 3)
      radius = jnp.linspace(self.min_radius, self.max_radius, self.batch_size)
      angles = jnp.linspace(self.min_angle, self.max_angle, self.batch_size)
      noise = self.std_noise * jax.random.normal(rng1, (self.batch_size, 2))
      x_coordinates = (radius + noise[:, 0]) * jnp.cos(angles)
      y_coordinates = (radius + noise[:, 1]) * jnp.sin(angles)
      samples = jnp.concatenate(
          (x_coordinates[:, jnp.newaxis], y_coordinates[:, jnp.newaxis]),
          axis=1
      )
      if self.dim_data == 3:
        third_axis = jax.random.uniform(
            rng2, (self.batch_size, 1)
        ) * self.max_radius
        samples = jnp.hstack((samples[:, 0:1], third_axis, samples[:, 1:]))

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
