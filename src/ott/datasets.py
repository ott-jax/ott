# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
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

__all__ = [
    "create_gaussian_mixture_samplers",
    "create_conditional_gaussian_mixture_samplers",
    "ConditionalDataset",
    "Dataset",
    "GaussianMixture",
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


class ConditionalDataset(NamedTuple):
  r"""Samplers from conditional source and target measures.

  Args:
    source_iter: loader for the source measure, ``[batch, d]``
    target_iter: loader for the target measure, ``[batch, d]``
    condition_iter: loader for condition vectors,
      ``[batch, dim_c]``
    label_iter: loader for integer condition labels, ``[batch]``
  """

  source_iter: Iterator[jnp.ndarray]
  target_iter: Iterator[jnp.ndarray]
  condition_iter: Iterator[jnp.ndarray]
  label_iter: Iterator[jnp.ndarray]


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
      ),
  )
  valid_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(name_source, batch_size=valid_batch_size, rng=rng3)
      ),
      target_iter=iter(
          GaussianMixture(name_target, batch_size=valid_batch_size, rng=rng4)
      ),
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data


@dataclasses.dataclass
class ConditionalGaussianMixture:
  """Conditional Gaussian sampler for testing.

  For each condition *k*, draws source ~ N(0, I) and
  target ~ source + offset_k.
  Condition vectors are one-hot encoded labels.

  Args:
    num_conditions: number of distinct conditions.
    batch_size: total batch size (divided equally among conditions).
    dim: data dimensionality.
    offsets: ``[num_conditions, dim]`` translation per condition.
    rng: initial PRNG key.
  """

  num_conditions: int
  batch_size: int
  dim: int
  offsets: jnp.ndarray
  rng: jax.Array

  def __iter__(self) -> Iterator[Tuple[jnp.ndarray, ...]]:
    return self._generate()

  def _generate(self) -> Iterator[Tuple[jnp.ndarray, ...]]:
    rng = self.rng
    per_cond = self.batch_size // self.num_conditions
    while True:
      rng, rng_s = jax.random.split(rng)
      sources, targets, conds, labels = [], [], [], []
      for k in range(self.num_conditions):
        rng_s, rng_k = jax.random.split(rng_s)
        s = jax.random.normal(rng_k, (per_cond, self.dim))
        t = s + self.offsets[k]
        c = jnp.zeros((per_cond, self.num_conditions)).at[:, k].set(1.0)
        lab = jnp.full((per_cond,), k, dtype=jnp.int32)
        sources.append(s)
        targets.append(t)
        conds.append(c)
        labels.append(lab)
      yield (
          jnp.concatenate(sources),
          jnp.concatenate(targets),
          jnp.concatenate(conds),
          jnp.concatenate(labels),
      )


def create_conditional_gaussian_mixture_samplers(
    num_conditions: int = 3,
    dim: int = 2,
    train_batch_size: int = 90,
    valid_batch_size: int = 90,
    rng: Optional[jax.Array] = None,
) -> Tuple[ConditionalDataset, ConditionalDataset, int, int, int]:
  """Create conditional Gaussian samplers for testing.

  Each condition defines a different translation of the source distribution.

  Args:
    num_conditions: number of distinct conditions.
    dim: data dimensionality.
    train_batch_size: training batch size (should be divisible by
      ``num_conditions``).
    valid_batch_size: validation batch size.
    rng: initial PRNG key.

  Returns:
    ``(train_dataset, valid_dataset, dim_data, num_conditions,
    max_measure_size)`` where ``max_measure_size =
    batch_size // num_conditions``.
  """
  rng = utils.default_prng_key(rng)
  rng1, rng2, rng_off = jax.random.split(rng, 3)

  # Each condition has a different offset (translation)
  offsets = jax.random.normal(rng_off, (num_conditions, dim)) * 3.0

  def _make_dataset(
      bs: int,
      key: jax.Array,
  ) -> ConditionalDataset:
    sampler = ConditionalGaussianMixture(
        num_conditions=num_conditions,
        batch_size=bs,
        dim=dim,
        offsets=offsets,
        rng=key,
    )
    gen = iter(sampler)
    # Cache the current batch so all 4 iterators stay synchronized.
    cache = {}

    def _next_batch():
      if "batch" not in cache:
        cache["batch"] = next(gen)
      return cache

    def _iter(idx: int) -> Iterator[jnp.ndarray]:
      while True:
        c = _next_batch()
        val = c["batch"][idx]
        # Mark consumed; when all 4 are done, clear cache.
        c.setdefault("consumed", set())
        c["consumed"].add(idx)
        if len(c["consumed"]) == 4:
          cache.clear()
        yield val

    return ConditionalDataset(
        source_iter=_iter(0),
        target_iter=_iter(1),
        condition_iter=_iter(2),
        label_iter=_iter(3),
    )

  train_ds = _make_dataset(train_batch_size, rng1)
  valid_ds = _make_dataset(valid_batch_size, rng2)
  max_measure_size = train_batch_size // num_conditions
  return train_ds, valid_ds, dim, num_conditions, max_measure_size
