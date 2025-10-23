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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from ott.solvers.linear import semidiscrete

__all__ = ["SemidiscreteDataloader"]


@dataclasses.dataclass(frozen=False, repr=False)
class SemidiscreteDataloader:
  """Semidiscrete dataloader.

  The dataloader samples from the continuous source distribution and
  couples it with the discrete target distribution. It returns aligned tuples of
  ``(source, target)`` arrays of shape ``[batch_size, ...]``.

  Args:
    out: Semidiscrete output.
    batch_size: Batch size.
    rng: Random number seed.
    subset_threshold: Threshold above which to sample from a subset of the
      coupling matrix. Only applicable when the problem is :meth:`entropically
      regularized <ott.geometry.semidiscrete_pointcloud.SemidiscretePointCloud.is_entropy_regularized>`.
      If :obj:`None`, don't subset the coupling matrix.
    subset_size: Size of the subset of the coupling matrix. This will
      subset a coupling of shape ``[batch, m]`` to ``[batch, subset_size]``
      keeping the :func:`top-k <jax.lax.top_k>` values if ``m > subset_threshold``.
    out_sharding: Output sharding.
  """  # noqa: E501
  out: semidiscrete.SemidiscreteOutput
  batch_size: int
  rng: jax.Array
  subset_threshold: Optional[int] = None
  subset_size: Optional[int] = None
  out_sharding: Optional[jax.sharding.Sharding] = None

  def __post_init__(self) -> None:
    _, m = self.out.geom.shape
    assert self.batch_size > 0, \
      f"Batch size must be positive, got {self.batch_size}."

    if self.subset_threshold is not None:
      assert 0 < self.subset_threshold < m, \
        f"Subset threshold must be in (0, {m}), got {self.subset_threshold}."
      assert 0 < self.subset_size < m, \
        f"Subset size must be in (0, {m}), got {self.subset_size}."

    self._rng_it: Optional[jax.Array] = None
    self._sample_fn = jax.jit(
        _sample,
        out_shardings=self.out_sharding,
        static_argnames=["batch_size", "subset_threshold", "subset_size"],
    )

  def __iter__(self) -> "SemidiscreteDataloader":
    """Return self."""
    self._rng_it = self.rng
    return self

  def __next__(self) -> Tuple[jax.Array, jax.Array]:
    """Sample from the source distribution and match it with the data.

    Returns:
      A tuple of samples and data, arrays of shape ``[batch_size, ...]``.
    """
    assert self._rng_it is not None, "Please call `iter()` first."
    self._rng_it, rng_sample = jr.split(self._rng_it, 2)
    return self._sample_fn(
        rng_sample,
        self.out,
        self.batch_size,
        self.subset_threshold,
        self.subset_size,
    )


def _sample(
    rng: jax.Array,
    out: semidiscrete.SemidiscreteOutput,
    batch_size: int,
    subset_threshold: Optional[int],
    subset_size: int,
) -> Tuple[jax.Array, jax.Array]:
  rng_sample, rng_tmat = jr.split(rng, 2)
  out_sampled = out.sample(rng_sample, batch_size)

  if isinstance(out_sampled, semidiscrete.HardAssignmentOutput):
    tgt_idx = out_sampled.paired_indices[1]
  else:
    tgt_idx = _sample_from_coupling(
        rng_tmat,
        coupling=out_sampled.matrix,
        subset_threshold=subset_threshold,
        subset_size=subset_size,
        axis=1,
    )

  src = out_sampled.geom.x
  tgt = out_sampled.geom.y[tgt_idx]
  return src, tgt


def _sample_from_coupling(
    rng: jax.Array,
    *,
    coupling: jax.Array,
    subset_threshold: Optional[int],
    subset_size: int,
    axis: int,
) -> jax.Array:
  assert axis in (0, 1), axis
  n, m = coupling.shape
  sampling_size = m if axis == 1 else n

  if subset_threshold is None or sampling_size <= subset_threshold:
    return jr.categorical(rng, jnp.log(coupling), axis=axis)

  oaxis = 1 - axis
  top_k_fn = jax.vmap(jax.lax.top_k, in_axes=[oaxis, None], out_axes=oaxis)
  coupling, idx = top_k_fn(coupling, subset_size)

  expected_shape = (subset_size, m) if axis == 0 else (n, subset_size)
  assert coupling.shape == expected_shape, (coupling.shape, expected_shape)

  sampled_idx = jr.categorical(rng, jnp.log(coupling), axis=axis)
  if axis == 0:
    return idx[sampled_idx, jnp.arange(m)]
  return idx[jnp.arange(n), sampled_idx]
