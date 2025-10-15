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
from typing import Dict, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from ott.solvers.linear import semidiscrete

__all__ = ["SemidiscreteLoader"]

Element = Dict[Literal["src", "tgt"], jax.Array]


@dataclasses.dataclass(frozen=False, repr=False)
class SemidiscreteLoader:
  """TODO."""
  out: semidiscrete.SemidiscreteOutput
  batch_size: int
  rng: jax.Array
  max_sampling_size: int = 65536
  top_k: int = 1024
  out_sharding: Optional[jax.sharding.Sharding] = None

  def __post_init__(self) -> None:
    self._sample_fn = jax.jit(
        _sample,
        out_shardings=self.out_sharding,
        static_argnames=["batch_size", "max_sampling_size", "top_k"],
    )

  def __iter__(self) -> "SemidiscreteLoader":
    return self

  def __next__(self) -> Element:
    self.rng, rng_sample = jr.split(self.rng, 2)
    return self._sample_fn(
        rng_sample,
        self.out,
        self.batch_size,
        self.max_sampling_size,
        self.top_k,
    )


def _sample(
    rng: jax.Array,
    out: semidiscrete.SemidiscreteOutput,
    batch_size: int,
    max_sampling_size: int,
    top_k: int,
) -> Element:
  rng_sample, rng_tmat = jr.split(rng, 2)
  out_sampled = out.sample(rng_sample, batch_size)

  if isinstance(out_sampled, semidiscrete.HardAssignmentOutput):
    tgt_idx = out_sampled.paired_indices[1]
  else:
    tgt_idx = _sample_from_coupling(
        rng_tmat,
        coupling=out_sampled.matrix,
        max_sampling_size=max_sampling_size,
        top_k=top_k,
        axis=1
    )

  src = out_sampled.geom.x
  tgt = out_sampled.geom.y[tgt_idx]
  return {"src": src, "tgt": tgt}


def _sample_from_coupling(
    rng: jax.Array,
    *,
    coupling: jax.Array,
    max_sampling_size: int,
    top_k: int,
    axis: int,
) -> jax.Array:
  assert axis in (0, 1), axis
  n, m = coupling.shape
  sampling_size = m if axis == 1 else n

  if sampling_size <= max_sampling_size:
    return jr.categorical(rng, jnp.log(coupling), axis=axis)

  oaxis = 1 - axis
  top_k_fn = jax.vmap(jax.lax.top_k, in_axes=[oaxis, None], out_axes=oaxis)
  coupling, idx = top_k_fn(coupling, top_k)

  expected_shape = (top_k, m) if axis == 0 else (n, top_k)
  assert coupling.shape == expected_shape, (coupling.shape, expected_shape)

  sampled_idx = jr.categorical(rng, jnp.log(coupling), axis=axis)
  if axis == 0:
    return idx[sampled_idx, jnp.arange(m)]
  return idx[jnp.arange(n), sampled_idx]
