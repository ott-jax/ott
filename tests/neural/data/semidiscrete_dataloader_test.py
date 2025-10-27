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
from typing import Optional, Tuple

import pytest

import jax
import jax.random as jr
import jax.sharding as jsh
import numpy as np

import optax

from ott.geometry import semidiscrete_pointcloud as sdpc
from ott.neural.data import semidiscrete_dataloader as sddl
from ott.solvers import linear
from ott.solvers.linear import semidiscrete


def _solve_semidiscrete(
    rng: jax.Array, *, shape: Tuple[int, ...], epsilon: float
) -> semidiscrete.SemidiscreteOutput:
  rng_data, rng_solve = jr.split(rng, 2)
  y = jr.normal(rng, shape)
  geom = sdpc.SemidiscretePointCloud(jr.normal, y, epsilon=epsilon)
  return linear.solve_semidiscrete(
      geom,
      num_iterations=10,
      batch_size=12,
      optimizer=optax.sgd(1e-2),
      rng=rng_solve
  )


class TestSemidiscreteDataloader:

  @pytest.mark.parametrize("batch_size", [10, 22])
  def test_reproducibility(self, rng: jax.Array, batch_size: int):
    m, d = 19, 2
    rng_solve, rng_dl = jr.split(rng, 2)
    out = _solve_semidiscrete(rng_solve, shape=(m, d), epsilon=0.1)
    dl = sddl.SemidiscreteDataloader(rng, out, batch_size=batch_size)

    src1, tgt1 = next(iter(dl))
    src2, tgt2 = next(iter(dl))

    assert src1.shape == (batch_size, d)
    assert tgt1.shape == (batch_size, d)
    np.testing.assert_array_equal(src1, src2)
    np.testing.assert_array_equal(tgt1, tgt2)

  def test_invalid_values(self, rng: jax.Array):
    m, d = 15, 5
    rng_solve, rng_dl = jr.split(rng, 2)
    out = _solve_semidiscrete(rng_solve, shape=(m, d), epsilon=0.2)

    with pytest.raises(AssertionError, match=r"Batch size must be positive"):
      _ = sddl.SemidiscreteDataloader(rng_dl, out, batch_size=0)
    with pytest.raises(AssertionError, match=r"Subset threshold must be in"):
      _ = sddl.SemidiscreteDataloader(
          rng_dl, out, batch_size=32, subset_size_threshold=0
      )
    with pytest.raises(AssertionError, match=r"Subset threshold must be in"):
      _ = sddl.SemidiscreteDataloader(
          rng_dl, out, batch_size=32, subset_size_threshold=m
      )
    with pytest.raises(AssertionError, match=r"Subset size must be in"):
      _ = sddl.SemidiscreteDataloader(
          rng_dl,
          out,
          batch_size=32,
          subset_size_threshold=m // 2,
          subset_size=0,
      )
    with pytest.raises(AssertionError, match=r"Subset size must be in"):
      _ = sddl.SemidiscreteDataloader(
          rng_dl,
          out,
          batch_size=32,
          subset_size_threshold=m // 2,
          subset_size=m,
      )

  @pytest.mark.parametrize(("subset_size_threshold", "subset_size"), [(7, 7),
                                                                      (8, 4),
                                                                      (4, 11)])
  def test_subset_size_threshold(
      self, rng: jax.Array, subset_size_threshold: int, subset_size: int
  ):
    m, d = 15, 5
    batch_size = 6
    rng_solve, rng_dl = jr.split(rng, 2)
    out = _solve_semidiscrete(rng_solve, shape=(m, d), epsilon=0.2)

    dl = sddl.SemidiscreteDataloader(
        rng_dl,
        out,
        batch_size=batch_size,
        subset_size_threshold=subset_size_threshold,
        subset_size=subset_size,
    )

    src, tgt = next(iter(dl))
    assert src.shape == (batch_size, d)
    assert tgt.shape == (batch_size, d)

  @pytest.mark.parametrize("epsilon", [0.0, 1e-2, None])
  def test_sharding(self, rng: jax.Array, epsilon: Optional[float]):
    m, d = 11, 4
    batch_size = 11
    rng_solve, rng_dl = jr.split(rng, 2)
    out = _solve_semidiscrete(rng_solve, shape=(m, d), epsilon=epsilon)

    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    sharding = jsh.NamedSharding(mesh, jsh.PartitionSpec("data"))
    dl = sddl.SemidiscreteDataloader(
        rng_dl, out, batch_size=batch_size, out_shardings=sharding
    )

    src, tgt = next(iter(dl))
    assert src.shape == (batch_size, d)
    assert src.sharding == sharding
    assert tgt.shape == (batch_size, d)
    assert tgt.sharding == sharding
