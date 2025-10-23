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
from typing import Tuple

import pytest

import jax
import jax.random as jr
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
    dim = 2
    rng_solve, rng_dl = jr.split(rng, 2)
    out = _solve_semidiscrete(rng_solve, shape=(19, dim), epsilon=0.1)
    dl = sddl.SemidiscreteDataloader(out, batch_size=batch_size, rng=rng_dl)

    src1, tgt1 = next(iter(dl))
    src2, tgt2 = next(iter(dl))

    assert src1.shape == (batch_size, dim)
    assert tgt1.shape == (batch_size, dim)
    np.testing.assert_array_equal(src1, src2)
    np.testing.assert_array_equal(tgt1, tgt2)

  def test_batch_size(self, rng: jax.Array):
    pass

  def test_subset_threshold(self):
    pass

  def test_sharding(self):
    pass
