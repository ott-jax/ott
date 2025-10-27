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
from typing import Iterable, Tuple

import jax
import jax.random as jr
import jax.sharding as jsh
import numpy as np

from ott.neural.data import ot_dataloader


def _get_dataset(
    rng: jax.Array, shape: Tuple[int, ...]
) -> Iterable[Tuple[jax.Array, jax.Array]]:
  while True:
    rng, rng_x0, rng_x1 = jr.split(rng, 3)
    x0 = jr.normal(rng_x0, shape)
    x1 = jr.normal(rng_x1, shape)
    yield x0, x1


class TestLinearOtDataloader:

  def test_reproducibility(self, rng: jax.Array):
    rng_ds, rng_dl = jr.split(rng, 2)
    shape = (32, 2)

    ds1 = _get_dataset(rng_ds, shape)
    ds2 = _get_dataset(rng_ds, shape)

    dl1 = ot_dataloader.LinearOTDataloader(rng_dl, ds1)
    dl2 = ot_dataloader.LinearOTDataloader(rng_dl, ds2)

    src1, tgt1 = next(iter(dl1))
    src2, tgt2 = next(iter(dl2))

    assert src1.shape == shape
    assert tgt1.shape == shape

    np.testing.assert_array_equal(src1, src2)
    np.testing.assert_array_equal(tgt1, tgt2)

  def test_sharding(self, rng: jax.Array):
    rng_ds, rng_dl = jr.split(rng, 2)
    shape = (32, 2)

    ds = _get_dataset(rng_ds, shape)

    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    sharding = jsh.NamedSharding(mesh, jsh.PartitionSpec("data"))
    dl = ot_dataloader.LinearOTDataloader(rng_dl, ds, shardings=sharding)

    src, tgt = next(iter(dl))

    assert src.sharding == sharding
    assert tgt.sharding == sharding
