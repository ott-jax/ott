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
from typing import Optional

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott import utils


class TestBatchedVmap:

  @pytest.mark.parametrize("batch_size", [1, 11, 32, 33])
  def test_batch_size(self, rng: jax.Array, batch_size: int):
    x = jax.random.normal(rng, (32, 2))
    gt_fn = jax.jit(jax.vmap(jnp.sum))
    fn = jax.jit(utils.batched_vmap(jnp.sum, batch_size=batch_size))

    np.testing.assert_array_equal(gt_fn(x), fn(x))

  def test_pytree(self, rng: jax.Array):
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    x = {
        "foo": {
            "bar": jax.random.normal(rng1, (5, 3, 3))
        },
        "baz": jax.random.normal(rng2, (2, 5)),
        "quux": (2.0, 3.0),
    }
    in_axes = [{"foo": {"bar": 0}, "baz": 1, "quux": (None, None)}]

    f = lambda x: x["foo"]["bar"].std() + x["baz"].mean() + x["quux"][0]
    gt_fn = jax.vmap(f, in_axes=in_axes)
    fn = utils.batched_vmap(f, in_axes=in_axes, batch_size=2)

    np.testing.assert_array_equal(gt_fn(x), fn(x))

  def test_empty_arrays(self):
    pass

  def test_no_remainder(self):
    pass

  def test_in_axes(self):
    pass

  def test_max_memory(self):
    pass


@pytest.mark.parametrize(("version", "msg"), [(None, "foo, bar, baz"),
                                              ("quux", None)])
def test_deprecation_warning(version: Optional[str], msg: Optional[str]):

  @utils.deprecate(version=version, alt=msg)
  def func() -> int:
    return 42

  expected_msg = rf"`{func.__name__}`"
  if version is None:
    expected_msg += r".*next release\."
  else:
    expected_msg += rf".*`ott-jax=={version}` release\."
  if msg is not None:
    expected_msg += f" {msg}"

  with pytest.warns(DeprecationWarning, match=expected_msg):
    res = func()
  assert res == 42
