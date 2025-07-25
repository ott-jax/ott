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
import functools
from typing import Any, Optional

import pytest

import chex
import jax
import jax.numpy as jnp
import numpy as np

from ott import utils


@pytest.mark.fast()
class TestBatchedVmap:

  @pytest.mark.parametrize("batch_size", [1, 11, 32, 33])
  def test_batch_size(self, rng: jax.Array, batch_size: int):
    x = jax.random.normal(rng, (32, 2))
    gt_fn = jax.jit(jax.vmap(jnp.sum))
    fn = jax.jit(utils.batched_vmap(jnp.sum, batch_size=batch_size))

    np.testing.assert_array_almost_equal(gt_fn(x), fn(x), decimal=4)

  def test_pytree(self, rng: jax.Array):

    def f(x: Any) -> jnp.ndarray:
      return x["foo"]["bar"].std() + x["baz"].mean(
      ) + x["quux"][0] * x["quux"][1]

    rng1, rng2, rng3 = jax.random.split(rng, 3)
    x = {
        "foo": {
            "bar": jax.random.normal(rng1, (5, 3, 3))
        },
        "baz": jax.random.normal(rng2, (2, 5)),
        "quux": (2.0, 3.0),
    }
    in_axes = [{"foo": {"bar": 0}, "baz": 1, "quux": (None, None)}]

    gt_fn = jax.vmap(f, in_axes=in_axes)
    fn = utils.batched_vmap(f, in_axes=in_axes, batch_size=2)

    np.testing.assert_array_equal(gt_fn(x), fn(x), rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("batch_size", [1, 7, 67, 133])
  @pytest.mark.parametrize("in_axes", [0, 1, -1, -2, [0, None], (0, -2)])
  def test_in_axes(self, rng: jax.Array, in_axes: Any, batch_size: int):

    def f(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
      x = jnp.atleast_2d(x)
      y = jnp.atleast_2d(y)
      return jnp.dot(x, y.T)

    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (133, 71)) + 10.0
    y = jax.random.normal(rng2, (133, 71))

    gt_fn = jax.jit(jax.vmap(f, in_axes=in_axes))
    fn = jax.jit(utils.batched_vmap(f, batch_size=batch_size, in_axes=in_axes))

    np.testing.assert_allclose(gt_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize(
      "in_axes", [((0, None, None),), (({
          "foo": 1,
          "baz": -1
      }, -1, None),), [
          ({
              "foo": {
                  "bar": 0
              },
              "baz": -2
          }, None, None),
      ]]
  )
  def test_in_axes_pytree(self, rng: jax.Array, in_axes: Any):

    def f(tree: Any) -> jnp.ndarray:
      x = tree[0]["foo"]["bar"]
      y = tree[0]["baz"]
      z, ((v,), w) = tree[1], tree[2]
      return x.mean() - y.std() + z.sum() - y * (v + w)

    batch_size = 1
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    tree = (
        {
            "foo": {
                "bar": jax.random.normal(rng1, (13, 5))
            },
            "baz": jax.random.normal(rng2, (13, 5))
        },
        jax.random.normal(rng3, (10, 5)),
        ((2,), 13.0),
    )

    gt_fn = jax.jit(jax.vmap(f, in_axes=in_axes))
    fn = jax.jit(utils.batched_vmap(f, in_axes=in_axes, batch_size=batch_size))

    chex.assert_trees_all_close(gt_fn(tree), fn(tree), rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize("out_axes", [0, 1, 2, -1, -2, -3])
  def test_out_axes(self, rng: jax.Array, out_axes: int):

    def f(x: jnp.ndarray, y: jnp.ndarray) -> Any:
      return (x.sum() + y.sum()).reshape(1, 1)

    rng1, rng2 = jax.random.split(rng, 2)
    x = jax.random.normal(rng1, (31, 13))
    y = jax.random.normal(rng2, (31, 6)) - 15.0

    gt_fn = jax.vmap(f, out_axes=out_axes)
    fn = utils.batched_vmap(f, batch_size=5, out_axes=out_axes)

    chex.assert_trees_all_close(gt_fn(x, y), fn(x, y), rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize(
      "out_axes", [0, (0, 0, 1), (0, {
          "x": {
              "y": 1
          }
      }, (1,))]
  )
  def test_out_axes_pytree(self, rng: jax.Array, out_axes: Any):

    def f(x: jnp.ndarray) -> Any:
      z = jnp.arange(9).reshape(3, 3)
      return x.mean(), {"x": {"y": jnp.ones(13)}}, (z,)

    x = jax.random.normal(rng, (13, 5))

    fn = utils.batched_vmap(f, batch_size=12, out_axes=out_axes)
    gt_fn = jax.vmap(f, out_axes=out_axes)

    chex.assert_trees_all_close(gt_fn(x), fn(x), rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("n", [16, 7])
  @pytest.mark.parametrize("batch_size", [1, 4, 5, 7, 16])
  def test_max_traces(self, rng: jax.Array, batch_size: int, n: int):
    max_traces = 1 + (n % batch_size != 0)

    @jax.jit
    @functools.partial(utils.batched_vmap, batch_size=batch_size)
    @chex.assert_max_traces(n=max_traces)
    def fn(x: jnp.ndarray) -> jnp.ndarray:
      return x.sum()

    chex.clear_trace_counter()
    x = jax.random.normal(rng, (n, 3))

    np.testing.assert_array_almost_equal(fn(x), x.sum(1), decimal=4)

  @pytest.mark.limit_memory("20MB")
  def test_vmap_max_memory(self, rng: jax.Array):
    n, m, d = 2 ** 16, 2 ** 11, 3
    rng, rng_data = jax.random.split(rng, 2)
    y = jax.random.normal(rng_data, (m, d))

    fn = utils.batched_vmap(
        lambda x, y: jnp.dot(y, x).sum(), in_axes=[0, None], batch_size=128
    )
    fn = jax.jit(fn)

    rng, rng_data = jax.random.split(rng, 2)
    x = jax.random.normal(rng_data, (n, d))
    res = fn(x, y)
    assert res.shape == (n,)

  @pytest.mark.parametrize("batch_size", [1, 5, 10])
  def test_inconsistent_array_sizes(self, rng: jax.Array, batch_size: int):
    rng1, rng2 = jax.random.split(rng, 2)

    x = jax.random.normal(rng1, (5, 2))
    y = jax.random.normal(rng2, (10, 2))

    gt_fn = jax.vmap(lambda x, y: (x + y).sum(), in_axes=0)
    fn = utils.batched_vmap(
        lambda x, y: (x + y).sum(), batch_size=batch_size, in_axes=0
    )

    with pytest.raises(ValueError, match=r"^vmap got inconsistent"):
      _ = gt_fn(x, y)
    num_splits = x.shape[0] // batch_size
    wrong_num_splits = y.shape[0] // batch_size
    with pytest.raises(
        AssertionError,
        match=rf"^Expected {num_splits} splits, got {wrong_num_splits}\."
    ):
      _ = fn(x, y)


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
