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
import itertools
from collections import abc
from typing import Any, Iterator, Mapping, Optional, Sequence

import pytest

import jax

import matplotlib as mpl

from tests import _utils


def pytest_addoption(parser: pytest.Parser) -> None:
  parser.addoption(
      "--jax-compilation-cache",
      metavar="DIR",
      default=None,
      help="Persist compiled XLA kernels in DIR and reuse them across runs. "
      "Most of the suite's runtime is compilation, so this makes repeated "
      "runs several times faster.",
  )
  parser.addoption(
      "--strict-rng",
      action="store_true",
      help="Fail if a PRNG key is consumed twice. Reusing a key silently "
      "correlates draws that are meant to be independent.",
  )


def pytest_configure(config: pytest.Config) -> None:
  if config.getoption("--strict-rng"):
    jax.config.update("jax_debug_key_reuse", True)

  cache_dir = config.getoption("--jax-compilation-cache")
  if cache_dir is None:
    return
  jax.config.update("jax_compilation_cache_dir", str(cache_dir))
  # none of these kernels reach JAX's 1s default persist threshold
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def pytest_sessionstart(session: pytest.Session) -> None:
  mpl.use("Agg")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
  if not hasattr(metafunc.function, "pytestmark"):
    # no annotation
    return

  fast_marks = [m for m in metafunc.function.pytestmark if m.name == "fast"]
  if fast_marks:
    mark, = fast_marks
    selected: Optional[Mapping[str, Any]] = mark.kwargs.pop("only_fast", None)
    ids: Optional[Sequence[str]] = mark.kwargs.pop("ids", None)

    if mark.args:
      argnames, argvalues = mark.args
    else:
      argnames = tuple(mark.kwargs.keys())
      argvalues = [(vs,) if not isinstance(vs, (str, abc.Iterable)) else vs
                   for vs in mark.kwargs.values()]
      argvalues = list(itertools.product(*argvalues))

    opt = str(metafunc.config.getoption("-m"))
    if "fast" in opt:  # filter if `-m fast` was passed
      if selected is None:
        combinations = argvalues
      elif isinstance(selected, dict):
        combinations = []
        for vs in argvalues:
          if selected == dict(zip(argnames, vs)):
            combinations.append(vs)
      elif isinstance(selected, (tuple, list)):
        # TODO(michalk8): support passing ids?
        combinations = [argvalues[s] for s in selected]
        ids = None if ids is None else [ids[s] for s in selected]
      elif isinstance(selected, int):
        combinations = [argvalues[selected]]
        ids = None if ids is None else [ids[selected]]
      else:
        raise TypeError(f"Invalid fast selection type `{type(selected)}`.")
    else:
      combinations = argvalues

    if argnames:
      metafunc.parametrize(argnames, combinations, ids=ids)


@pytest.fixture()
def rng() -> jax.Array:
  """A root random key, fresh for every test.

  Deliberately not shared across tests: ``--strict-rng`` rejects consuming
  one key twice, and a shared key would trip on the second test to use it.
  Fixtures that outlive a single test seed themselves via
  :func:`tests._utils.root_key` instead of depending on this.
  """
  return _utils.root_key()


@pytest.fixture()
def enable_x64() -> Iterator[None]:
  """Run the test in double precision."""
  with jax.enable_x64(True):
    yield


@pytest.fixture(scope="session")
def clouds() -> _utils.PointClouds:
  """Two weighted point clouds with strictly positive marginals.

  Modules needing different data - marginals with exact zeros, other sizes -
  override this fixture at module or class level.
  """
  return _utils.random_clouds(_utils.root_key())
