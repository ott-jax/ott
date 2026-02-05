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
from typing import Any, Mapping, Optional, Sequence

import pytest

import jax
import jax.experimental

import matplotlib as mpl


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
  return jax.random.key(0)


@pytest.fixture()
def enable_x64() -> bool:
  ctx = (
      jax.enable_x64(True)
      if hasattr(jax, "enable_x64") else jax.experimental.enable_x64(True)
  )
  with ctx:
    try:
      yield
    finally:
      pass
