import collections.abc
import itertools
from typing import Any, Mapping, Optional, Sequence

import jax
import pytest
from _pytest.python import Metafunc


def pytest_generate_tests(metafunc: Metafunc) -> None:
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
      argvalues = [
          (vs,) if not isinstance(vs, (str, collections.abc.Iterable)) else vs
          for vs in mark.kwargs.values()
      ]
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

    metafunc.parametrize(argnames, combinations, ids=ids)


@pytest.fixture(scope="session")
def rng():
  return jax.random.PRNGKey(0)
