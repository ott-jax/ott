import collections
import itertools
from typing import Any, Mapping, Optional

import jax
import pytest
from _pytest.python import Metafunc


def pytest_generate_tests(metafunc: Metafunc) -> None:
  fast_marks = [m for m in metafunc.function.pytestmark if m.name == "fast"]
  if fast_marks:
    mark, = fast_marks
    selected: Optional[Mapping[str, Any]] = mark.kwargs.pop("only_fast", None)
    argnames = tuple(mark.kwargs.keys())
    argvalues = [(vs,) if not isinstance(vs,
                                         (str, collections.Iterable)) else vs
                 for vs in mark.kwargs.values()]
    argvalues = list(itertools.product(*argvalues))

    opt = str(metafunc.config.getoption("-m"))
    if selected is not None and "fast" in opt:
      # filter if `-m fast` was passed
      combinations = []
      for vs in argvalues:
        if selected == dict(zip(argnames, vs)):
          combinations.append(vs)
    else:
      combinations = argvalues

    metafunc.parametrize(argnames, combinations)


@pytest.fixture(scope="session")
def rng():
  return jax.random.PRNGKey(0)
