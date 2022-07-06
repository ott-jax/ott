import collections
import itertools
from typing import Any, Mapping, Optional

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
    # TODO(michalk8): handle mark.args?
    selected: Optional[Mapping[str, Any]] = mark.kwargs.pop("only_fast", None)
    argnames = tuple(mark.kwargs.keys())
    argvalues = [(vs,) if not isinstance(vs,
                                         (str, collections.Iterable)) else vs
                 for vs in mark.kwargs.values()]
    argvalues = list(itertools.product(*argvalues))

    opt = str(metafunc.config.getoption("-m"))
    if "fast" in opt:  # filter if `-m fast` was passed
      if isinstance(selected, dict):
        combinations = []
        for vs in argvalues:
          if selected == dict(zip(argnames, vs)):
            combinations.append(vs)
      elif isinstance(selected, int):
        # TODO(michalk8): limit only when 1 parametrized value
        # convenient way of specifying by indexing
        combinations = [argvalues[selected]]
      else:
        raise TypeError(f"Invalid fast selection type `{type(selected)}`.")
    else:
      combinations = argvalues

    metafunc.parametrize(argnames, combinations)


@pytest.fixture(scope="session")
def rng():
  return jax.random.PRNGKey(0)
