from typing import Optional

import pytest

from ott import utils


@pytest.mark.parametrize(
    "version,msg", [(None, "foo, bar, baz"), ("quux", None)]
)
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
