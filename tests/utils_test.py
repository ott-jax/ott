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
from ott import utils


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
