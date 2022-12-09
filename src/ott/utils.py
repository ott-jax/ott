# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""pytree_nodes Dataclasses."""
import dataclasses
import functools
import warnings
from typing import Any, Callable, Optional

import jax

__all__ = ["register_pytree_node", "deprecate"]


def register_pytree_node(cls: type) -> type:
  """Register dataclasses as pytree_nodes."""
  cls = dataclasses.dataclass()(cls)
  flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
  unflatten = lambda d, children: cls(**d.unflatten(children))
  jax.tree_util.register_pytree_node(cls, flatten, unflatten)
  return cls


def deprecate(
    *,
    version: Optional[str] = None,
    alt_msg: Optional[str] = None,
    func: Optional[Callable[[Any], Any]] = None
) -> Callable[[Any], Any]:

  def wrapper(*args: Any, **kwargs: Any) -> Any:
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    return func(*args, **kwargs)

  if func is None:
    return lambda fn: deprecate(version=version, alt_msg=alt_msg, func=fn)

  if version is None:
    msg = f"`{func.__name__}` will be removed the next version."
  else:
    msg = f"`{func.__name__}` will be removed in version `{version}`."
  if alt_msg:
    msg += " " + alt_msg

  return functools.wraps(func)(wrapper)
