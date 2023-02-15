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
"""pytree_nodes Dataclasses."""
import dataclasses
import functools
import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["register_pytree_node", "deprecate", "is_jax_array"]


def register_pytree_node(cls: type) -> type:
  """Register dataclasses as pytree_nodes."""
  cls = dataclasses.dataclass()(cls)
  flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
  unflatten = lambda d, children: cls(**d.unflatten(children))
  jax.tree_util.register_pytree_node(cls, flatten, unflatten)
  return cls


def deprecate(  # noqa: D103
    *,
    version: Optional[str] = None,
    alt: Optional[str] = None,
    func: Optional[Callable[[Any], Any]] = None
) -> Callable[[Any], Any]:

  def wrapper(*args: Any, **kwargs: Any) -> Any:
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    return func(*args, **kwargs)

  if func is None:
    return lambda fn: deprecate(version=version, alt=alt, func=fn)

  msg = f"`{func.__name__}` will be removed in the "
  msg += ("next" if version is None else f"`ott-jax=={version}`") + " release."
  if alt:
    msg += " " + alt

  return functools.wraps(func)(wrapper)


def is_jax_array(obj: Any) -> bool:
  """Check if an object is a Jax array."""
  if hasattr(jax, "Array"):
    # https://jax.readthedocs.io/en/latest/jax_array_migration.html
    return isinstance(obj, (jax.Array, jnp.DeviceArray))
  return isinstance(obj, jnp.DeviceArray)


def default_progress_fn(
    status: Tuple[np.ndarray, np.ndarray, np.ndarray, NamedTuple], *args
) -> None:
  """This default callback function reports progress by printing to the console.

  Args:
      status: tuple describing the current iteration number,
        the number of inner iterations after which the error is computed,
        the total number of iterations, each wrapped in a ``jax.Array``, and
        the current Sinkhorn state.
      args: unused, see <https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html>

  This function updates the progress bar only when the error is computed
  (every `solver.inner_iterations`, typically `10`).

  Note: the user can provide a slightly modified version of this callback in
  order to use a progress bar. To do so, replace the print statement with the
  following:

    ```
    pbar.set_description_str(f"error: {error:.6f}")
    pbar.total = total_iter
    pbar.update()
    ```

  assuming `pbar` is defined at the callsite with the following code:

    ```
    with tqdm() as pbar:
      out_sink = jax.jit(sinkhorn.Sinkhorn(progress_fn=progress_fn))(lin_problem)
    ```

  """
  # Convert arguments.
  iteration, inner_iterations, total_iter, state = status
  iteration = int(iteration)
  inner_iterations = int(inner_iterations)
  total_iter = int(total_iter)
  errors = np.array(state.errors).ravel()

  # Avoid reporting error on each iteration,
  # because errors are only computed every `inner_iterations`.
  if (iteration + 1) % inner_iterations == 0:
    error_idx = max((iteration + 1) // inner_iterations - 1, 0)
    error = errors[error_idx]

    # pbar.set_description_str(f"error: {error:.6f}")
    # pbar.total = total_iter
    # pbar.update()

    print(f"{iteration} / {total_iter} -- {error}")
