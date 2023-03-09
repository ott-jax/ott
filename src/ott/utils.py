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
import dataclasses
import functools
import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "register_pytree_node", "deprecate", "is_jax_array", "default_progress_fn"
]


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
    status: Tuple[np.ndarray, np.ndarray, np.ndarray, NamedTuple], *args: Any
) -> None:
  """Callback function that reports progress of
  :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` by printing to the console.

  It updates the progress bar only when the error is computed, that is every
  :attr:`~ott.solvers.linear.sinkhorn.Sinkhorn.inner_iterations`.

  Note:
    This function is called during solver iterations via
    :func:`~jax.experimental.host_callback.id_tap` so the solver execution
    remains :func:`jittable <jax.jit>`.

  Args:
    status: status consisting of:

      - the current iteration number
      - the number of inner iterations after which the error is computed
      - the total number of iterations
      - the current :class:`~ott.solvers.linear.sinkhorn.SinkhornState`

    args: unused, see :mod:`jax.experimental.host_callback`.

  Returns:
    Nothing, just prints.

  Examples:
    If instead of printing you want to report progress using a progress bar such
    as `tqdm <https://tqdm.github.io>`_, then simply provide a slightly modified
    version of this callback, for instance:

    .. code-block:: python

      import jax
      import numpy as np
      from tqdm import tqdm

      from ott.problems.linear import linear_problem
      from ott.solvers.linear import sinkhorn

      def progress_fn(status, *args):
        iteration, inner_iterations, total_iter, state = status
        iteration = int(iteration)
        inner_iterations = int(inner_iterations)
        total_iter = int(total_iter)
        errors = np.asarray(state.errors).ravel()

        # Avoid reporting error on each iteration,
        # because errors are only computed every `inner_iterations`.
        if (iteration + 1) % inner_iterations == 0:
          error_idx = max((iteration + 1) // inner_iterations - 1, 0)
          error = errors[error_idx]

          pbar.set_postfix_str(f"error: {error:0.6e}")
          pbar.total = total_iter
          pbar.update()

      prob = linear_problem.LinearProblem(...)
      solver = sinkhorn.Sinkhorn(progress_fn=progress_fn)

      with tqdm() as pbar:
        out_sink = jax.jit(solver)(prob)
  """  # noqa: D205
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

    print(f"{iteration} / {total_iter} -- {error}")  # noqa: T201
