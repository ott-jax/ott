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
import io
import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import numpy as np

try:
  from tqdm import tqdm
except ImportError:
  tqdm = Any

__all__ = [
    "register_pytree_node",
    "deprecate",
    "default_prng_key",
    "default_progress_fn",
    "tqdm_progress_fn",
]

Status_t = Tuple[np.ndarray, np.ndarray, np.ndarray, NamedTuple]
IOCallback_t = Callable[[Status_t], None]


def register_pytree_node(cls: type) -> type:
  """Register dataclasses as pytree_nodes."""
  cls = dataclasses.dataclass()(cls)
  flatten = lambda obj: jax.tree_util.tree_flatten(dataclasses.asdict(obj))
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


def default_prng_key(rng: Optional[jax.Array] = None) -> jax.Array:
  """Get the default PRNG key.

  Args:
    rng: PRNG key.

  Returns:
    If ``rng = None``, returns the default PRNG key.
    Otherwise, it returns the unmodified ``rng`` key.
  """
  return jax.random.PRNGKey(0) if rng is None else rng


def default_progress_fn(
    fmt: str = "{iter} / {max_iter} -- {error}",
    stream: Optional[io.TextIOBase] = None,
) -> IOCallback_t:
  """Return a callback that prints the progress when solving
  :mod:`linear problems <ott.problems.linear>`.

  It prints the progress only when the error is computed, that is every
  :attr:`~ott.solvers.linear.sinkhorn.Sinkhorn.inner_iterations`.

  Args:
    fmt: Format used to print. It can format ``iter``, ``max_iter`` and
      ``error`` values.
    stream: Output IO stream.

  Returns:
    A callback function accepting the following arguments

    - the current iteration number,
    - the number of inner iterations after which the error is computed,
    - the total number of iterations, and
    - the current :class:`~ott.solvers.linear.sinkhorn.SinkhornState` or
      :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornState`.

  Examples:
    .. code-block:: python

      import jax
      import jax.numpy as jnp

      from ott import utils
      from ott.geometry import pointcloud
      from ott.solvers.linear import sinkhorn

      x = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
      geom = pointcloud.PointCloud(x)

      progress_fn = utils.default_progress_fn()
      solve_fn = jax.jit(sinkhorn.solve, static_argnames=["progress_fn"])
      out = solve_fn(geom, progress_fn=progress_fn)
  """  # noqa: D205

  def progress_callback(status: Status_t) -> None:
    iteration, inner_iterations, total_iter, errors = _prepare_info(status)
    # Avoid reporting error on each iteration,
    # because errors are only computed every `inner_iterations`.
    if iteration % inner_iterations == 0:
      error_idx = max(0, iteration // inner_iterations - 1)
      error = errors[error_idx]

      print(
          fmt.format(iter=iteration, max_iter=total_iter, error=error),
          file=stream
      )

  return progress_callback


def tqdm_progress_fn(
    pbar: tqdm,
    fmt: str = "error: {error:0.6e}",
) -> IOCallback_t:
  """Return a callback that updates a progress bar when solving
  :mod:`linear problems <ott.problems.linear>`.

  It updates the progress bar only when the error is computed, that is every
  :attr:`~ott.solvers.linear.sinkhorn.Sinkhorn.inner_iterations`.

  Args:
    pbar: `tqdm <https://tqdm.github.io/docs/tqdm/>`_ progress bar.
    fmt: Format used for the postfix. It can format ``iter``, ``max_iter`` and
      ``error`` values.

  Returns:
    A callback function accepting the following arguments

    - the current iteration number,
    - the number of inner iterations after which the error is computed,
    - the total number of iterations, and
    - the current :class:`~ott.solvers.linear.sinkhorn.SinkhornState` or
      :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornState`.

  Examples:
    .. code-block:: python

      import tqdm

      import jax
      import jax.numpy as jnp

      from ott import utils
      from ott.geometry import pointcloud
      from ott.solvers.linear import sinkhorn

      x = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
      geom = pointcloud.PointCloud(x)

      with tqdm.tqdm() as pbar:
        progress_fn = utils.tqdm_progress_fn(pbar)
        solve_fn = jax.jit(sinkhorn.solve, static_argnames=["progress_fn"])
        out = solve_fn(geom, progress_fn=progress_fn)
  """  # noqa: D205

  def progress_callback(status: Status_t) -> None:
    iteration, inner_iterations, total_iter, errors = _prepare_info(status)
    # Avoid reporting error on each iteration,
    # because errors are only computed every `inner_iterations`.
    if iteration % inner_iterations == 0:
      error_idx = max(0, iteration // inner_iterations - 1)
      error = errors[error_idx]

      postfix = fmt.format(iter=iteration, max_iter=total_iter, error=error)
      pbar.set_postfix_str(postfix)
      pbar.total = total_iter // inner_iterations
      pbar.update()

  return progress_callback


def _prepare_info(status: Status_t) -> Tuple[int, int, int, np.ndarray]:
  iteration, inner_iterations, total_iter, state = status
  iteration = int(iteration) + 1
  inner_iterations = int(inner_iterations)
  total_iter = int(total_iter)
  errors = np.array(state.errors).ravel()

  return iteration, inner_iterations, total_iter, errors
