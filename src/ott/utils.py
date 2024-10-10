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
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeVar, Union

# TODO(michalk8): add to requirements
from typing_extensions import ParamSpec

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import batching

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
    "batched_vmap",
]

IOStatus = Tuple[np.ndarray, np.ndarray, np.ndarray, NamedTuple]
IOCallback = Callable[[IOStatus], None]
P = ParamSpec("P")
R = TypeVar("R")


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
    func: Optional[Callable[P, R]] = None
) -> Callable[P, R]:

  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
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
  return jax.random.key(0) if rng is None else rng


def default_progress_fn(
    fmt: str = "{iter} / {max_iter} -- {error}",
    stream: Optional[io.TextIOBase] = None,
) -> IOCallback:
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
      from ott.solvers import linear

      x = jax.random.normal(jax.random.key(0), (100, 5))
      geom = pointcloud.PointCloud(x)

      progress_fn = utils.default_progress_fn()
      solve_fn = jax.jit(linear.solve, static_argnames=["progress_fn"])
      out = solve_fn(geom, progress_fn=progress_fn)
  """  # noqa: D205

  def progress_callback(status: IOStatus) -> None:
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
) -> IOCallback:
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
      from ott.solvers import linear

      x = jax.random.normal(jax.random.key(0), (100, 5))
      geom = pointcloud.PointCloud(x)

      with tqdm.tqdm() as pbar:
        progress_fn = utils.tqdm_progress_fn(pbar)
        solve_fn = jax.jit(linear.solve, static_argnames=["progress_fn"])
        out = solve_fn(geom, progress_fn=progress_fn)
  """  # noqa: D205

  def progress_callback(status: IOStatus) -> None:
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


def _prepare_info(status: IOStatus) -> Tuple[int, int, int, np.ndarray]:
  iteration, inner_iterations, total_iter, state = status
  iteration = int(iteration) + 1
  inner_iterations = int(inner_iterations)
  total_iter = int(total_iter)
  errors = np.array(state.errors).ravel()

  return iteration, inner_iterations, total_iter, errors


def _batch_and_remainder(
    args: Any,
    *,
    batch_size: int,
    in_axes: Optional[Union[int, Sequence[int], Any]],
) -> Tuple[Any, Any]:
  leaves, in_tree = jax.tree.flatten(args, is_leaf=batching.is_vmappable)
  in_axes = jax.api_util.flatten_axes(
      "vmap in_axes", in_tree, in_axes, kws=True
  )

  has_scan, has_remainder = False, False
  scan_leaves, remainder_leaves = [], []
  for leaf, axis in zip(leaves, in_axes):
    if axis is None:
      scan_leaf = remainder_leaf = leaf
    else:
      assert batch_size > 0, batch_size
      num_splits, _ = divmod(leaf.shape[axis], batch_size)
      num_elems = num_splits * batch_size

      scan_leaf = jax.lax.slice_in_dim(leaf, None, num_elems, axis=axis)
      new_shape = leaf.shape[:axis] + (-1, batch_size) + leaf.shape[axis + 1:]
      scan_leaf = scan_leaf.reshape(new_shape)
      remainder_leaf = jax.lax.slice_in_dim(leaf, num_elems, None, axis=axis)

      has_scan = has_scan or scan_leaf.shape[axis]
      has_remainder = has_remainder or remainder_leaf.shape[axis]

    scan_leaves.append(scan_leaf)
    remainder_leaves.append(remainder_leaf)

  assert has_scan or has_remainder, "TODO"

  scan_tree = in_tree.unflatten(scan_leaves) if has_scan else None
  remainder_tree = in_tree.unflatten(
      remainder_leaves
  ) if has_remainder else None
  return scan_tree, remainder_tree


def _apply_scan(
    vmapped_fun: Callable[P, R], in_axes: Optional[Union[int, Sequence[int],
                                                         Any]]
) -> Callable[P, R]:

  def select_batch(arg: Any, index: int, *, axis: int) -> Any:
    fn = functools.partial(
        jax.lax.dynamic_index_in_dim, index=index, axis=axis, keepdims=False
    )
    return jax.tree.map(fn, arg)

  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:

    def body_fn(carry: None, index: int) -> Tuple[None, R]:
      del carry
      new_args = tuple(
          arg if axis is None else select_batch(arg, index, axis=axis)
          for arg, axis in zip(args, axes)
      )
      return None, vmapped_fun(*new_args, **kwargs)

    in_tree = jax.tree.structure(args, is_leaf=batching.is_vmappable)
    axes = jax.api_util.flatten_axes("vmap in_axes", in_tree, in_axes, kws=True)

    ix = next(ix for ix, axis in enumerate(axes) if axis is not None)
    leaf, *_ = jax.tree.leaves(args[ix])
    axis, *_ = jax.tree.leaves(axes[ix])
    xs = np.arange(leaf.shape[axis])

    _, result = jax.lax.scan(body_fn, init=None, xs=xs)
    return result

  return wrapper


def batched_vmap(
    fun: Callable[P, R],
    *,
    batch_size: int,
    in_axes: Optional[Union[int, Sequence[int], Any]] = 0,
    # TODO(michalk8): correctly handle non-negative indices
    out_axes: Any = 0,
) -> Callable[P, R]:
  """TODO."""

  def unbatch(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    x = jnp.moveaxis(x, 0, axis)
    return jax.lax.collapse(x, axis, axis + 2)

  def concat(x: jnp.ndarray, y: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.concatenate([x, y], axis=axis)

  @functools.wraps(fun)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    batched, remainder = _batch_and_remainder(
        args, batch_size=batch_size, in_axes=in_axes
    )
    has_batched = batched is not None
    has_remainder = remainder is not None

    if has_batched:
      batched = batched_fun(*batched, **kwargs)
      batched = jax.tree.map(unbatch, batched, out_axes)
    if has_remainder:
      remainder = vmapped_fun(*remainder, **kwargs)

    if has_batched and has_remainder:
      return jax.tree.map(concat, batched, remainder, out_axes)
    if has_batched:
      return batched
    # TODO(michalk8): check for empty arrays
    return remainder

  if isinstance(in_axes, list):
    in_axes = tuple(in_axes)

  vmapped_fun = jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)
  batched_fun = _apply_scan(vmapped_fun, in_axes=in_axes)

  return wrapper
