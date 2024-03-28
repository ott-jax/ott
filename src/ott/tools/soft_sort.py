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
import functools
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ott import utils
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers import linear
from ott.solvers.linear import sinkhorn

__all__ = [
    "sort", "ranks", "sort_with", "quantile", "quantile_normalization",
    "quantize", "topk_mask", "multivariate_cdf_quantile_maps"
]

Func_t = Callable[[jnp.ndarray], jnp.ndarray]


def transport_for_sort(
    inputs: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    target_weights: Optional[jnp.ndarray] = None,
    squashing_fun: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    epsilon: float = 1e-2,
    **kwargs: Any,
) -> sinkhorn.SinkhornOutput:
  r"""Solve reg. OT, from inputs to a weighted family of increasing values.

  Args:
    inputs: Array[num_points]. Must be one dimensional.
    weights: Array[num_points]. Weight vector `a` for input values.
    target_weights: Array[num_targets]: Weight vector of the target
      measure. It may be of different size than `weights`.
    squashing_fun: function taking an array to squash all its entries in [0,1].
      sigmoid of whitened values by default. Can be set to be the identity by
      passing ``squashing_fun = lambda x : x`` instead.
    epsilon: the regularization parameter.
    kwargs: keyword arguments for
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.

  Returns:
    A :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput` object.
  """
  shape = inputs.shape
  if len(shape) > 2 or (len(shape) == 2 and shape[1] != 1):
    raise ValueError(
        f"Shape ({shape}) not supported. The input should be one-dimensional."
    )

  x = jnp.expand_dims(jnp.squeeze(inputs), axis=1)
  if squashing_fun is None:
    squashing_fun = lambda z: jax.nn.sigmoid((z - jnp.mean(z)) /
                                             (jnp.std(z) + 1e-10))
  x = squashing_fun(x)

  a = jnp.squeeze(weights)
  b = jnp.squeeze(target_weights)
  num_targets = inputs.shape[0] if b is None else b.shape[0]
  y = jnp.linspace(0.0, 1.0, num_targets)[:, jnp.newaxis]

  geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
  prob = linear_problem.LinearProblem(geom, a=a, b=b)

  solver = sinkhorn.Sinkhorn(**kwargs)

  return solver(prob)


def apply_on_axis(op, inputs, axis, *args, **kwargs: Any) -> jnp.ndarray:
  """Apply a differentiable operator on a given axis of the input.

  Args:
    op: a differentiable operator (can be ranks, quantile, etc.)
    inputs: Array of any shape.
    axis: the axis (int) or tuple of ints on which to apply the operator. If
      several axes are passed to the operator, those are merged as a single
      dimension.
    args: other positional arguments to the operator.
    kwargs: other positional arguments to the operator.

  Returns:
    An Array holding the output of the differentiable operator on the given
    axis.
  """
  op_inner = functools.partial(op, **kwargs)
  axis = (axis,) if isinstance(axis, int) else axis
  num_points = np.prod(np.array(inputs.shape)[(axis,)])
  permutation = np.arange(len(inputs.shape))
  axis = tuple(permutation[a] for a in axis)
  permutation = tuple(sorted(set(permutation) - set(axis)) + sorted(axis))
  inputs = jnp.transpose(inputs, permutation)

  batch_fn = jax.vmap(op_inner, in_axes=(0,) + (None,) * len(args))
  result = batch_fn(jnp.reshape(inputs, (-1, num_points)), *args)
  shrink = len(axis)
  result = jnp.reshape(result, inputs.shape[:-shrink] + result.shape[-1:])

  permutation = tuple(range(len(result.shape)))
  rank = len(result.shape) - 1
  axis = min(axis)
  permutation = permutation[:axis] + (rank,) + permutation[axis:-1]
  return jnp.transpose(result, permutation)


def _sort(
    inputs: jnp.ndarray, topk: int, num_targets: Optional[int], **kwargs: Any
) -> jnp.ndarray:
  """Apply the soft sort operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  if 0 < topk < num_points:
    start_index = 1
    b = jnp.concatenate([
        jnp.array([(num_points - topk) / num_points]),
        jnp.ones(topk) / num_points
    ])
  else:
    # Use the "sorting" initializer if default uniform weights of same size.
    if num_targets is None or num_targets == num_points:
      num_targets = num_points
      # use sorting initializer in this case.
      kwargs.setdefault("initializer", "sorting")
    start_index = 0
    b = jnp.ones((num_targets,)) / num_targets
  ot = transport_for_sort(inputs, a, b, **kwargs)
  out = 1.0 / b * ot.apply(inputs, axis=0)
  return out[start_index:]


def sort(
    inputs: jnp.ndarray,
    axis: int = -1,
    topk: int = -1,
    num_targets: Optional[int] = None,
    **kwargs: Any,
) -> jnp.ndarray:
  r"""Apply the soft sort operator on a given axis of the input.

  For instance:

  .. code-block:: python

    x = jax.random.uniform(rng, (100,))
    x_sorted = sort(x)

  will output sorted convex-combinations of values contained in ``x``, that are
  differentiable approximations to the sorted vector of entries in ``x``.
  These can be compared with the values produced by :func:`jax.numpy.sort`,

  .. code-block:: python

    x_sorted = jax.numpy.sort(x)

  Args:
    inputs: Array of any shape.
    axis: the axis on which to apply the soft-sorting operator.
    topk: if set to a positive value, the returned vector will only contain
      the top-k values. This also reduces the complexity of soft-sorting, since
      the number of target points to which the slice of the ``inputs`` tensor
      will be mapped to will be equal to ``topk + 1``.
    num_targets: if ``topk`` is not specified, a vector of size``num_targets``
      is returned. This defines the number of (composite) sorted values computed
      from the inputs (each value is a convex combination of values recorded in
      the inputs, provided in increasing order). If neither ``topk`` nor
      ``num_targets`` are specified, ``num_targets`` defaults to the size of the
      slices of the input that are sorted, i.e. ``inputs.shape[axis]``, and the
      number of composite sorted values is equal to the slice of the inputs that
      are sorted. As a result, the output is of the same size as ``inputs``.
    kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in :math:`[0,1]` (sigmoid of whitened values by default)
      to solve the optimal transport problem;
      :class:`cost_fn <ott.geometry.costs.CostFn>` object of
      :class:`~ott.geometry.pointcloud.PointCloud`, which defines the ground
      1D cost function to transport from ``inputs`` to the ``num_targets``
      target values; ``epsilon`` regularization parameter. Remaining ``kwargs``
      are passed on to parameterize the
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
    An Array of the same shape as the input, except on ``axis``, where that size
    will be equal to ``topk`` or ``num_targets``, with soft-sorted values on the
    given axis. Same size as ``inputs`` if both these parameters are ``None``.
  """
  return apply_on_axis(_sort, inputs, axis, topk, num_targets, **kwargs)


def _ranks(
    inputs: jnp.ndarray, num_targets, target_weights, **kwargs: Any
) -> jnp.ndarray:
  """Apply the soft ranks operator on a one dimensional array."""
  num_points = inputs.shape[0]
  if target_weights is None:
    num_targets = num_points if num_targets is None else num_targets
    target_weights = jnp.ones((num_targets,)) / num_targets
  else:
    num_targets = target_weights.shape[0]
  a = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(inputs, a, target_weights, **kwargs)
  out = 1.0 / a * ot.apply(jnp.arange(num_targets), axis=1)
  out *= (num_points - 1.0) / (num_targets - 1.0)
  return jnp.reshape(out, inputs.shape)


def ranks(
    inputs: jnp.ndarray,
    axis: int = -1,
    num_targets: Optional[int] = None,
    target_weights: Optional[jnp.ndarray] = None,
    **kwargs: Any,
) -> jnp.ndarray:
  r"""Apply the soft rank operator on input tensor.

  For instance:

  .. code-block:: python

    x = jax.random.uniform(rng, (100,))
    x_ranks = ranks(x)

  will output values that are differentiable approximations to the ranks of
  entries in ``x``. These should be compared to the non-differentiable rank
  vectors, namely the normalized inverse permutation produced by
  :func:`jax.numpy.argsort`, which can be obtained as:

  .. code-block:: python

    x_ranks = jax.numpy.argsort(jax.numpy.argsort(x))

  Args:
    inputs: Array of any shape.
    axis: the axis on which to apply the soft-sorting operator.
    target_weights: This vector contains weights (summing to 1) that describe
      amount of mass shipped to targets.
    num_targets: If ``target_weights`  is ``None``, ``num_targets`` is
      considered to define the number of targets used to rank inputs. Each
      rank in the output will be a convex combination of
      ``{1, .., num_targets}``. The weight of each of these points
      is assumed to be uniform. If neither ``num_targets`` nor
      ``target_weights`` are specified, ``num_targets`` defaults to the size
      of the slices of the input that are sorted, i.e. ``inputs.shape[axis]``.
    kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in :math:`[0,1]` (sigmoid of whitened values by default)
      to solve the optimal transport problem;
      :class:`cost_fn <ott.geometry.costs.CostFn>` object of
      :class:`~ott.geometry.pointcloud.PointCloud`, which defines the ground
      1D cost function to transport from ``inputs`` to the ``num_targets``
      target values; ``epsilon`` regularization parameter. Remaining ``kwargs``
      are passed on to parameterize the
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
    An Array of the same shape as the input with soft-rank values
    normalized to be in :math:`[0, n-1]` where :math:`n` is
    ``inputs.shape[axis]``.
  """
  return apply_on_axis(
      _ranks, inputs, axis, num_targets, target_weights, **kwargs
  )


def topk_mask(
    inputs: jnp.ndarray,
    axis: int = -1,
    k: int = 1,
    **kwargs: Any,
) -> jnp.ndarray:
  r"""Soft :math:`\text{top-}k` selection mask.

  For instance:

  .. code-block:: python

    k = 5
    x = jax.random.uniform(rng, (100,))
    mask = topk_mask(x, k=k)

  will output a vector of shape ``x.shape``, with values in :math:`[0,1]`, that
  are differentiable approximations to the binary mask selecting the top $k$
  entries in ``x``. These should be compared to the non-differentiable mask
  obtained with :func:`jax.numpy.sort`, which can be obtained as:

  .. code-block:: python

    mask = x >= jax.numpy.sort(x).flip()[k-1]

  Args:
    inputs: Array of any shape.
    axis: the axis on which to apply the soft-sorting operator.
    k: topk parameter. Should be smaller than ``inputs.shape[axis]``.
    kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in :math:`[0,1]` (sigmoid of whitened values by default)
      to solve the optimal transport problem;
      :class:`cost_fn <ott.geometry.costs.CostFn>` object of
      :class:`~ott.geometry.pointcloud.PointCloud`, which defines the ground
      1D cost function to transport from ``inputs`` to the ``num_targets``
      target values; ``epsilon`` regularization parameter. Remaining ``kwargs``
      are passed on to parameterize the
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
    The soft :math:`\text{top-}k` selection mask.
  """
  num_points = inputs.shape[axis]
  assert k < num_points, (
      f"`k` must be smaller than `inputs.shape[axis]`, yet {k} >= {num_points}."
  )
  target_weights = jnp.array([1.0 - k / num_points, k / num_points])
  out = apply_on_axis(
      _ranks,
      inputs,
      axis,
      num_targets=None,
      target_weights=target_weights,
      **kwargs
  )
  return out / (num_points - 1)


def quantile(
    inputs: jnp.ndarray,
    q: Optional[Union[float, jnp.ndarray]],
    axis: Union[int, Tuple[int, ...]] = -1,
    weight: Optional[Union[float, jnp.ndarray]] = None,
    **kwargs: Any,
) -> jnp.ndarray:
  r"""Apply the soft quantiles operator on the input tensor.

  For instance:

  .. code-block:: python

    x = jax.random.uniform(rng, (100,))
    x_quantiles = quantile(x, q=jnp.array([0.2, 0.8]))

  ``x_quantiles`` will hold an approximation to the 20 and 80 percentiles in
  ``x``, computed as a convex combination (a weighted mean, with weights summing
  to 1) of all values in ``x`` (and not, as for standard quantiles, the
  values ``x_sorted[20]`` and ``x_sorted[80]`` if ``x_sorted=jnp.sort(x)``).
  These values offer a trade-off between accuracy (closeness to the true
  percentiles) and gradient (the Jacobian of ``x_quantiles`` w.r.t ``x`` will
  impact all values listed in ``x``, not just those indexed at 20 and 80).

  The non-differentiable version is given by :func:`jax.numpy.quantile`, e.g.

  .. code-block:: python

    x_quantiles = jax.numpy.quantile(x, q=jnp.array([0.2, 0.8]))

  Args:
   inputs: an Array of any shape.
   q: values of the quantile level to be computed, e.g. [0.5] for median.
     These values should all lie within the segment :math:`]0,1[`, excluding
     boundary values :math:`0` and :math:`1`.
   axis: the axis on which to apply the operator.
   weight: the weight assigned to each quantile target value in the OT problem.
     This weight should be small, typically of the order of ``1/n``, where ``n``
     is the size of ``x``. Note: Since the size of ``q`` times ``weight``
     must be strictly smaller than ``1``, in order to leave enough mass to set
     other target values in the transport problem, the algorithm might ensure
     this by setting, when needed, a lower value.
   kwargs: keyword arguments passed on to lower level functions. Of interest
     to the user are ``squashing_fun``, which will redistribute the values in
     ``inputs`` to lie in :math:`[0,1]` (sigmoid of whitened values by default)
     to solve the optimal transport problem;
     :class:`cost_fn <ott.geometry.costs.CostFn>` object of
     :class:`~ott.geometry.pointcloud.PointCloud`, which defines the ground
     1D cost function to transport from ``inputs`` to the ``num_targets``
     target values; ``epsilon`` regularization parameter. Remaining ``kwargs``
     are passed on to parameterize the
     :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
    An Array, which has the same shape as ``inputs``, except on the ``axis``
    that is passed, which has size ``q.shape[0]``, to collect soft-quantile
    values.
  """

  def _quantile(
      inputs: jnp.ndarray, q: float, weight: float, **kwargs
  ) -> jnp.ndarray:
    num_points = inputs.shape[0]
    q = jnp.array([0.2, 0.5, 0.8]) if q is None else jnp.atleast_1d(q)
    num_quantiles = q.shape[0]
    a = jnp.ones((num_points,)) / num_points
    idx = jnp.argsort(q)
    q = q[idx]

    extended_q = jnp.concatenate([jnp.array([0.0]), q, jnp.array([1.0])])
    filler_weights = extended_q[1:] - extended_q[:-1]
    safe_weight = 0.5 * jnp.concatenate([
        jnp.array([1.0 / num_quantiles]), filler_weights
    ])
    if weight is None:
      # Populate with other options.
      safe_weight = jnp.concatenate([
          safe_weight,
          jnp.array(
              [0.02]
          ),  # reasonable mass per quantile for a small number of points
          jnp.array(
              [1.5 / num_points]
          ),  # this means each quantile would be ~ assigned 1.5 points.
      ])
    else:
      safe_weight = jnp.concatenate([safe_weight, jnp.atleast_1d(weight)])
    weight = jnp.min(safe_weight)
    weights = jnp.ones(filler_weights.shape) * weight

    # Takes into account quantile_width in the definition of weights
    shift = -jnp.ones(filler_weights.shape)
    shift = shift + 0.5 * (
        jax.nn.one_hot(0, num_quantiles + 1) +
        jax.nn.one_hot(num_quantiles, num_quantiles + 1)
    )
    filler_weights = filler_weights + weights * shift

    # Adds one more value to have tensors of the same shape to interleave them.
    quantile_weights = jnp.ones(num_quantiles + 1) * weights

    # Interleaves the filler_weights with the quantile weights.
    weights = jnp.reshape(
        jnp.stack([filler_weights, quantile_weights], axis=1), (-1,)
    )[:-1]

    ot = transport_for_sort(inputs, a, weights, **kwargs)
    out = 1.0 / weights * ot.apply(jnp.squeeze(inputs), axis=0)

    # Recover odd indices corresponding to the desired quantiles.
    odds = np.concatenate([
        np.zeros((num_quantiles + 1, 1), dtype=bool),
        np.ones((num_quantiles + 1, 1), dtype=bool)
    ],
                          axis=1).ravel()[:-1]
    return out[odds][idx]

  return apply_on_axis(_quantile, inputs, axis, q, weight, **kwargs)


def multivariate_cdf_quantile_maps(
    inputs: jnp.ndarray,
    target_sampler: Optional[Callable[[jax.Array, Tuple[int, int]],
                                      jax.Array]] = None,
    rng: Optional[jax.Array] = None,
    num_target_samples: Optional[int] = None,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    input_weights: Optional[jnp.ndarray] = None,
    target_weights: Optional[jnp.ndarray] = None,
    **kwargs: Any
) -> Tuple[Func_t, Func_t]:
  r"""Returns multivariate CDF and quantile maps, given input samples.

  Implements the multivariate generalizations for CDF and quantiles proposed in
  :cite:`chernozhukov:17`. The reference measure is assumed to be the uniform
  measure by default, but can be modified. For consistency, the reference
  measure should be symmetrically centered around
  :math:`(\tfrac{1}{2},\cdots,\tfrac{1}{2})` and supported on :math:`[0, 1]^d`.

  The implementation return two entropic map estimators, one for the CDF map,
  the other for the quantiles map.

  Args:
    inputs: 2D array of ``[n, d]`` vectors.
    target_sampler: Callable that takes a ``rng`` and ``[m, d]`` shape.
      ``m`` is passed on as ``target_num_samples``, dimension ``d`` is inferred
      directly from the shape passed in ``inputs``. This is assumed by default
      to be :func:`~jax.random.uniform`, and could be any other random sampler
      properly wrapped to have the signature above.
    rng: rng key used by ``target_sampler``.
    num_target_samples: number ``m`` of points generated in the target
      distribution.
    cost_fn: Cost function, used to compare ``inputs`` and ``targets``.
      Passed on to instantiate a
      :class:`~ott.geometry.pointcloud.PointCloud` object. If :obj:`None`,
      :class:`~ott.geometry.costs.SqEuclidean` is used.
    epsilon: entropic regularization parameter used to instantiate the
      :class:`~ott.geometry.pointcloud.PointCloud` object.
    input_weights: ``[n,]`` vector of weights for input measure. Assumed to
      be uniform by default.
    target_weights: ``[m,]`` vector of weights for target measure. Assumed
      to be uniform by default.
    kwargs: keyword arguments passed on to the :func:`~ott.solvers.linear.solve`
      function, which solves the OT problem between ``inputs`` and ``targets``
      using the :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` algorithm.

  Returns:
    - The multivariate CDF map, taking a ``[b, d]`` batch of vectors in the
      range of the ``inputs``, and mapping each vector within the range
      of the reference measure (assumed by default to be :math:`[0, 1]^d`).
    - The quantile map, mapping a batch ``[b, d]`` of multivariate quantile
      vectors onto ``[b, d]`` vectors in :math:`[0, 1]^d`, the range of
      the reference measure.
  """
  n, d = inputs.shape
  rng = utils.default_prng_key(rng)

  if num_target_samples is None:
    num_target_samples = n
  if target_sampler is None:
    target_sampler = jax.random.uniform

  targets = target_sampler(rng, (num_target_samples, d))
  geom = pointcloud.PointCloud(
      inputs, targets, cost_fn=cost_fn, epsilon=epsilon
  )

  out = linear.solve(geom, a=input_weights, b=target_weights, **kwargs)
  potentials = out.to_dual_potentials()

  cdf_map = jtu.Partial(lambda x, p: p.transport(x), p=potentials)
  quantile_map = jtu.Partial(
      lambda x, p: p.transport(x, forward=False), p=potentials
  )
  return cdf_map, quantile_map


def _quantile_normalization(
    inputs: jnp.ndarray, targets: jnp.ndarray, weights: float, **kwargs: Any
) -> jnp.ndarray:
  """Apply soft quantile normalization on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(inputs, a, weights, **kwargs)
  return 1.0 / a * ot.apply(targets, axis=1)


def quantile_normalization(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    axis: int = -1,
    **kwargs: Any,
) -> jnp.ndarray:
  r"""Re-normalize inputs so that its quantiles match those of targets/weights.

  Quantile normalization rearranges the values in inputs to values that match
  the distribution of values described in the discrete distribution ``targets``
  weighted by ``weights``. This transformation preserves the order of values
  in ``inputs`` along the specified ``axis``.

  Args:
    inputs: array of any shape whose values will be changed to match those in
      ``targets``.
    targets: sorted array (in ascending order) of dimension 1 describing a
      discrete distribution. Note: the ``targets`` values must be provided as
      a sorted vector.
    weights: vector of non-negative weights, summing to :math:`1`, of the same
      size as ``targets``. When not set, this defaults to the uniform
      distribution.
    axis: the axis along which the quantile transformation is applied.
    kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in :math:`[0,1]` (sigmoid of whitened values by default)
      to solve the optimal transport problem;
      :class:`cost_fn <ott.geometry.costs.CostFn>` object of
      :class:`~ott.geometry.pointcloud.PointCloud`, which defines the ground
      1D cost function to transport from ``inputs`` to the ``num_targets``
      target values; ``epsilon`` regularization parameter. Remaining ``kwargs``
      are passed on to parameterize the
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
    An array, which has the same shape as the input, except on the give axis on
    which the dimension is 1.

  Raises:
    A ValueError in case the weights and the targets are both set and not of
    compatible shapes.
  """
  if weights is not None and weights.shape != targets.shape:
    raise ValueError(
        "The target weights and targets values should have the "
        f"same shape: {targets.shape} != {weights.shape}"
    )
  if weights is None:
    num_targets = targets.shape[0]
    weights = jnp.ones((num_targets,)) / num_targets

  op = _quantile_normalization
  return apply_on_axis(op, inputs, axis, targets, weights, **kwargs)


def sort_with(
    inputs: jnp.ndarray,
    criterion: jnp.ndarray,
    topk: int = -1,
    **kwargs: Any,
) -> jnp.ndarray:
  r"""Sort a multidimensional array according to a real valued criterion.

  Given ``batch`` vectors of dimension `dim`, to which, for each, a real value
  ``criterion`` is associated, this function produces ``topk`` (or
  ``batch`` if ``topk`` is set to -1, as by default) composite vectors of size
  ``dim`` that will be convex combinations of all vectors, ranked from smallest
  to largest criterion. Composite vectors with the largest indices will contain
  convex combinations of those vectors with highest criterion, vectors with
  smaller indices will contain combinations of vectors with smaller criterion.

  Args:
    inputs: Array of size [batch, dim].
    criterion: the values according to which to sort the inputs. It has shape
      [batch, 1].
    topk: The number of outputs to keep.
    kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in :math:`[0,1]` (sigmoid of whitened values by default)
      to solve the optimal transport problem;
      :class:`cost_fn <ott.geometry.costs.CostFn>` object of
      :class:`~ott.geometry.pointcloud.PointCloud`, which defines the ground
      1D cost function to transport from ``inputs`` to the ``num_targets``
      target values; ``epsilon`` regularization parameter. Remaining ``kwargs``
      are passed on to parameterize the
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
    An Array of size [batch | topk, dim].
  """
  num_points = criterion.shape[0]
  weights = jnp.ones(num_points) / num_points
  if 0 < topk < num_points:
    start_index = 1
    target_weights = jnp.concatenate([
        jnp.array([(num_points - topk) / num_points]),
        jnp.ones(topk) / num_points
    ])
  else:
    start_index = 0
    target_weights = jnp.ones((num_points,)) / num_points
  ot = transport_for_sort(criterion, weights, target_weights, **kwargs)
  # Applies the topk on each of the dimensions of the inputs.
  sort_fn = jax.vmap(
      lambda x: (1.0 / target_weights * ot.apply(x, axis=0))[start_index:],
      in_axes=(1,),
      out_axes=1
  )
  return sort_fn(inputs)


def _quantize(inputs: jnp.ndarray, num_q: int, **kwargs: Any) -> jnp.ndarray:
  """Apply the soft quantization operator on a one dimensional array."""
  num_points = inputs.shape[0]
  a = jnp.ones((num_points,)) / num_points
  b = jnp.ones((num_q,)) / num_q
  ot = transport_for_sort(inputs, a, b, **kwargs)
  return 1.0 / a * ot.apply(1.0 / b * ot.apply(inputs), axis=1)


def quantize(
    inputs: jnp.ndarray,
    num_levels: int = 10,
    axis: int = -1,
    **kwargs: Any,
) -> jnp.ndarray:
  r"""Soft quantizes an input according using ``num_levels`` values along axis.

  The quantization operator consists in concentrating several values around
  a few predefined ``num_levels``. The soft quantization operator proposed here
  does so by carrying out a `soft` concentration that is differentiable. The
  ``inputs`` values are first soft-sorted, resulting in ``num_levels`` values.
  In a second step, the ``inputs`` values are then provided again a composite
  value that is equal (for each) to a convex combination of those soft-sorted
  values using the transportation matrix. As the regularization parameter
  ``epsilon`` of regularized optimal transport goes to 0, this operator recovers
  the expected behavior of quantization, namely each value in ``inputs`` is
  assigned a single level. When using ``epsilon>0`` the behavior is similar but
  differentiable.

  Args:
    inputs: an Array of size [batch, dim].
    num_levels: number of quantiles available to quantize the signal.
    axis: axis along which quantization is carried out.
    kwargs: keyword arguments passed on to lower level functions. Of interest
      to the user are ``squashing_fun``, which will redistribute the values in
      ``inputs`` to lie in :math:`[0,1]` (sigmoid of whitened values by default)
      to solve the optimal transport problem;
      :class:`cost_fn <ott.geometry.costs.CostFn>` object of
      :class:`~ott.geometry.pointcloud.PointCloud`, which defines the ground
      1D cost function to transport from ``inputs`` to the ``num_targets``
      target values; ``epsilon`` regularization parameter. Remaining ``kwargs``
      are passed on to parameterize the
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
    An Array of the same size as ``inputs``.
  """
  return apply_on_axis(_quantize, inputs, axis, num_levels, **kwargs)
