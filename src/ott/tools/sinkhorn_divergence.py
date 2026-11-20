# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp

from ott.geometry import costs, geometry, pointcloud, segment
from ott.problems.linear import linear_problem, potentials
from ott.solvers import linear
from ott.solvers.linear import acceleration, sinkhorn, sinkhorn_lr

__all__ = [
    "sinkhorn_divergence", "segment_sinkhorn_divergence",
    "SinkhornDivergenceOutput"
]

Potentials = Tuple[jnp.ndarray, jnp.ndarray]
Factors = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
Output = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput]


class SinkhornDivergenceOutput(NamedTuple):
  r"""Holds the outputs of a call to :func:`sinkhorn_divergence`.

  Objects of this class contain both solutions and problem definition of the
  two or three regularized OT problems instantiated when computing a Sinkhorn
  divergence between two probability distributions.

  When ``static_b=True``, the problem between the second measure and itself is
  not solved: its geometry, potentials or factors and errors are :obj:`None`,
  it is marked as converged and its number of iterations is :math:`0`.

  Args:
    divergence: value of the Sinkhorn divergence
    geoms: three geometries describing the Sinkhorn divergence, of respective
      sizes ``[n, m], [n, n], [m, m]`` if their cost or kernel matrices were
      instantiated.
    a: first ``[n,]`` vector of marginal weights.
    b: second ``[m,]`` vector of marginal weights.
    potentials: three pairs of dual potential vectors, of sizes
      ``[n,], [m,]``, ``[n,], [n,]``, ``[m,], [m,]``, returned when the call
      to the :func:`~ott.solvers.linear.solve` solver to compute the divergence
      relies on a vanilla :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.
    factors: three triplets of matrices, of sizes
      ``([n, rank], [m, rank], [rank,])``, ``([n, rank], [n, rank], [rank,])``
      and ``([m, rank], [m, rank], [rank,])``, returned when the call
      to the :func:`~ott.solvers.linear.solve` solver to compute the divergence
      relies on a low-rank :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`
      solver.
    errors: errors of each of the three problems, see
      :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.errors`.
    converged: triplet of booleans indicating the convergence of each of the
      three problems run to compute the divergence.
    n_iters: number of iterations keeping track of compute effort needed to
      complete each of the three terms in the divergence.
  """
  divergence: float
  geoms: Tuple[geometry.Geometry, geometry.Geometry,
               Optional[geometry.Geometry]]
  a: jnp.ndarray
  b: jnp.ndarray
  potentials: Optional[Tuple[Potentials, Potentials, Potentials]]
  factors: Optional[Tuple[Factors, Factors, Factors]]
  errors: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray],
                Optional[jnp.ndarray]]
  converged: Tuple[bool, bool, bool]
  n_iters: Tuple[int, int, int]

  def to_dual_potentials(
      self, epsilon: Optional[float] = None
  ) -> potentials.DualPotentials:
    """Return dual potential functions :cite:`pooladian:22`.

    Using vectors stored in ``potentials``, instantiate a dual potentials object
    that will provide approximations to optimal dual potential functions for
    the dual OT problem defined for the geometry stored in ``geoms[0]``.
    These correspond to Equation 8 in :cite:`pooladian:22`.

    .. note::
      When ``static_b=True``, the :math:`g` potential function
      will not be debiased.

    Args:
      epsilon: Epsilon regularization. If :obj:`None`, use in ``geoms[0]``.

    Returns:
      The debiased dual potential functions.
    """
    if self.is_low_rank:
      raise ValueError(
          "Dual potentials are not available, the divergence was computed "
          "with a low-rank solver."
      )
    geom_xy, *_ = self.geoms
    prob_xy = linear_problem.LinearProblem(geom_xy, a=self.a, b=self.b)
    (f_xy, g_xy), (f_x, _), (_, g_y) = self.potentials

    f_xy_fn = prob_xy.potential_fn_from_dual_vec(g_xy, epsilon=epsilon, axis=1)
    g_x_fn = prob_xy.potential_fn_from_dual_vec(f_x, epsilon=epsilon, axis=0)

    g_xy_fn = prob_xy.potential_fn_from_dual_vec(f_xy, epsilon=epsilon, axis=0)
    f_y_fn = None if g_y is None else prob_xy.potential_fn_from_dual_vec(
        g_y, epsilon=epsilon, axis=1
    )

    return potentials.DualPotentials(
        f=lambda x: f_xy_fn(x) - g_x_fn(x),
        g=g_xy_fn if f_y_fn is None else (lambda x: g_xy_fn(x) - f_y_fn(x)),
        cost_fn=geom_xy.cost_fn,
    )

  @property
  def is_low_rank(self) -> bool:
    """Whether the output is low-rank."""
    return self.factors is not None


def sinkdiv(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[jnp.ndarray, SinkhornDivergenceOutput]:
  """Wrapper to get the :term:`Sinkhorn divergence` between two point clouds.

  Convenience wrapper around
  :meth:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence` provided to
  compute the :term:`Sinkhorn divergence` between two point clouds compared with
  any ground cost :class:`~ott.geometry.costs.CostFn`. See other relevant
  arguments in :meth:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.

  Args:
    x: Array of input points, of shape `[num_x, feature]`.
    y: Array of target points, of shape `[num_y, feature]`.
    cost_fn: cost function of interest.
    epsilon: entropic regularization.
    kwargs: keywords arguments passed on to the generic
      :meth:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence` method. Of
      notable interest are ``a`` and ``b`` weight vectors, ``static_b`` and
      ``offset_static_b`` which can be used to bypass the computations of the
      transport problem between points stored in ``y`` (possibly with weights
      ``b``) and themselves, and ``solve_kwargs`` to parameterize the linear
      OT solver.

  Returns:
    The Sinkhorn divergence value, and output object detailing computations.
  """
  return sinkhorn_divergence(
      pointcloud.PointCloud,
      x=x,
      y=y,
      cost_fn=cost_fn,
      epsilon=epsilon,
      **kwargs
  )


def sinkhorn_divergence(
    geom: Type[geometry.Geometry],
    *args: Any,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    solve_kwargs: Mapping[str, Any] = MappingProxyType({}),
    static_b: bool = False,
    offset_static_b: Optional[float] = None,
    share_epsilon: bool = True,
    symmetric_sinkhorn: bool = True,
    **kwargs: Any,
) -> Tuple[jnp.ndarray, SinkhornDivergenceOutput]:
  r"""Compute :term:`Sinkhorn divergence` between two measures.

  The :term:`Sinkhorn divergence` is computed between two measures :math:`\mu`
  and :math:`\nu` by specifying three :class:`~ott.geometry.Geometry` objects,
  each describing pairwise costs within points in :math:`\mu,\nu`,
  :math:`\mu,\mu`, and :math:`\nu,\nu`.

  This implementation proposes the most general interface, to generate those
  three geometries by specifying first the type of
  :class:`~ott.geometry.Geometry` that is used to compare
  them, followed by the arguments used to generate these three
  :class:`~ott.geometry.Geometry` instances through its corresponding
  :meth:`~ott.geometry.geometry.Geometry.prepare_divergences` method.

  Args:
    geom: Type of the geometry.
    args: Positional arguments to
      :meth:`~ott.geometry.geometry.Geometry.prepare_divergences` that are
      specific to each geometry.
    a: the weight of each input point.
    b: the weight of each target point.
    solve_kwargs: keywords arguments for
      :func:`~ott.solvers.linear.solve` that is called either twice
      if ``static_b == True`` or three times when ``static_b == False``.
    static_b: if :obj:`True`, divergence of the second measure
      (with weights ``b``) to itself is **not** recomputed.
    offset_static_b: when ``static_b`` is :obj:`True`, use that value to offset
      computation. Useful when the value of the divergence of the second measure
      to itself is precomputed and not expected to change.
    share_epsilon: if True, enforces that the same epsilon regularizer is shared
      for all 2 or 3 terms of the Sinkhorn divergence. In that case, the epsilon
      will be by default that used when comparing x to y (contained in the first
      geometry). This flag is set to True by default, because in the default
      setting, the epsilon regularization is a function of the std of the
      entries in the cost matrix.
    symmetric_sinkhorn: Use Sinkhorn updates in Eq. 25 of :cite:`feydy:19` for
      symmetric terms comparing x/x and y/y.
    kwargs: keywords arguments to the generic class. This is specific to each
      geometry.

  Returns:
    The Sinkhorn divergence value, and output object detailing computations.
  """
  geom_xy, geom_x, *geom_y = geom.prepare_divergences(
      *args, static_b=static_b, **kwargs
  )
  geom_y = geom_y[0] if geom_y else None
  num_a, num_b = geom_xy.shape

  if share_epsilon:
    geom_x = geom_x.copy_epsilon(geom_xy)
    if geom_y is not None:
      geom_y = geom_y.copy_epsilon(geom_xy)

  dtype = geom_xy.dtype
  a = jnp.full((num_a,), 1.0 / num_a, dtype=dtype) if a is None else a
  b = jnp.full((num_b,), 1.0 / num_b, dtype=dtype) if b is None else b
  out = _sinkhorn_divergence(
      geom_xy,
      geom_x,
      geom_y,
      a=a,
      b=b,
      symmetric_sinkhorn=symmetric_sinkhorn,
      offset_yy=offset_static_b,
      **solve_kwargs
  )
  return out.divergence, out


def _sinkhorn_divergence(
    geometry_xy: geometry.Geometry,
    geometry_xx: geometry.Geometry,
    geometry_yy: Optional[geometry.Geometry],
    a: jnp.ndarray,
    b: jnp.ndarray,
    symmetric_sinkhorn: bool,
    offset_yy: Optional[float],
    **kwargs: Any,
) -> SinkhornDivergenceOutput:
  """Compute the (unbalanced) Sinkhorn divergence for the wrapper function.

  This definition includes a correction depending on the total masses of each
  measure, as defined in :cite:`sejourne:19`, eq. 15, and is extended to also
  accept :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn` solvers, as
  advocated in :cite:`scetbon:22b`.

  Args:
    geometry_xy: Geometry between the points of the two measures.
    geometry_xx: Geometry between the points of the first measure.
    geometry_yy: Geometry between the points of the second measure. If
      :obj:`None`, the regularized OT cost of the second measure onto itself is
      not computed and ``offset_yy`` is used instead.
    a: Weights of the first measure, of shape ``[n,]``.
    b: Weights of the second measure, of shape ``[m,]``.
    symmetric_sinkhorn: Use Sinkhorn updates in Eq. 25 of :cite:`feydy:19` for
      symmetric terms comparing x/x and y/y.
    offset_yy: Precomputed regularized OT cost of ``b`` onto itself, used when
      ``geometry_yy`` is :obj:`None`. If :obj:`None`, use :math:`0`.
    kwargs: Keyword arguments to :func:`~ott.solvers.linear.solve`.

  Returns:
    The output object.
  """
  kwargs_symmetric = kwargs.copy()
  is_low_rank = kwargs.get("rank", -1) > 0

  if symmetric_sinkhorn and not is_low_rank:
    # When computing a Sinkhorn divergence, the (x,y) terms and (x,x) / (y,y)
    # terms are computed independently. The user might want to pass some
    # kwargs to parameterize the solver's behavior, but those should
    # only apply to the (x,y) part.
    #
    # When using the Sinkhorn solver, for the (x,x) / (y,y) part, we fall back
    # on a simpler choice (parallel_dual_updates + momentum 0.5) that is known
    # to work well in such settings.
    #
    # Since symmetric terms are computed assuming a = b, the linear systems
    # arising in implicit differentiation (if used) of the potentials computed
    # for the symmetric parts should be marked as symmetric.
    kwargs_symmetric.update(
        parallel_dual_updates=True,
        momentum=acceleration.Momentum(start=0, value=0.5),
        anderson=None,
    )
    implicit_diff = kwargs.get("implicit_diff")
    if implicit_diff is not None:
      kwargs_symmetric["implicit_diff"] = implicit_diff.replace(symmetric=True)

  out_xy = linear.solve(geometry_xy, a=a, b=b, **kwargs)
  out_xx = linear.solve(geometry_xx, a=a, b=a, **kwargs_symmetric)
  if geometry_yy is None:
    # with `static_b`, the problem between `b` and itself is not solved
    out_yy = None
    cost_yy = 0.0 if offset_yy is None else offset_yy
  else:
    out_yy = linear.solve(geometry_yy, a=b, b=b, **kwargs_symmetric)
    cost_yy = out_yy.reg_ot_cost

  eps = jax.lax.stop_gradient(geometry_xy.epsilon)
  div = out_xy.reg_ot_cost - 0.5 * (out_xx.reg_ot_cost + cost_yy)
  # correction for the total masses, eq. 15 in :cite:`sejourne:19`
  div += 0.5 * eps * (jnp.sum(a) - jnp.sum(b)) ** 2

  solutions, errors, converged, n_iters = zip(
      _unpack(out_xy, is_low_rank),
      _unpack(out_xx, is_low_rank),
      _unpack(out_yy, is_low_rank),
  )

  return SinkhornDivergenceOutput(
      divergence=div,
      geoms=(geometry_xy, geometry_xx, geometry_yy),
      a=a,
      b=b,
      potentials=None if is_low_rank else solutions,
      factors=solutions if is_low_rank else None,
      errors=errors,
      converged=converged,
      n_iters=n_iters,
  )


def segment_sinkhorn_divergence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    num_segments: Optional[int] = None,
    max_measure_size: Optional[int] = None,
    cost_fn: Optional[costs.CostFn] = None,
    segment_ids_x: Optional[jnp.ndarray] = None,
    segment_ids_y: Optional[jnp.ndarray] = None,
    indices_are_sorted: bool = False,
    num_per_segment_x: Optional[Tuple[int, ...]] = None,
    num_per_segment_y: Optional[Tuple[int, ...]] = None,
    weights_x: Optional[jnp.ndarray] = None,
    weights_y: Optional[jnp.ndarray] = None,
    solve_kwargs: Mapping[str, Any] = MappingProxyType({}),
    static_b: bool = False,
    share_epsilon: bool = True,
    symmetric_sinkhorn: bool = True,
    **kwargs: Any
) -> jnp.ndarray:
  """Compute Sinkhorn divergence between subsets of vectors in `x` and `y`.

  Helper function designed to compute Sinkhorn divergences between several point
  clouds of varying size, in parallel, using padding for efficiency.
  In practice, The inputs `x` and `y` (and their weight vectors `weights_x` and
  `weights_y`) are assumed to be large weighted point clouds, that describe
  points taken from multiple measures. To extract several subsets of points, we
  provide two interfaces. The first interface assumes that a vector of id's is
  passed, describing for each point of `x` (resp. `y`) to which measure the
  point belongs to. The second interface assumes that `x` and `y` were simply
  formed by concatenating several measures contiguously, and that only indices
  that segment these groups are needed to recover them.

  For both interfaces, both `x` and `y` should contain the same total number of
  segments. Each segment will be padded as necessary, all segments rearranged as
  a tensor, and `vmap` used to evaluate Sinkhorn divergences in parallel.

  Args:
    x: Array of input points, of shape `[num_x, feature]`.
      Multiple segments are held in this single array.
    y: Array of target points, of shape `[num_y, feature]`.
    num_segments: Number of segments contained in `x` and `y`.
      Providing this is required for JIT compilation to work,
      see also :func:`~ott.geometry.segment.segment_point_cloud`.
    max_measure_size: Total size of measures after padding. Should ideally be
      set to an upper bound on points clouds processed with the segment
      interface. Should also be smaller than total length of `x` or `y`.
      Providing this is required for JIT compilation to work.
    cost_fn: Cost function,
      defaults to :class:`~ott.geometry.costs.SqEuclidean`.
    segment_ids_x: **1st interface** The segment ID for which each row of `x`
      belongs. This is a similar interface to :func:`jax.ops.segment_sum`.
    segment_ids_y: **1st interface** The segment ID for which each row of `y`
      belongs.
    indices_are_sorted: **1st interface** Whether `segment_ids_x` and
      `segment_ids_y` are sorted.
    num_per_segment_x: **2nd interface** Number of points in each segment in
      `x`. For example, [100, 20, 30] would imply that `x` is segmented into
      three arrays of length `[100]`, `[20]`, and `[30]` respectively.
    num_per_segment_y: **2nd interface** Number of points in each segment in
      `y`.
    weights_x: Weights of each input points, arranged in the same segmented
      order as `x`.
    weights_y: Weights of each input points, arranged in the same segmented
      order as `y`.
    solve_kwargs: Optionally a dict containing the keywords arguments for
      calls to the `sinkhorn` function, called three times to evaluate for each
      segment the Sinkhorn regularized OT cost between `x`/`y`, `x`/`x`, and
      `y`/`y` (except when `static_b` is `True`, in which case `y`/`y` is not
      evaluated)
    static_b: if True, divergence of measure b against itself is NOT computed
    share_epsilon: if True, enforces that the same epsilon regularizer is shared
      for all 2 or 3 terms of the Sinkhorn divergence. In that case, the epsilon
      will be by default that used when comparing x to y (contained in the first
      geometry). This flag is set to True by default, because in the default
      setting, the epsilon regularization is a function of the std of the cost
      matrix.
    symmetric_sinkhorn: Use Sinkhorn updates in Eq. 25 of :cite:`feydy:19` for
      symmetric terms comparing x/x and y/y.
    kwargs: keywords arguments passed to form
      :class:`~ott.geometry.pointcloud.PointCloud` geometry objects from the
      subsets of points and masses selected in `x` and `y`, this could be for
      instance entropy regularization float, scheduler or normalization.

  Returns:
    An array of Sinkhorn divergence values for each segment.
  """
  # instantiate padding vector
  dim = x.shape[1]
  if cost_fn is None:
    # default padder
    padding_vector = costs.CostFn._padder(dim=dim)
  else:
    padding_vector = cost_fn._padder(dim=dim)

  def eval_fn(
      padded_x: jnp.ndarray,
      padded_y: jnp.ndarray,
      padded_weight_x: jnp.ndarray,
      padded_weight_y: jnp.ndarray,
  ) -> float:
    div, _ = sinkhorn_divergence(
        pointcloud.PointCloud,
        padded_x,
        padded_y,
        a=padded_weight_x,
        b=padded_weight_y,
        solve_kwargs=solve_kwargs,
        static_b=static_b,
        share_epsilon=share_epsilon,
        symmetric_sinkhorn=symmetric_sinkhorn,
        cost_fn=cost_fn,
        **kwargs,
    )
    return div

  return segment._segment_interface(
      x,
      y,
      eval_fn,
      num_segments=num_segments,
      max_measure_size=max_measure_size,
      segment_ids_x=segment_ids_x,
      segment_ids_y=segment_ids_y,
      indices_are_sorted=indices_are_sorted,
      num_per_segment_x=num_per_segment_x,
      num_per_segment_y=num_per_segment_y,
      weights_x=weights_x,
      weights_y=weights_y,
      padding_vector=padding_vector
  )


def _unpack(
    out: Optional[Output], is_low_rank: bool
) -> Tuple[Tuple[Optional[jnp.ndarray], ...], Optional[jnp.ndarray], Any, Any]:
  """Get the solution, errors, convergence and number of iterations.

  ``out`` is :obj:`None` when the problem between ``b`` and itself is not
  solved, i.e., when ``static_b=True``.
  """
  if out is None:
    return (None,) * (3 if is_low_rank else 2), None, True, 0
  solution = (out.q, out.r, out.g) if is_low_rank else (out.f, out.g)
  return solution, out.errors, out.converged, out.n_iters
