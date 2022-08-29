# Copyright 2022 Apple Inc
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
"""Classes defining OT problem(s) (objective function + utilities)."""
import functools
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import quad_problems, segment
from ott.geometry import costs, geometry, pointcloud

__all__ = ["BarycenterProblem", "GWBarycenterProblem", "barycentric_projection"]


@jax.tree_util.register_pytree_node_class
class BarycenterProblem:
  """Wasserstein barycenter problem :cite:`cuturi:14`.

  Args:
    y: Array of shape ``[num_total_points, ndim]`` merging the points of all
      measures. Alternatively, already segmented array of shape
      ``[num_measures, max_measure_size, ndim]`` can be passed.
      See also :func:`~ott.core.segment.segment_point_cloud`.
    b: Array of shape ``[num_total_points,]`` containing the weights of all
      the points within the measures that define the barycenter problem.
      Similarly as ``y``, segmented array of weights of shape
      ``[num_measures, max_measure_size]`` can be passed.
      If ``y`` is already pre-segmented, this array must be always specified.
    weights: Array of shape ``[num_measures,]`` containing the weights of the
      measures.
    cost_fn: Cost function used. If `None`,
      use :class:`~ott.geometry.costs.Euclidean` cost.
    epsilon: Epsilon regularization used to solve reg-OT problems.
    debiased: **Currently not implemented.**
      Whether the problem is debiased, in the sense that
      the regularized transportation cost of barycenter to itself will
      be considered when computing gradient. Note that if the debiased option
      is used, the barycenter size in
      :meth:`~ott.core.continuous_barycenter.WassersteinBarycenter.init_state`
      needs to be smaller than the maximum measure size for parallelization to
      operate efficiently.
    kwargs: Keyword arguments :func:`~ott.core.segment.segment_point_cloud`.
      Only used when ``y`` is not already segmented. When passing
      ``segment_ids``, 2 arguments must be specified for jitting to work:

        - ``num_segments`` - the total number of measures.
        - ``max_measure_size`` -  maximum of support sizes of these measures.
  """

  def __init__(
      self,
      y: jnp.ndarray,
      b: Optional[jnp.ndarray] = None,
      weights: Optional[jnp.ndarray] = None,
      cost_fn: Optional[costs.CostFn] = None,
      epsilon: Optional[float] = None,
      debiased: bool = False,
      **kwargs: Any,
  ):
    self._y = y
    if y.ndim == 3 and b is None:
      raise ValueError("Specify weights if `y` is already segmented.")
    self._b = b
    self._weights = weights
    self.cost_fn = costs.Euclidean() if cost_fn is None else cost_fn
    self.epsilon = epsilon
    self.debiased = debiased
    self._kwargs = kwargs

    if self._is_segmented:
      # (num_measures, max_measure_size, ndim)
      # (num_measures, max_measure_size)
      assert self._y.shape[:2] == self._b.shape
    else:
      # (num_total_points, ndim)
      # (num_total_points,)
      assert self._b is None or self._y.shape[0] == self._b.shape[0]

  @property
  def segmented_y_b(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Tuple of arrays containing the segmented measures and weights.

    Additional segment may be added when the problem is debiased.

      - Segmented measures of shape ``[num_measures, max_measure_size, ndim]``.
      - Segmented weights of shape ``[num_measures, max_measure_size]``.
    """
    if self._is_segmented:
      y, b = self._y, self._b
    else:
      y, b = segment.segment_point_cloud(
          x=self._y,
          a=self._b,
          padding_vector=self.cost_fn.padder(self.ndim),
          **self._kwargs
      )

    if self.debiased:
      return self._add_slice_for_debiased(y, b)
    return y, b

  def _add_slice_for_debiased(
      self, y: jnp.ndarray, b: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    y, b = self._y, self._b
    _, n, ndim = y.shape  # (num_measures, max_measure_size, ndim)
    # yapf: disable
    y = jnp.concatenate((y, jnp.zeros((1, n, ndim))), axis=0)
    b = jnp.concatenate((b, jnp.zeros((1, n))), axis=0)
    # yapf: enable
    return y, b

  @property
  def flattened_y(self) -> jnp.ndarray:
    """Array of shape ``[num_measures * (N_1 + N_2 + ...), ndim]``."""
    if self._is_segmented:
      return self._y.reshape((-1, self._y.shape[-1]))
    return self._y

  @property
  def flattened_b(self) -> Optional[jnp.ndarray]:
    """Array of shape ``[num_measures * (N_1 + N_2 + ...),]``."""
    return None if self._b is None else self._b.ravel()

  @property
  def num_measures(self) -> int:
    """Number of measures."""
    return self.segmented_y_b[0].shape[0]

  @property
  def max_measure_size(self) -> int:
    """Maximum number of points across all measures."""
    return self.segmented_y_b[0].shape[1]

  @property
  def ndim(self) -> int:
    """Number of dimensions of ``y``."""
    return self._y.shape[-1]

  @property
  def weights(self) -> jnp.ndarray:
    """Barycenter weights of shape ``[num_measures,]`` that sum to 1."""
    if self._weights is None:
      weights = jnp.ones((self.num_measures,)) / self.num_measures
    else:
      # Check that the number of measures coincides with the weights' size.
      assert self._weights.shape[0] == self.num_measures
      # By default, we assume that weights sum to 1, and enforce this if needed.
      weights = self._weights / jnp.sum(self._weights)
    if self.debiased:
      weights = jnp.concatenate((weights, jnp.array([-0.5])))
    return weights

  @property
  def _is_segmented(self) -> bool:
    return self._y.ndim == 3

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return ([self._y, self._b, self._weights], {
        'cost_fn': self.cost_fn,
        'epsilon': self.epsilon,
        'debiased': self.debiased,
        **self._kwargs,
    })

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "BarycenterProblem":
    y, b, weights = children
    return cls(y=y, b=b, weights=weights, **aux_data)


@jax.tree_util.register_pytree_node_class
class GWBarycenterProblem(BarycenterProblem):
  """(Fused) Gromov-Wasserstein barycenter problem :cite:`peyre:16,vayer:19`.

  Args:
    y: Array of shape ``[num_total_points, ndim]`` merging the points of all
      measures. Alternatively, already segmented array of shape
      ``[num_measures, max_measure_size, ndim]`` can be passed.
      See also :func:`~ott.core.segment.segment_point_cloud`.
    b: Array of shape ``[num_total_points,]`` containing the weights of all
      the points within the measures that define the barycenter problem.
      Similarly as ``y``, segmented array of weights of shape
      ``[num_measures, max_measure_size]`` can be passed.
      If ``y`` is already pre-segmented, this array must be passed.
    weights: Array of shape ``[num_measures,]`` containing the weights of the
      barycenter problem.
    costs: Alternative to ``y``, an array of shape
      ``[num_measures, max_measure_size, max_measure_size]`` that defines padded
      cost matrices for each measure. Used in the quadratic term.
      Only one of ``y`` and ``cost`` can be specified.
    y_fused: Array of shape ``[num_total_points, ndim_fused]`` containing
      the data of the points of all measures used to define the linear term
      in the fused case. Similarly as ``y``, can be specified as a pre-segmented
      array of shape ``[num_measures, max_measure_size, ndim_fused]``.
    gw_loss: Gromov-Wasserstein loss.
    fused_penalty: Multiplier of the linear term. Only used when
      ``y_fused != None``.
    scale_cost: Scaling of cost matrices passed to geometries.
    kwargs: Keyword arguments for
      :class:`~ott.core.bar_problems.BarycenterProblem`.
  """

  def __init__(
      self,
      y: Optional[jnp.ndarray] = None,
      b: Optional[jnp.ndarray] = None,
      weights: Optional[jnp.ndarray] = None,
      costs: Optional[jnp.ndarray] = None,
      y_fused: Optional[jnp.ndarray] = None,
      fused_penalty: float = 1.0,
      gw_loss: Literal['sqeucl', 'kl'] = 'sqeucl',
      scale_cost: Union[int, float, Literal["mean", "max_cost"]] = 1.0,
      **kwargs: Any,
  ):
    assert y is None or costs is None, "Cannot specify both `y` and `costs`."
    y = y if costs is None else costs

    super().__init__(y=y, b=b, weights=weights, **kwargs)

    self._y_fused = y_fused
    self.fused_penalty = fused_penalty
    self._loss_name = gw_loss
    self.scale_cost = scale_cost
    self._y_as_costs = costs is not None

    if self._y_as_costs:
      # (num_measures, max_measure_size, max_measure_size)
      _, n, m = self._y.shape
      assert n == m, "Cost matrices must be square."
    if self.is_fused:
      seg_y = self._is_segmented
      seg_fused = self._y_fused.ndim == 3
      if seg_y and seg_fused:
        # (num_measures, max_measure_size, ndim_fused)
        # (num_measures, max_measure_size, ndim)
        assert self._y_fused.shape[:2] == self._y.shape[:2]
      if not seg_y and not seg_fused:
        # (num_total_points, ndim_fused), (num_total_points, ndim)
        assert self._y_fused.shape[0] == self._y.shape[0]
      # TODO(michalk8): in the future, consider checking the other 2 cases
      # using `segmented_y` and `segmented_y_fused`?

  def update_barycenter(
      self, transports: jnp.ndarray, a: jnp.ndarray
  ) -> jnp.ndarray:
    """Update the barycenter cost matrix.

    Uses the eq. 14 and 15 of :cite:`peyre:16`.

    Args:
      transports: Transport maps of shape
        ``[num_measures, bar_size, max_measure_size]``.
      a: Barycenter weights of shape ``[bar_size,]``.

    Returns:
      Update cost matrix of shape ``[bar_size, bar_size]``.
    """

    @functools.partial(jax.vmap, in_axes=[0, 0, 0, None])
    def project(
        y: jnp.ndarray,
        b: jnp.ndarray,
        transport: jnp.ndarray,
        fn: Optional[quad_problems.Loss],
    ) -> jnp.ndarray:
      geom = self._create_y_geometry(y, mask=b > 0.)
      fn, lin = (None, True) if fn is None else (fn.func, fn.is_linear)

      tmp = geom.apply_cost(
          transport.T,
          axis=0,
          fn=fn,
          is_linear=lin,
      )
      return transport @ tmp

    fn = None if self._loss_name == 'sqeucl' else self.gw_loss.h2
    y, b = self.segmented_y_b
    weights = self.weights[:, None, None]

    barycenter = jnp.sum(weights * project(y, b, transports, fn), axis=0)
    inv_a = jnp.where(a > 0, 1.0 / a, 1.0)
    barycenter = (barycenter * inv_a[None, :]) * inv_a[:, None]

    # TODO(michalk8): in future, use `isinstanceof(self.gw_loss, ...)`
    # once refactoring has been done
    if self._loss_name == 'kl':
      barycenter = jnp.exp(barycenter)
    return barycenter

  def update_features(self, transports: jnp.ndarray,
                      a: jnp.ndarray) -> Optional[jnp.ndarray]:
    """Update the barycenter features in the fused case :cite:`vayer:19`.

    Uses :cite:`cuturi:14` eq. 8, and is implemented only
    for the squared :class:`~ott.geometry.costs.Euclidean` cost.

    Args:
      transports: Transport maps of shape
        ``[num_measures, bar_size, max_measure_size]``.
      a: Barycenter weights of shape ``[bar_size,]``.

    Returns:
      Updated features of shape ``[bar_size, ndim_fused]``.
    """
    if not self.is_fused:
      raise RuntimeError(
          "Updating features is available only in the fused case."
      )

    y_fused = self.segmented_y_fused
    weights = self.weights[:, None, None]
    inv_a = jnp.where(a > 0, 1.0 / a, 1.0)
    transports = transports * inv_a[None, :, None]

    if self._loss_name == "sqeucl":
      cost = costs.Euclidean()
      return jnp.sum(
          weights * barycentric_projection(transports, y_fused, cost), axis=0
      )
    raise NotImplementedError(self._loss_name)

  def _create_bary_geometry(
      self,
      cost_matrix: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None
  ) -> geometry.Geometry:
    return geometry.Geometry(
        cost_matrix=cost_matrix,
        src_mask=mask,
        tgt_mask=mask,
        epsilon=self.epsilon,
        scale_cost=self.scale_cost
    )

  def _create_y_geometry(
      self,
      y: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None
  ) -> geometry.Geometry:
    if self._y_as_costs:
      assert y.shape[0] == y.shape[1], y.shape
      return geometry.Geometry(
          y,
          epsilon=self.epsilon,
          scale_cost=self.scale_cost,
          src_mask=mask,
          tgt_mask=mask
      )
    return pointcloud.PointCloud(
        y,
        epsilon=self.epsilon,
        scale_cost=self.scale_cost,
        cost_fn=self.cost_fn,
        src_mask=mask,
        tgt_mask=mask
    )

  def _create_fused_geometry(
      self,
      x: jnp.ndarray,
      y: jnp.ndarray,
      src_mask: Optional[jnp.ndarray] = None,
      tgt_mask: Optional[jnp.ndarray] = None
  ) -> pointcloud.PointCloud:
    return pointcloud.PointCloud(
        x,
        y,
        cost_fn=self.cost_fn,
        epsilon=self.epsilon,
        scale_cost=self.scale_cost,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

  def _create_problem(
      self,
      state: 'GWBarycenterState',  # noqa: F821
      y: jnp.ndarray,
      b: jnp.ndarray,
      f: Optional[jnp.ndarray] = None
  ) -> quad_problems.QuadraticProblem:
    # TODO(michalk8): in the future, mask in the problem for convenience?
    bary_mask = state.a > 0.
    y_mask = b > 0.

    geom_xx = self._create_bary_geometry(state.cost, mask=bary_mask)
    geom_yy = self._create_y_geometry(y, mask=y_mask)
    if self.is_fused:
      assert f is not None
      assert state.x.shape[1] == f.shape[1]
      geom_xy = self._create_fused_geometry(
          state.x, f, src_mask=bary_mask, tgt_mask=y_mask
      )
    else:
      geom_xy = None

    return quad_problems.QuadraticProblem(
        geom_xx=geom_xx,
        geom_yy=geom_yy,
        geom_xy=geom_xy,
        a=state.a,
        b=b,
        fused_penalty=self.fused_penalty,
    )

  @property
  def is_fused(self) -> bool:
    """Whether the problem is fused."""
    return self._y_fused is not None

  @property
  def segmented_y_fused(self) -> Optional[jnp.ndarray]:
    """Feature array of shape ``[num_measures, max_measure_size, ndim_fused]`` \
    used in the fused case."""
    if not self.is_fused or self._y_fused.ndim == 3:
      return self._y_fused
    y_fused, _ = segment.segment_point_cloud(
        x=self._y_fused,
        padding_vector=self.cost_fn.padder(self.ndim_fused),
        **self._kwargs
    )
    return y_fused

  @property
  def ndim(self) -> Optional[int]:
    return None if self._y_as_costs else self._y.shape[-1]

  @property
  def ndim_fused(self) -> Optional[int]:
    """Number of dimensions of the fused term."""
    return self._y_fused.shape[-1] if self.is_fused else None

  @property
  def gw_loss(self) -> quad_problems.GWLoss:
    """Gromov-Wasserstein loss."""
    # TODO(michalk8): custom losses would require inverting some fns;
    # `https://jax.readthedocs.io/en/latest/notebooks/ some fns;
    # Writing_custom_interpreters_in_Jax.html#your-first-interpreter-invert`
    # might be useful
    if self._loss_name == 'sqeucl':
      return quad_problems.make_square_loss()
    if self._loss_name == 'kl':
      return quad_problems.make_kl_loss()
    raise NotImplementedError(
        f"Loss `{self._loss_name}` is not yet implemented."
    )

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    (y, b, weights), aux = super().tree_flatten()
    if self._y_as_costs:
      children = [None, b, weights, y]
    else:
      children = [y, b, weights, None]
    aux['fused_penalty'] = self.fused_penalty
    aux['gw_loss'] = self._loss_name
    aux['scale_cost'] = self.scale_cost
    return children + [self._y_fused], aux

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "GWBarycenterProblem":
    y, b, weights, costs, y_fused = children
    return cls(
        y=y, b=b, weights=weights, costs=costs, y_fused=y_fused, **aux_data
    )


@functools.partial(jax.vmap, in_axes=[0, 0, None])
def barycentric_projection(
    matrix: jnp.ndarray, y: jnp.ndarray, cost_fn
) -> jnp.ndarray:
  return jax.vmap(cost_fn.barycenter, in_axes=[0, None])(matrix, y)
