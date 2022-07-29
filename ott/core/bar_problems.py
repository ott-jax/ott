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
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import quad_problems
from ott.geometry import costs, geometry, pointcloud

__all__ = ["BarycenterProblem", "GWBarycenterProblem", "barycentric_projection"]


@jax.tree_util.register_pytree_node_class
class BarycenterProblem:
  """Wasserstein barycenter problem :cite:`cuturi:14`.

  Assumes ``y`` and ``b`` are already pre-segmented using
  :func:`~ott.core.segment.segment_point_cloud`.

  Args:
    y: Array of shape ``[num_measures, max_measure_size, ndim]`` containing
      the padded measures. See :func:`ott.core.segment.segment_point_cloud`
      for how to segment the data.
    b: Array of shape ``[num_measures, max_measure_size]`` containing
      all the weights of all the points within the measures that define
      the barycenter problem.
    weights: Array of shape ``[num_measures,]`` containing the weights of the
      barycenter problem.
    cost_fn: Cost function used. If ``None``, use
      :class:`~ott.geometry.costs.Euclidean`.
    epsilon: Epsilon regularization used to solve reg-OT problems.
    debiased: **Currently not implemented.**
      Whether the problem is debiased, in the sense that
      the regularized transportation cost of barycenter to itself will
      be considered when computing gradient. Note that if the debiased option
      is used, the barycenter size in
      :meth:`~ott.core.continuous_barycenter.WassersteinBarycenter.init_state`
      needs to be smaller than the maximum measure size for parallelization to
      operate efficiently.
  """

  def __init__(
      self,
      y: jnp.ndarray,
      b: jnp.ndarray,
      weights: Optional[jnp.ndarray] = None,
      cost_fn: Optional[costs.CostFn] = None,
      epsilon: Optional[jnp.ndarray] = None,
      debiased: bool = False,
  ):
    self._y = y
    self._b = b
    self._weights = weights
    self.cost_fn = costs.Euclidean() if cost_fn is None else cost_fn
    self.epsilon = epsilon
    self.debiased = debiased
    assert self._y.shape[:2] == self._b.shape

  @property
  def segmented_y_b(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Tuple of arrays containing the segmented measures and weights.

    Additional segment may be added when the problem is debiased.

    - Segmented measures of shape ``[num_measures, max_measure_size, ndim]``.
    - Segmented weights of shape ``[num_measures, max_measure_size]``.
    """
    if self.debiased:
      return self._add_slice_for_debiased()
    return self._y, self._b

  def _add_slice_for_debiased(
      self
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    y, b = self._y, self._b
    n = y.shape[1]  # (num_measures, max_measure_size, dim)
    # yapf: disable
    y = jnp.concatenate((y, jnp.zeros((1, n, self.ndim))), axis=0)
    b = jnp.concatenate((b, jnp.zeros((1, n))), axis=0)
    # yapf: enable
    return y, b

  @property
  def flattened_y(self) -> jnp.ndarray:
    """Array of shape ``[num_measures * (N_1 + N_2 + ...), ndim]``."""
    return self._y.reshape((-1, self._y.shape[-1]))

  @property
  def flattened_b(self) -> Optional[jnp.ndarray]:
    """Array of shape ``[num_measures * (N_1 + N_2 + ...),]``."""
    return self._b.ravel()

  @property
  def num_measures(self) -> int:
    """Number of measures."""
    return self._y.shape[0]

  @property
  def max_measure_size(self) -> int:
    """Maximum number of points across all measures."""
    return self._y.shape[1]

  @property
  def ndim(self) -> int:
    """Number of dimensions of the data."""
    return self._y.shape[2]

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

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return ([self._y, self._b, self._weights], {
        'cost_fn': self.cost_fn,
        'epsilon': self.epsilon,
        'debiased': self.debiased,
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
    y: Array of shape ``[num_measures, max_measure_size, ndim]`` containing
      the padded measures. See :func:`ott.core.segment.segment_point_cloud`
      for how to segment the data.
    b: Array of shape ``[num_measures, max_measure_size]`` containing
      all the weights of all the points within the measures that define
      the barycenter problem.
    weights: Array of shape ``[num_measures,]`` containing the weights of the
      barycenter problem.
    costs: Alternative to ``y``, an array of shape
      ``[num_measures, max_measure_size, max_measure_size]`` that defines padded
      cost matrices for each measure. Used in the quadratic term.
      Only one of ``y`` and ``cost`` can be specified.
    y_fused: Array of shape ``[num_measures, max_measure_size, ndim_fused]``
      containing the features of all points used to define the linear term
      in the fused case.
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

    super().__init__(y, b, weights=weights, **kwargs)

    self._y_fused = y_fused
    self.fused_penalty = fused_penalty
    self.gw_loss, self._loss_name = self._create_loss(gw_loss), gw_loss
    self.scale_cost = scale_cost
    self._y_as_costs = costs is not None

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

    @partial(jax.vmap, in_axes=[0, 0, None])
    def project(
        y: jnp.ndarray,
        transport: jnp.ndarray,
        fn: Optional[quad_problems.Loss],
    ) -> jnp.ndarray:
      if self._y_as_costs:
        assert y.shape[0] == y.shape[1], y.shape
        geom = geometry.Geometry(
            y, epsilon=self.epsilon, scale_cost=self.scale_cost
        )
      else:
        geom = pointcloud.PointCloud(
            y,
            cost_fn=self.cost_fn,
            epsilon=self.epsilon,
            scale_cost=self.scale_cost
        )
      fn, lin = (None, True) if fn is None else (fn, fn.is_linear)
      tmp = geom.apply_cost(
          transport.T,
          axis=0,
          fn=fn,
          is_linear=lin,
      )
      return transport @ tmp

    fn = None if self._loss_name == 'sqeucl' else self.gw_loss.h2
    # TODO(michalk8): handle mask?
    y, _ = self.segmented_y_b
    weights = self.weights[:, None, None]

    barycenter = jnp.sum(weights * project(y, transports, fn), axis=0)
    # TODO(michalk8): more efficient impl.
    barycenter /= jnp.outer(a, a)

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
    y_fused = self.segmented_y_fused
    if y_fused is None:
      raise RuntimeError(
          "Feature updates are available only in the fused case."
      )

    weights = self.weights[:, None, None]
    divide_a = jnp.where(a > 0, 1.0 / a, 1.0)
    transports = transports * divide_a[None, :, None]

    if self._loss_name == "sqeucl":
      cost = costs.Euclidean()
      return jnp.sum(
          weights * barycentric_projection(transports, y_fused, cost), axis=0
      )
    raise NotImplementedError(self._loss_name)

  @property
  def is_fused(self) -> bool:
    """Whether the problem is fused."""
    return self._y_fused is not None

  @property
  def segmented_y_fused(self) -> Optional[jnp.ndarray]:
    """Feature array of shape ``[num_measures, max_measure_size, ndim_fused]`` \
    used in the in the fused case."""
    return self._y_fused

  @property
  def ndim_fused(self) -> Optional[int]:
    """Number of dimensions of the fused term."""
    y_fused = self.segmented_y_fused
    return None if y_fused is None else y_fused.shape[2]

  @staticmethod
  def _create_loss(loss: Literal['sqeucl', 'kl']) -> quad_problems.GWLoss:
    if loss == 'sqeucl':
      return quad_problems.make_square_loss()
    if loss == 'kl':
      return quad_problems.make_kl_loss()
    raise NotImplementedError(f"Loss `{loss}` is not yet implemented.")

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    (y, b, weights), aux = super().tree_flatten()
    if self._y_as_costs:
      children = (None, b, weights, y)
    else:
      children = (y, b, weights, None)
    children += (self.segmented_y_fused,)
    aux['fused_penalty'] = self.fused_penalty
    aux['gw_loss'] = self._loss_name
    aux['scale_cost'] = self.scale_cost
    return children, aux

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
