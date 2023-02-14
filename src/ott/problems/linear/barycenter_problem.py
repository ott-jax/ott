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
"""Classes defining OT problem(s) (objective function + utilities)."""
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import costs, segment

__all__ = ["BarycenterProblem"]


@jax.tree_util.register_pytree_node_class
class BarycenterProblem:
  """Wasserstein barycenter problem :cite:`cuturi:14`.

  Args:
    y: Array of shape ``[num_total_points, ndim]`` merging the points of all
      measures. Alternatively, already segmented array of shape
      ``[num_measures, max_measure_size, ndim]`` can be passed.
      See also :func:`~ott.geometry.segment.segment_point_cloud`.
    b: Array of shape ``[num_total_points,]`` containing the weights of all
      the points within the measures that define the barycenter problem.
      Same as ``y``, pre-segmented array of weights of shape
      ``[num_measures, max_measure_size]`` can be passed.
      If ``y`` is already pre-segmented, this array must be always specified.
    weights: Array of shape ``[num_measures,]`` containing the weights of the
      measures.
    cost_fn: Cost function used. If `None`,
      use the :class:`~ott.geometry.costs.SqEuclidean` cost.
    epsilon: Epsilon regularization used to solve reg-OT problems.
    debiased: **Currently not implemented.**
      Whether the problem is debiased, in the sense that
      the regularized transportation cost of barycenter to itself will
      be considered when computing gradient. Note that if the debiased option
      is used, the barycenter size in
      :meth:`~ott.solvers.linear.continuous_barycenter.WassersteinBarycenter.init_state`
      needs to be smaller than the maximum measure size for parallelization to
      operate efficiently.
    kwargs: Keyword arguments :func:`~ott.geometry.segment.segment_point_cloud`.
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
    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
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
          padding_vector=self.cost_fn._padder(self.ndim),
          **self._kwargs
      )

    if self.debiased:
      return self._add_slice_for_debiased(y, b)
    return y, b

  @staticmethod
  def _add_slice_for_debiased(
      y: jnp.ndarray, b: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
