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
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import costs, geometry, segment

__all__ = ["FreeBarycenterProblem", "FixedBarycenterProblem"]


@jax.tree_util.register_pytree_node_class
class FreeBarycenterProblem:
  """Free Wasserstein barycenter problem :cite:`cuturi:14`.

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
      **kwargs: Any,
  ):
    self._y = y
    if y.ndim == 3 and b is None:
      raise ValueError("Specify weights if `y` is already segmented.")
    self._b = b
    self._weights = weights
    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    self.epsilon = epsilon
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
      return jnp.ones((self.num_measures,)) / self.num_measures
    # Check that the number of measures coincides with the weights' size.
    assert self._weights.shape[0] == self.num_measures
    # By default, we assume that weights sum to 1, and enforce this if needed.
    return self._weights / jnp.sum(self._weights)

  @property
  def _is_segmented(self) -> bool:
    return self._y.ndim == 3

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return ([self._y, self._b, self._weights], {
        "cost_fn": self.cost_fn,
        "epsilon": self.epsilon,
        **self._kwargs,
    })

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "FreeBarycenterProblem":
    y, b, weights = children
    return cls(y=y, b=b, weights=weights, **aux_data)


@jax.tree_util.register_pytree_node_class
class FixedBarycenterProblem:
  """Fixed-support Wasserstein barycenter problem.

  Args:
    geom: Geometry object.
    a: batch of histograms of shape ``[batch, num_a]`` where ``num_a`` matches
      the first value of the :attr:`~ott.geometry.Geometry.shape` attribute of
      ``geom``.
    weights: ``[batch,]`` positive weights summing to :math:`1`. Uniform by
      default.
  """

  def __init__(
      self,
      geom: geometry.Geometry,
      a: jnp.ndarray,
      weights: Optional[jnp.ndarray] = None,
  ):
    self.geom = geom
    self.a = a
    self._weights = weights

  @property
  def num_measures(self) -> int:
    """Number of measures."""
    return self.a.shape[0]

  @property
  def weights(self) -> jnp.ndarray:
    """Barycenter weights of shape ``[num_measures,]`` that sum to :math`1`."""
    if self._weights is None:
      return jnp.ones((self.num_measures,)) / self.num_measures

    # check that the number of measures coincides with the weights' size
    assert self._weights.shape[0] == self.num_measures
    # by default, we assume that weights sum to 1, and enforce this if needed
    return self._weights / jnp.sum(self._weights)

  def tree_flatten(self):  # noqa: D102
    return [self.geom, self.a, self._weights], None

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "FixedBarycenterProblem":
    del aux_data
    geom, a, weights = children
    return cls(geom=geom, a=a, weights=weights)
