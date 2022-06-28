from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import bar_problems, continuous_barycenter, quad_problems, segment
from ott.geometry import costs, pointcloud


@jax.tree_util.register_pytree_node_class
class GWBarycenterProblem(bar_problems.BarycenterProblem):

  def __init__(
      self,
      *args: Any,
      y_fused: Optional[jnp.ndarray] = None,
      fused_penalty: float = 1.0,
      loss: Literal['sqeucl', 'kl'] = 'sqeucl',
      scale_cost: Optional[Literal["TODO"]] = None,
      is_cost: bool = False,
      **kwargs: Any,
  ):
    """TODO.

    Args:
      args: Positional arguments for
        :class:`ott.core.bar_problems.BarycenterProblem`.
      y_fused: TODO.
      loss: TODO.
      fused_penalty: TODO.
        Only used when ``y_fused != None``.
      scale_cost: TODO.
      is_cost: Whether ``y`` represents a cost matrix or a point cloud.
      kwargs: Keyword arguments for
        :class:`ott.core.bar_problems.BarycenterProblem`.
    """
    super().__init__(*args, **kwargs)
    self._y_fused = y_fused
    self.fused_penalty = fused_penalty
    self.loss = self._create_loss(loss)
    self._loss_name = loss
    self.scale_cost = scale_cost
    self.is_cost = is_cost

  def update_barycenter(
      self, transports: jnp.ndarray, a: jnp.ndarray
  ) -> jnp.ndarray:
    """TODO.

    Args:
      transports: (num_measures, TODO, TODO)
      a: barycenter weights.

    Returns:
      TODO.
    """

    @partial(jax.vmap, in_axes=[0, 0, None])
    def update(
        y: jnp.ndarray, transport: jnp.ndarray, fn: Callable[[jnp.ndarray],
                                                             jnp.ndarray]
    ) -> jnp.ndarray:
      geom = pointcloud.PointCloud(y, epsilon=self.epsilon)
      tmp = geom.apply_cost(transport.T, axis=0, fn=fn)
      return transport @ tmp

    fn = None if self._loss_name == 'sqeucl' else self.loss[1][1]
    y, _ = self.segmented_y_b
    weights = self.weights[:, None, None]

    barycenter = jnp.sum(weights * update(y, transports, fn), axis=0)
    barycenter *= 1. / jnp.vdot(a, a)

    if self._loss_name == 'kl':
      barycenter = jnp.exp(barycenter)
    return barycenter

  def update_features(self, transports: jnp.ndarray,
                      a: jnp.ndarray) -> Optional[jnp.ndarray]:
    if not self.is_fused:
      return None

    y_fused = self.segmented_y_fused
    weights = self.weights[:, None, None]

    # TODO(michalk8): verify
    if self._loss_name == "sqeucl":
      cost = costs.Euclidean()
      divide_a = jnp.where(a > 0, 1.0 / a, 1.0)
      transports = transports * divide_a[None, :, None]
      return jnp.sum(
          weights * continuous_barycenter
          .barycentric_projection(transports, y_fused, cost),
          axis=0
      )
    raise NotImplementedError(self._loss_name)

  @property
  def is_fused(self) -> bool:
    """Whether this problem is fused."""
    return self._y_fused is not None

  @property
  def segmented_y_fused(self) -> Optional[jnp.ndarray]:
    if self._y_fused is None or self._y_fused.ndim == 3:
      return self._y_fused
    segmented_y_fused, _, _ = segment.segment_point_cloud(
        self._y_fused, None, self._segment_ids, self._num_segments,
        self._indices_are_sorted, self._num_per_segment, self.max_measure_size
    )
    return segmented_y_fused

  @staticmethod
  def _create_loss(loss: Literal['sqeucl', 'kl']):
    # TODO(michalk8): consider refactoring as a quad. loss class
    if loss == 'sqeucl':
      return quad_problems.make_square_loss()
    if loss == 'kl':
      return quad_problems.make_kl_loss()
    raise NotImplementedError(f"Loss `{loss}` is not yet implemented.")

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux = super().tree_flatten()
    aux["y_fused"] = self._y_fused
    aux['fused_penalty'] = self.fused_penalty
    aux['loss'] = self._loss_name
    aux['scale_cost'] = self.scale_cost
    aux['is_cost'] = self.is_cost
    return children, aux


def segment_cost_matrix(
    costs: Sequence[jnp.ndarray],
    axis: int = 1,
    **kwargs: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  num_per_segment = jnp.asarray([c.shape[axis] for c in costs])
  fcs, fb, _ = segment.segment_point_cloud(
      jnp.concatenate(costs, axis=axis).T,
      num_per_segment=num_per_segment,
      num_segments=len(costs),
      **kwargs,
  )
  if axis == 1:
    fcs = jnp.swapaxes(fcs, 1, 2)
  return fcs, fb
