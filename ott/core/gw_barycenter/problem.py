from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import bar_problems, segment


@jax.tree_util.register_pytree_node_class
class GWBarycenterProblem(bar_problems.BarycenterProblem):

  def __init__(
      self,
      *args: Any,
      y_fused: Optional[jnp.ndarray] = None,
      fused_penalty: float = 1.0,
      loss: Literal['sqeucl', 'kl'] = 'sqeucl',
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
      kwargs: Keyword arguments for
        :class:`ott.core.bar_problems.BarycenterProblem`.
    """
    super().__init__(*args, **kwargs)
    self._loss_name = loss
    self.loss = self._create_loss(loss)
    self._y_fused = y_fused
    self.fused_penalty = fused_penalty

  @staticmethod
  def _create_loss(loss: Literal['sqeucl', 'kl']):
    from ott.core.quad_problems import make_kl_loss, make_square_loss

    # TODO(michalk8): consider refactoring as a quad. loss class
    if loss == 'sqeucl':
      return make_square_loss()
    if loss == 'kl':
      return make_kl_loss()
    raise NotImplementedError(f"Loss `{loss}` is not yet implemented.")

  @property
  def is_fused(self) -> bool:
    """Whether this problem is fused."""
    return self._y_fused is not None

  @property
  def segmented_y_fused(self) -> Optional[jnp.ndarray]:
    return self._y_fused

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux = super().tree_flatten()
    aux['loss'] = self._loss_name
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
