from typing import Any, Dict, Sequence, Tuple

import jax
from typing_extensions import Literal

from ott.core import bar_problems


@jax.tree_util.register_pytree_node_class
class GWBarycenterProblem(bar_problems.BarycenterProblem):

  def __init__(
      self,
      *args: Any,
      loss: Literal['sqeucl', 'kl'] = 'sqeucl',
      **kwargs: Any,
  ):
    """TODO.

    Args:
      args: Positiona arguments for
        :class:`ott.core.bar_problems.BarycenterProblem`.
      loss: TODO.
      kwargs: Keyword arguments for
        :class:`ott.core.bar_problems.BarycenterProblem`.
    """
    super().__init__(*args, **kwargs)
    self.loss = self._create_loss(loss)
    # TODO(michalk8): better name
    self._loss = loss

  @staticmethod
  def _create_loss(loss: Literal['sqeucl', 'kl']):
    from ott.core.quad_problems import make_kl_loss, make_square_loss

    # TODO(michalk8): consider refactoring as a quad. loss class
    if loss == 'sqeucl':
      return make_square_loss()
    if loss == 'kl':
      return make_kl_loss()
    raise NotImplementedError(loss)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux = super().tree_flatten()
    aux['loss'] = self._loss
    return children, aux
