from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from typing_extensions import Literal


@jax.tree_util.register_pytree_node_class
class GromovWassersteinBarycenterProblem:

  def __init__(
      self,
      geometries: jnp.ndarray,
      # TODO(michalk8): consider providing endpoint
      #  for already computed geometries
      b: Optional[Sequence[jnp.ndarray]] = None,
      weights: Optional[jnp.ndarray] = None,
      loss: Literal['sqeucl', 'kl'] = 'sqeucl',
      epsilon: float = 1.0,
      **kwargs: Any,
  ):
    """TODO.

    Args:
      geometries: (num_measures, n_points, n_dim)
      b: (num_measures, n_points)
      weights: (num_measures,)
      loss: TODO.
      epsilon: TODO.
      kwargs: Keyword arguments for :class:`ott.geometry.pointcloud.PointCloud`.
    """
    self.geometries = geometries
    self.b = b
    self._weights = weights
    self._kwargs = kwargs
    self.loss, self._loss = self._create_loss(loss), loss
    self.epsilon = epsilon

  @staticmethod
  def _create_loss(loss: Literal['squecl', 'kl']):
    from ott.core.quad_problems import make_kl_loss, make_square_loss

    if loss == 'sqeucl':
      return make_square_loss()
    if loss == 'kl':
      return make_kl_loss()
    raise NotImplementedError(loss)

  @property
  def size(self) -> int:
    """Number of measures."""
    return 0 if self.geometries is None else len(self.geometries)

  @property
  def weights(self) -> jnp.ndarray:
    """Weights for each measure."""
    weights = self._weights
    if weights is None:
      weights = jnp.ones((len(self.geometries),)) / len(self.geometries)
    assert weights.shape[0] == len(self.geometries)
    return weights

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self.geometries, self.b, self._weights], {
        '_kwargs': self._kwargs,
        'loss': self._loss,
        'epsilon': self.epsilon,
    }

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "GromovWassersteinBarycenterProblem":
    kwargs = aux_data.pop("_kwargs")
    return cls(*children, **aux_data, **kwargs)
