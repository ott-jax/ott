from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import geometry


@jax.tree_util.register_pytree_node_class
class GromovWassersteinBarycenterProblem:

  def __init__(
      self,
      geometries: Sequence[geometry.Geometry],
      b: Optional[Sequence[jnp.ndarray]] = None,
      weights: Optional[jnp.ndarray] = None,
  ):
    self.geometries = geometries
    self.b = b
    self._weights = weights

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

  def tree_flatten(self) -> Tuple[Sequence[Any], Mapping[str, Any]]:
    return [self.geometries, self._weights], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Mapping[str, Any], children: Sequence[Any]
  ) -> "GromovWassersteinBarycenterProblem":
    return cls(*children, **aux_data)
