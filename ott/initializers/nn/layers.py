from flax import linen as nn
from jax import numpy as jnp

__all__ = ["MetaMLP"]


class MetaMLP(nn.Module):
  r"""A Meta MLP potential for :class:`~ott.core.initializers.MetaInitializer`.

  This provides an MLP :math:`\hat f_\theta(a, b)` that maps from the
  probabilities of the measures to the optimal dual potentials :math:`f`.

  Args:
    potential_size: The dimensionality of :math:`f`.
    num_hidden_units: The number of hidden units in each layer.
    num_hidden_layers: The number of hidden layers.
  """

  potential_size: int
  num_hidden_units: int = 512
  num_hidden_layers: int = 3

  @nn.compact
  def __call__(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    r"""Make a prediction.

    Args:
      a: Probabilities of the :math:`\alpha` measure's atoms.
      b: Probabilities of the :math:`\beta` measure's atoms.

    Returns:
      The :math:`f` potential.
    """
    dtype = a.dtype
    z = jnp.concatenate((a, b))
    for _ in range(self.num_hidden_layers):
      z = nn.relu(nn.Dense(self.num_hidden_units, dtype=dtype)(z))
    f = nn.Dense(self.potential_size, dtype=dtype)(z)
    return f
