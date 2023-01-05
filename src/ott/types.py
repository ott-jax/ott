from typing_extensions import Protocol

import jax.numpy as jnp

# TODO(michalk8): introduce additional types here


class Transport(Protocol):
  """Interface for the solution of a transport problem.

  Classes implementing those function do not have to inherit from it, the
  class can however be used in type hints to support duck typing.
  """

  @property
  def matrix(self) -> jnp.ndarray:
    ...

  def apply(self, inputs: jnp.ndarray, axis: int) -> jnp.ndarray:
    ...

  def marginal(self, axis: int = 0) -> jnp.ndarray:
    ...
