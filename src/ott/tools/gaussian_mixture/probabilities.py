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
from typing import Optional

import jax
import jax.numpy as jnp

__all__ = ["Probabilities"]


@jax.tree_util.register_pytree_node_class
class Probabilities:
  """Parameterized array of probabilities of length n.

  The internal representation is a length n-1 unconstrained array. We convert
  to a length n simplex by appending a 0 and taking a softmax.
  """

  _params: jnp.ndarray

  def __init__(self, params):
    self._params = params

  @classmethod
  def from_random(
      cls,
      rng: jax.Array,
      n_dimensions: int,
      stdev: Optional[float] = 0.1,
  ) -> "Probabilities":
    """Construct a random Probabilities."""
    return cls(params=jax.random.normal(rng, shape=(n_dimensions - 1,)) * stdev)

  @classmethod
  def from_probs(cls, probs: jnp.ndarray) -> "Probabilities":
    """Construct Probabilities from a vector of probabilities."""
    log_probs = jnp.log(probs)
    log_probs_normalized, norm = log_probs[:-1], log_probs[-1]
    log_probs_normalized -= norm
    return cls(params=log_probs_normalized)

  @property
  def params(self):  # noqa: D102
    return self._params

  @property
  def dtype(self):  # noqa: D102
    return self._params.dtype

  def unnormalized_log_probs(self) -> jnp.ndarray:
    """Get the unnormalized log probabilities."""
    return jnp.concatenate([self._params, jnp.zeros((1,))], axis=-1)

  def log_probs(self) -> jnp.ndarray:
    """Get the log probabilities."""
    return jax.nn.log_softmax(self.unnormalized_log_probs())

  def probs(self) -> jnp.ndarray:
    """Get the probabilities."""
    return jax.nn.softmax(self.unnormalized_log_probs())

  def sample(self, rng: jax.Array, size: int) -> jnp.ndarray:
    """Sample from the distribution."""
    return jax.random.categorical(
        rng, logits=self.unnormalized_log_probs(), shape=(size,)
    )

  def tree_flatten(self):  # noqa: D102
    children = (self.params,)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)

  def __repr__(self):
    class_name = type(self).__name__
    children, aux = self.tree_flatten()
    return "{}({})".format(
        class_name, ", ".join([repr(c) for c in children] +
                              [f"{k}: {repr(v)}" for k, v in aux.items()])
    )

  def __hash__(self):
    return jax.tree_util.tree_flatten(self).__hash__()

  def __eq__(self, other):
    return jax.tree_util.tree_flatten(self) == jax.tree_util.tree_flatten(other)
