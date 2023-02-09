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

from typing import Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import frozen_dict

from ott.solvers.nn import potential_base

__all__ = ["MLP"]


class MLP(potential_base.PotentialBase):
  """A non-convex MLP.

  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is automatically set to
      1 (if `is_potential=True`) or the dimension
      of the input (if `is_potential=False`).
    is_potential: Model the potential if `True`, otherwise
      model the gradient of the potential
    act_fn: Activation function
  """

  dim_hidden: Sequence[int]
  is_potential: bool = True
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    squeeze = x.ndim == 1
    if squeeze:
      x = jnp.expand_dims(x, 0)
    assert x.ndim == 2, x.ndim
    n_input = x.shape[-1]

    z = x
    for n_hidden in self.dim_hidden:
      Wx = nn.Dense(n_hidden, use_bias=True)
      z = self.act_fn(Wx(z))

    if self.is_potential:
      Wx = nn.Dense(1, use_bias=True)
      z = Wx(z).squeeze(-1)

      quad_term = 0.5 * jax.vmap(jnp.dot)(x, x)
      z += quad_term
    else:
      Wx = nn.Dense(n_input, use_bias=True)
      z = x + Wx(z)

    return z.squeeze(0) if squeeze else z

  def create_train_state(
      self,
      rng: jnp.ndarray,
      optimizer: optax.OptState,
      input: Union[int, Tuple[int, ...]],
      params: Optional[frozen_dict.FrozenDict[str, jnp.ndarray]] = None,
  ) -> potential_base.PotentialTrainState:
    """Create initial `TrainState`."""
    if params is None:
      params = self.init(rng, jnp.ones(input))["params"]
    return potential_base.PotentialTrainState.create(
        apply_fn=self.apply,
        params=params,
        tx=optimizer,
        potential_value_fn=self.potential_value,
        potential_gradient_fn=self.potential_gradient,
    )
