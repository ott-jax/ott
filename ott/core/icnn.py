# Copyright 2022 Google LLC.
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

# Lint as: python3
"""Implementation of Amos+(2017) input convex neural networks (ICNN)."""

from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax.nn import initializers

from ott.core.layers import PosDefPotentials, PositiveDense
from ott.geometry import matrix_square_root

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class ICNN(nn.Module):
  """Input convex neural network (ICNN) architecture with initialization.

  Implementation of input convex neural networks as introduced in
  :cite:`amos:17` with initialization schemes proposed by :cite:`bunne:22`.

  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is 1 by default.
    init_std: value of standard deviation of weight initialization method.
    init_fn: choice of initialization method for weight matrices (default:
      `jax.nn.initializers.normal`).
    act_fn: choice of activation function used in network architecture
      (needs to be convex, default: `nn.relu`).
    pos_weights: choice to enforce positivity of weight or use regularizer.
    dim_data: data dimensionality (default: 2).
    gaussian_map: data inputs of source and target measures for
      initialization scheme based on Gaussian approximation of input and
      target measure (if None, identity initialization is used).
  """

  dim_hidden: Sequence[int]
  init_std: float = 1e-1
  init_fn: Callable = jax.nn.initializers.normal
  act_fn: Callable = nn.relu
  pos_weights: bool = True
  dim_data: int = 2
  gaussian_map: Tuple[jnp.ndarray, jnp.ndarray] = None

  def setup(self) -> None:
    self.num_hidden = len(self.dim_hidden)

    if self.pos_weights:
      hid_dense = PositiveDense
      # this function needs to be the inverse map of function
      # used in PositiveDense layers
      rescale = hid_dense.inv_rectifier_fn
    else:
      hid_dense = nn.Dense
      rescale = lambda x: x
    self.use_init = False
    # check if Gaussian map was provided
    if self.gaussian_map is not None:
      factor, mean = self._compute_gaussian_map(self.gaussian_map)
    else:
      factor, mean = self._compute_identity_map(self.dim_data)

    w_zs = []
    # keep track of previous size to normalize accordingly
    normalization = 1
    # subsequent layers propagate value of potential provided by
    # first layer in x normalization factor is rescaled accordingly
    for i in range(self.num_hidden):
      w_zs.append(
          hid_dense(
              self.dim_hidden[i],
              kernel_init=initializers.constant(rescale(1.0 / normalization)),
              use_bias=False,
          )
      )
      normalization = self.dim_hidden[i]
    # final layer computes average, still with normalized rescaling
    w_zs.append(
        hid_dense(
            1,
            kernel_init=initializers.constant(rescale(1.0 / normalization)),
            use_bias=False,
        )
    )
    self.w_zs = w_zs

    w_xs = []
    # first square layer, initialized to identity
    w_xs.append(
        PosDefPotentials(
            self.dim_data,
            num_potentials=1,
            kernel_init=lambda *args, **kwargs: factor,
            bias_init=lambda *args, **kwargs: mean,
            use_bias=True,
        )
    )

    # subsequent layers reinjected into convex functions
    for i in range(self.num_hidden):
      w_xs.append(
          nn.Dense(
              self.dim_hidden[i],
              kernel_init=self.init_fn(self.init_std),
              bias_init=self.init_fn(self.init_std),
              use_bias=True,
          )
      )
    # final layer, to output number
    w_xs.append(
        nn.Dense(
            1,
            kernel_init=self.init_fn(self.init_std),
            bias_init=self.init_fn(self.init_std),
            use_bias=True,
        )
    )
    self.w_xs = w_xs

  @staticmethod
  def _compute_gaussian_map(
      inputs: Tuple[jnp.ndarray, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    def compute_moments(x, reg=1e-4, sqrt_inv=False):
      shape = x.shape
      z = x.reshape(shape[0], -1)
      mu = jnp.expand_dims(jnp.mean(z, axis=0), 0)
      z = z - mu
      matmul = lambda a, b: jnp.matmul(a, b)
      sigma = jax.vmap(matmul)(jnp.expand_dims(z, 2), jnp.expand_dims(z, 1))
      # unbiased estimate
      sigma = jnp.sum(sigma, axis=0) / (shape[0] - 1)
      # regularize
      sigma = sigma + reg * jnp.eye(shape[1])

      if sqrt_inv:
        sigma_sqrt, sigma_inv_sqrt, _ = matrix_square_root.sqrtm(sigma)
        return sigma, sigma_sqrt, sigma_inv_sqrt, mu
      else:
        return sigma, mu

    source, target = inputs
    _, covs_sqrt, covs_inv_sqrt, mus = compute_moments(source, sqrt_inv=True)
    covt, mut = compute_moments(target, sqrt_inv=False)

    mo = matrix_square_root.sqrtm_only(
        jnp.dot(jnp.dot(covs_sqrt, covt), covs_sqrt)
    )
    A = jnp.dot(jnp.dot(covs_inv_sqrt, mo), covs_inv_sqrt)
    b = jnp.squeeze(mus) - jnp.linalg.solve(A, jnp.squeeze(mut))
    A = matrix_square_root.sqrtm_only(A)

    return jnp.expand_dims(A, 0), jnp.expand_dims(b, 0)

  @staticmethod
  def _compute_identity_map(input_dim: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A = jnp.eye(input_dim).reshape((1, input_dim, input_dim))
    b = jnp.zeros((1, input_dim))

    return A, b

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> float:
    for i in range(self.num_hidden + 2):
      if i == 0:
        z = self.w_xs[i](x)
      # apply both transform on hidden state and x
      # x is one step ahead as there is one more hidden layer for x
      else:
        z = jnp.add(self.w_zs[i - 1](z), self.w_xs[i](x))
      if i != 0 or i != self.num_hidden + 1:
        z = self.act_fn(z)
    return jnp.squeeze(z)

  def create_train_state(
      self,
      rng: jnp.ndarray,
      optimizer: optax.OptState,
      input: Union[int, Tuple[int, ...]],
  ) -> train_state.TrainState:
    """Create initial `TrainState`."""
    params = self.init(rng, jnp.ones(input))["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
