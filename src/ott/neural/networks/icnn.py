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
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from flax import linen as nn

from ott.neural.networks import potentials
from ott.neural.networks.layers import posdef

__all__ = ["ICNN"]

DEFAULT_KERNEL_INIT = posdef.DEFAULT_KERNEL_INIT
DEFAULT_RECTIFIER = nn.activation.relu
DEFAULT_ACTIVATION = nn.activation.relu


class ICNN(potentials.BasePotential):
  """Input convex neural network (ICNN).

  Implementation of input convex neural networks as introduced in
  :cite:`amos:17` with initialization schemes proposed by :cite:`bunne:22`,
  and (low-rank + diagonal) quadratic on inputs at each layer, by
  :cite:`vesseron:24`.

  Args:
    dim_data: data dimensionality.
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is 1 by default.
    ranks: ranks of the matrices :math:`A_i` used as low-rank factors
      for the quadratic potentials. If a sequence is passed, it must contain
      ``len(dim_hidden) + 2`` elements, where the last 2 elements correspond
      to the ranks of the final layer with dimension 1 and the potentials,
      respectively.
    init_fn: Initializer for the kernel weight matrices.
      The default is :func:`~flax.linen.initializers.normal`.
    act_fn: choice of activation function used in network architecture,
      needs to be convex. The default is :func:`~flax.linen.activation.relu`.
    pos_weights: Enforce positive weights with a projection.
      If :obj:`False`, the positive weights should be enforced with clipping
      or regularization in the loss.
    rectifier_fn: function to ensure the non negativity of the weights.
      The default is :func:`~flax.linen.activation.relu`.
    gaussian_map_samples: Tuple of source and target points, used to initialize
      the ICNN to mimic the linear Bures map that morphs the (Gaussian
      approximation) of the input measure to that of the target measure. If
      :obj:`None`, the identity initialization is used, and ICNN mimics half the
      squared Euclidean norm.
  """

  dim_data: int
  dim_hidden: Sequence[int]
  ranks: Union[int, Tuple[int, ...]] = 1
  init_fn: Callable[[jax.Array, Tuple[int, ...], Any],
                    jnp.ndarray] = DEFAULT_KERNEL_INIT
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = DEFAULT_ACTIVATION
  pos_weights: bool = False
  rectifier_fn: Callable[[jnp.ndarray], jnp.ndarray] = DEFAULT_RECTIFIER
  gaussian_map_samples: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None

  def setup(self) -> None:  # noqa: D102
    dim_hidden = list(self.dim_hidden) + [1]
    *ranks, pos_def_rank = self._normalize_ranks()

    # final layer computes average, still with normalized rescaling
    self.w_zs = [self._get_wz(dim) for dim in dim_hidden[1:]]
    # subsequent layers re-injected into convex functions
    self.w_xs = [
        self._get_wx(dim, rank) for dim, rank in zip(dim_hidden, ranks)
    ]
    self.pos_def_potentials = self._get_pos_def_potentials(pos_def_rank)

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    w_x, *w_xs = self.w_xs
    assert len(self.w_zs) == len(w_xs), (len(self.w_zs), len(w_xs))

    z = self.act_fn(w_x(x))
    for w_z, w_x in zip(self.w_zs, w_xs):
      z = self.act_fn(w_z(z) + w_x(x))
    z = z + self.pos_def_potentials(x)

    return z.squeeze()

  def _get_wz(self, dim: int) -> nn.Module:
    if self.pos_weights:
      return posdef.PositiveDense(
          dim,
          kernel_init=self.init_fn,
          use_bias=False,
          rectifier_fn=self.rectifier_fn,
      )

    return nn.Dense(
        dim,
        kernel_init=self.init_fn,
        use_bias=False,
    )

  def _get_wx(self, dim: int, rank: int) -> nn.Module:
    return posdef.PosDefPotentials(
        rank=rank,
        num_potentials=dim,
        use_linear=True,
        use_bias=True,
        kernel_diag_init=nn.initializers.zeros,
        kernel_lr_init=self.init_fn,
        kernel_linear_init=self.init_fn,
        bias_init=nn.initializers.zeros,
    )

  def _get_pos_def_potentials(self, rank: int) -> posdef.PosDefPotentials:
    kwargs = {
        "num_potentials": 1,
        "use_linear": True,
        "use_bias": True,
        "bias_init": nn.initializers.zeros
    }

    if self.gaussian_map_samples is None:
      return posdef.PosDefPotentials(
          rank=rank,
          kernel_diag_init=nn.initializers.ones,
          kernel_lr_init=nn.initializers.zeros,
          kernel_linear_init=nn.initializers.zeros,
          **kwargs,
      )

    source, target = self.gaussian_map_samples
    return posdef.PosDefPotentials.init_from_samples(
        source,
        target,
        rank=self.dim_data,
        kernel_diag_init=nn.initializers.zeros,
        **kwargs,
    )

  def _normalize_ranks(self) -> Tuple[int, ...]:
    # +2 for the newly added layer with 1 + the final potentials
    n_ranks = len(self.dim_hidden) + 2
    if isinstance(self.ranks, int):
      return (self.ranks,) * n_ranks

    assert len(self.ranks) == n_ranks, (len(self.ranks), n_ranks)
    return tuple(self.ranks)

  @property
  def is_potential(self) -> bool:  # noqa: D102
    return True
