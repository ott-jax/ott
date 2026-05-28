# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input convex neural networks and KeyNet (vector-output variant)."""

from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from flax import nnx

from ott.neural.networks.layers import initializers, posdef

__all__ = ["ICNN", "KeyNet"]

DEFAULT_KERNEL_INIT = nnx.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nnx.initializers.zeros_init()


def _get_act_alpha(act_fn: Callable) -> float:
  """Get the negative slope (alpha) of an activation for principled init."""
  if act_fn is jax.nn.relu:
    return 0.0
  if act_fn is jax.nn.leaky_relu:
    return 0.01
  # For unknown activations, assume ReLU-like
  return 0.0


def _normalize_wx_inject(
    wx_inject: Union[bool, Tuple[bool, ...], int],
    num_layers: int,
) -> Tuple[bool, ...]:
  """Convert wx_inject specification to a boolean tuple.

  Args:
    wx_inject: Controls input re-injection pattern. Can be:
      - ``bool``: inject at all layers (True) or none (False).
      - ``tuple[bool, ...]``: explicit per-layer mask.
      - ``int``: frequency (e.g., 3 means inject every 3rd layer).
    num_layers: Number of layers after the first (len(dims) - 2).

  Returns:
    Tuple of booleans for each layer after wx0.
  """
  if isinstance(wx_inject, bool):
    return (wx_inject,) * num_layers
  if isinstance(wx_inject, int):
    return tuple((i + 1) % wx_inject == 0 for i in range(num_layers))
  assert len(wx_inject) == num_layers, (len(wx_inject), num_layers)
  return tuple(wx_inject)


class ICNN(nnx.Module):
  """Input convex neural network (ICNN).

  Implementation of input convex neural networks as introduced in
  :cite:`amos:17` with flexible input re-injection, multiple rectifier
  options, and optional positive-definite quadratic potentials
  :cite:`vesseron:24`.

  The network computes a convex function :math:`f: R^d -> R^k` where convexity
  holds component-wise when :math:`k > 1`.

  Architecture::

    z_0 = act(W_x0 @ x)
    z_i = act(W_z_i @ z_{i-1} + W_x_i @ x)   # wx_inject controls W_x_i
    out = z_N + pos_def_potentials(x)        # optional

  Convexity is enforced by requiring W_z_i >= 0 (via rectifier) and
  using convex activation functions.

  Args:
    dim_hidden: Sequence of hidden layer sizes. The output dimension
      defaults to 1 (scalar potential); set ``output_dim`` for vector output.
    input_dim: Dimension of the input ``x``.
    output_dim: Output dimension. Defaults to 1 (scalar convex function).
      When > 1, each output component is convex in the input.
    rectifier_fn: Function applied to W_z kernels to enforce
      non-negativity. The default is :func:`~jax.nn.softplus`.
    act_fn: Activation function (must be convex for the network to be
      convex). The default is :func:`~jax.nn.relu`.
    wx_inject: Controls input re-injection at intermediate layers.
    use_bias: Whether to use bias terms.
    use_softmax: If True, the ``W_z`` :class:`PositiveDense
      <ott.neural.networks.layers.posdef.PositiveDense>` layers use
      column-wise softmax normalization instead of ``rectifier_fn``.
    use_sinkhorn: If True, the ``W_z`` :class:`PositiveDense
      <ott.neural.networks.layers.posdef.PositiveDense>` layers use
      Sinkhorn normalization instead of ``rectifier_fn``.
    pos_def_rank: Rank of optional PosDefPotentials term. Set to 0
      to disable (default).
    principled_init: If True, override ``wz_kernel_init`` and the W_z
      bias initializer with the principled ICNN initialization of
      :cite:`hoedt:2023`, which controls correlation and variance
      propagation through layers with positive weights.
    kernel_init: Initializer for W_x (unrestricted) weights.
    wz_kernel_init: Initializer for W_z (positive) weights. Ignored when
      ``principled_init=True``.
    bias_init: Initializer for biases.
    rngs: Random number generators.
  """

  def __init__(
      self,
      dim_hidden: Sequence[int],
      *,
      input_dim: int,
      output_dim: int = 1,
      rectifier_fn: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
      act_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      wx_inject: Union[bool, Tuple[bool, ...], int] = True,
      use_bias: bool = True,
      use_softmax: bool = False,
      use_sinkhorn: bool = False,
      pos_def_rank: int = 0,
      principled_init: bool = False,
      kernel_init: nnx.initializers.Initializer = DEFAULT_KERNEL_INIT,
      wz_kernel_init: nnx.initializers.Initializer = DEFAULT_KERNEL_INIT,
      bias_init: nnx.initializers.Initializer = DEFAULT_BIAS_INIT,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self._output_dim = output_dim
    self._act_fn_call = act_fn

    dim_hidden = list(dim_hidden) + [output_dim]
    dims = [input_dim] + dim_hidden
    num_layers = len(dims) - 2
    inject_mask = _normalize_wx_inject(wx_inject, num_layers)

    # Compute per-layer wz initializers for principled init
    if principled_init:
      alpha = _get_act_alpha(act_fn)
      wz_kernel_inits = []
      wz_bias_inits = []
      for d_in in dims[1:-2]:
        w_init, b_init = initializers.principled_icnn_init(
            d_in, alpha=alpha, rectifier_fn=rectifier_fn
        )
        wz_kernel_inits.append(w_init)
        wz_bias_inits.append(b_init)
      # Last layer: identity activation (alpha=1)
      if num_layers > 0:
        w_init, b_init = initializers.principled_icnn_init(
            dims[-2], alpha=1.0, rectifier_fn=rectifier_fn
        )
        wz_kernel_inits.append(w_init)
        wz_bias_inits.append(b_init)
    else:
      wz_kernel_inits = [wz_kernel_init] * num_layers
      wz_bias_inits = [bias_init] * num_layers

    self.wx0 = nnx.Linear(
        input_dim,
        dims[1],
        use_bias=use_bias,
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs,
    )

    self.wx_layers = nnx.List([
        nnx.Linear(
            input_dim,
            d_out,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
        ) if inject else None
        for d_out, inject in zip(dims[2:], inject_mask, strict=True)
    ])

    self.wz_layers = nnx.List([
        posdef.PositiveDense(
            d_in,
            d_out,
            rectifier_fn=rectifier_fn,
            use_softmax=use_softmax,
            use_sinkhorn=use_sinkhorn,
            use_bias=use_bias,
            kernel_init=k_init,
            bias_init=b_init,
            rngs=rngs,
        ) for d_in, d_out, k_init, b_init in
        zip(dims[1:-1], dims[2:], wz_kernel_inits, wz_bias_inits, strict=True)
    ])

    self.pos_def_potentials = (
        posdef.PosDefPotentials(
            input_dim,
            output_dim,
            rank=pos_def_rank,
            use_linear=True,
            use_bias=True,
            rngs=rngs,
        ) if pos_def_rank > 0 else None
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Evaluate the ICNN.

    Args:
      x: Input of shape ``[..., input_dim]``.

    Returns:
      Output of shape ``[...]`` if ``output_dim == 1``, else
      ``[..., output_dim]``.
    """
    squeeze = x.ndim == 1
    if squeeze:
      x = x[None]

    z = self._act_fn_call(self.wx0(x))

    for wx, wz in zip(self.wx_layers, self.wz_layers, strict=True):
      if wx is not None:
        z = self._act_fn_call(wz(z) + wx(x))
      else:
        z = self._act_fn_call(wz(z))

    if self.pos_def_potentials is not None:
      z = z + self.pos_def_potentials(x)

    if self._output_dim == 1:
      z = z.squeeze(-1)

    return z.squeeze(0) if squeeze else z

  def gradient(self, x: jax.Array) -> jax.Array:
    """Gradient of the convex potential w.r.t. input.

    For scalar output (``output_dim == 1``), returns the gradient.
    For vector output, returns the Jacobian.

    Args:
      x: Input of shape ``[batch, input_dim]``.

    Returns:
      Gradients of shape ``[batch, input_dim]`` (scalar output) or
      ``[batch, output_dim, input_dim]`` (vector output).
    """

    def forward(x: jax.Array) -> jax.Array:
      return nnx.merge(graphdef, state)(x)

    graphdef, state = nnx.split(self)
    if self._output_dim == 1:
      return jax.vmap(jax.grad(forward))(x)
    return jax.vmap(jax.jacobian(forward))(x)

  @property
  def is_potential(self) -> bool:
    """Whether this module represents a potential (True) or vector field."""
    return True


class KeyNet(nnx.Module):
  """Vector-output network with ICNN-like architecture.

  Unlike :class:`ICNN` which outputs a scalar convex function and requires
  autodiff to compute gradients, KeyNet directly outputs vectors. The
  architecture mirrors ICNN but without non-negativity constraints on the
  layer-to-layer weights.

  The scalar potential is recovered as f(x) = <KeyNet(x), x>.

  When ``resnet=True``, output is ``x + F(x)`` (residual mode), making
  the model learn a correction to the input query.

  Args:
    dim_hidden: Sequence of hidden layer sizes.
    input_dim: Dimension of the input ``x``.
    output_dim: Output vector dimension. Defaults to ``input_dim``.
      Typically, equals the input dimension for gradient-of-potential
      interpretation.
    num_outputs: Number of output vectors.
    resnet: If True, output ``x + F(x)`` instead of ``F(x)``.
    act_fn: Activation function.
    wx_inject: Controls input re-injection pattern.
    use_bias: Whether to use bias terms.
    kernel_init: Initializer for all weights.
    bias_init: Initializer for biases.
    final_layer_scale: Scale for final layer init. Defaults to 0.01
      for resnet mode (small initial corrections), 1.0 otherwise.
    rngs: Random number generators.
  """

  def __init__(
      self,
      dim_hidden: Sequence[int],
      *,
      input_dim: int,
      output_dim: Optional[int] = None,
      num_outputs: Optional[int] = None,
      resnet: bool = False,
      act_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      wx_inject: Union[bool, Tuple[bool, ...], int] = True,
      use_bias: bool = True,
      kernel_init: nnx.initializers.Initializer = DEFAULT_KERNEL_INIT,
      bias_init: nnx.initializers.Initializer = DEFAULT_BIAS_INIT,
      final_layer_scale: Optional[float] = None,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self._resnet = resnet
    self._act_fn_call = act_fn
    self._num_outputs = num_outputs

    output_dim = output_dim if output_dim is not None else input_dim
    if self._num_outputs is not None:
      output_dim = output_dim * self._num_outputs
    dims = [input_dim] + list(dim_hidden) + [output_dim]
    num_layers = len(dims) - 2
    inject_mask = _normalize_wx_inject(wx_inject, num_layers)

    scale = final_layer_scale
    if scale is None:
      scale = 0.01 if resnet else 1.0

    def scaled_init(s):

      def init_fn(key, shape, dtype=None):
        return s * kernel_init(key, shape, dtype)

      return init_fn

    self.wx0 = nnx.Linear(
        input_dim,
        dims[1],
        use_bias=use_bias,
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs,
    )

    self.wx_layers = nnx.List([
        nnx.Linear(
            input_dim,
            d_out,
            use_bias=False,
            kernel_init=scaled_init(scale) if
            (i == num_layers - 1) else kernel_init,
            rngs=rngs
        ) if inject else None
        for i, (d_out,
                inject) in enumerate(zip(dims[2:], inject_mask, strict=True))
    ])

    self.wz_layers = nnx.List([
        nnx.Linear(
            d_in,
            d_out,
            use_bias=use_bias,
            kernel_init=scaled_init(scale) if
            (i == len(dims) - 3) else kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        ) for i, (d_in,
                  d_out) in enumerate(zip(dims[1:-1], dims[2:], strict=True))
    ])

  def __call__(self, x: jax.Array) -> jax.Array:
    """Compute scalar potential f(x) = <grad(x), x>.

    Args:
      x: Input of shape ``[batch_size, input_dim]``.

    Returns:
      Scalar output of shape ``[batch_size,]`` or ``[batch_size, num_outputs]``.
    """
    g = self.gradient(x)
    if self._num_outputs is None:
      return jnp.sum(g * x, axis=-1)
    return jnp.sum(g * x[:, None], axis=-1)

  def gradient(self, x: jax.Array) -> jax.Array:
    """Compute the vector output (predicted gradient / key).

    Args:
      x: Input of shape ``[batch_size, input_dim]``.

    Returns:
      Output of shape ``[batch_size, output_dim]`` or
      ``[batch_size, num_outputs, output_dim]``.
    """
    batch_size, _ = x.shape
    z = self._act_fn_call(self.wx0(x))

    for wx, wz in zip(self.wx_layers, self.wz_layers, strict=True):
      if wx is not None:
        z = self._act_fn_call(wz(z) + wx(x))
      else:
        z = self._act_fn_call(wz(z))

    if self._resnet:
      z = x + z
    if self._num_outputs is not None:
      z = z.reshape(batch_size, self._num_outputs, -1)
    return z

  @property
  def is_potential(self) -> bool:
    """KeyNet models a potential via f(x) = <gradient(x), x>."""
    return True
