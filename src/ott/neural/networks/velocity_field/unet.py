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
"""Modified from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py."""
import abc
import functools
import math
from typing import Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from flax import nnx

__all__ = ["UNet"]


def timestep_embedding(
    timesteps: jax.Array, dim: int, max_period: int = 10000
) -> jax.Array:
  half = dim // 2
  freqs = jnp.exp(
      -math.log(max_period) *
      jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
  )
  args = timesteps[:, None].astype(jnp.float32) * freqs[None]
  embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  if dim % 2:
    embedding = jnp.concatenate([embedding,
                                 jnp.zeros_like(embedding[:, :1])],
                                axis=-1)
  return embedding


class GroupNorm32(nnx.GroupNorm):

  def __call__(
      self, x: jax.Array, *, mask: Optional[jax.Array] = None
  ) -> jax.Array:
    return super().__call__(x.astype(jnp.float32), mask=mask).astype(x.dtype)


def conv_nd(
    dims: int,
    in_channels: Union[int, Tuple[int, ...]],
    out_channels: Union[int, Tuple[int, ...]],
    kernel_size: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]] = 1,
    *,
    dtype: Optional[jnp.dtype] = None,
    param_dtype: jnp.dtype = jnp.float32,
    padding: Union[int, Tuple[int, ...]] = 0,
    zero_init: bool = False,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> nnx.Conv:
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size,) * dims
  if isinstance(strides, int):
    strides = (strides,) * dims
  if isinstance(padding, int):
    padding = (padding,) * dims

  if zero_init:
    kwargs["kernel_init"] = nnx.initializers.constant(value=0.0)
    kwargs["bias_init"] = nnx.initializers.constant(value=0.0)

  return nnx.Conv(
      in_features=in_channels,
      out_features=out_channels,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      dtype=dtype,
      param_dtype=param_dtype,
      rngs=rngs,
      **kwargs,
  )


def normalization(
    channels: int,
    *,
    dtype: Optional[jnp.dtype] = None,
    param_dtype: jnp.dtype = jnp.float32,
    rngs: nnx.Rngs,
) -> nnx.GroupNorm:
  return GroupNorm32(
      num_groups=32,
      num_features=channels,
      epsilon=1e-5,
      dtype=dtype,
      param_dtype=param_dtype,
      use_fast_variance=False,
      rngs=rngs,
  )


class TimestepBlock(nnx.Module):

  @abc.abstractmethod
  def __call__(
      self,
      x: jax.Array,
      emb: jax.Array,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ) -> jax.Array:
    pass


class TimestepEmbedSequential(nnx.Module):

  def __init__(self, *layers: nnx.Module):
    super().__init__()
    self.layers = nnx.List(layers) if hasattr(nnx, "List") else list(layers)

  def __call__(
      self,
      x: jax.Array,
      emb: Optional[jax.Array] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ) -> jax.Array:
    for layer in self.layers:
      if isinstance(layer, TimestepBlock):
        x = layer(x, emb, rngs=rngs)
      else:
        x = layer(x)
    return x


class Upsample(nnx.Module):

  def __init__(
      self,
      channels: int,
      use_conv: bool,
      *,
      out_channels: Optional[int] = None,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.channels = channels
    self.out_channels = out_channels or channels
    if use_conv:
      self.conv = conv_nd(
          2,
          self.channels,
          self.out_channels,
          3,
          padding=1,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
    else:
      self.conv = None

  def __call__(self, x: jax.Array) -> jax.Array:
    assert x.shape[-1] == self.channels
    b, h, w, c = x.shape
    shape = (b, 2 * h, 2 * w, c)
    x = jax.image.resize(x, shape, method="nearest")
    if self.conv is not None:
      x = self.conv(x)
    return x


class Downsample(nnx.Module):

  def __init__(
      self,
      channels: int,
      use_conv: bool,
      *,
      out_channels: Optional[int] = None,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.channels = channels
    self.out_channels = out_channels or channels
    if use_conv:
      self.op = conv_nd(
          2,
          self.channels,
          self.out_channels,
          3,
          strides=2,
          padding=1,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
    else:
      self.op = functools.partial(
          nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)
      )

  def __call__(self, x: jax.Array) -> jax.Array:
    assert x.shape[-1] == self.channels
    return self.op(x)


class ResBlock(TimestepBlock):

  def __init__(
      self,
      channels: int,
      emb_channels: int,
      dropout: float,
      *,
      out_channels: Optional[int] = None,
      use_conv: bool = False,
      up: bool = False,
      down: bool = False,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.channels = channels
    self.emb_channels = emb_channels
    self.dropout = dropout
    self.out_channels = out_channels or channels
    self.use_conv = use_conv
    self.updown = up or down

    self.in_norm = normalization(
        channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
    )
    self.in_act = nnx.silu
    self.in_conv = conv_nd(
        2,
        channels,
        self.out_channels,
        3,
        padding=1,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    if up:
      self.h_upd = Upsample(
          channels,
          use_conv=False,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
      self.x_upd = Upsample(
          channels,
          use_conv=False,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
    elif down:
      self.h_upd = Downsample(
          channels,
          use_conv=False,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
      self.x_upd = Downsample(
          channels,
          use_conv=False,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
    else:
      self.h_upd = lambda x: x
      self.x_upd = lambda x: x

    self.emb_act = nnx.silu
    self.emb_layers = nnx.Linear(
        emb_channels,
        self.out_channels,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    self.out_norm = normalization(
        self.out_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
    )
    self.out_act = nnx.silu
    self.out_dropout = nnx.Dropout(rate=dropout)
    self.out_conv = conv_nd(
        2,
        self.out_channels,
        self.out_channels,
        3,
        padding=1,
        zero_init=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    if self.out_channels == channels:
      self.skip_connection = lambda x: x
    elif use_conv:
      self.skip_connection = conv_nd(
          2,
          channels,
          self.out_channels,
          3,
          padding=1,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
    else:
      self.skip_connection = conv_nd(
          2,
          channels,
          self.out_channels,
          1,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )

  def __call__(
      self,
      x: jax.Array,
      emb: jax.Array,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ) -> jax.Array:
    if self.updown:
      h = self.in_norm(x)
      h = self.in_act(h)
      h = self.h_upd(h)
      x = self.x_upd(x)
      h = self.in_conv(h)
    else:
      h = self.in_norm(x)
      h = self.in_act(h)
      h = self.in_conv(h)

    emb_out = self.emb_act(emb).astype(h.dtype)
    emb_out = self.emb_layers(emb_out)
    emb_out = emb_out[:, None, None, :]  # [b, 1, 1, t_emb]
    h = h + emb_out
    h = self.out_norm(h)
    h = self.out_act(h)
    h = self.out_dropout(h, rngs=rngs)
    h = self.out_conv(h)

    return self.skip_connection(x) + h


class QKVAttention(nnx.Module):

  def __init__(
      self,
      n_heads: int,
      attn_implementation: Optional[Literal["xla", "cudnn"]] = None,
  ):
    super().__init__()
    self.n_heads = n_heads
    self.attn_implementation = attn_implementation

  def __call__(self, qkv: jax.Array) -> jax.Array:
    bs, length, width = qkv.shape
    head_dim, rest = divmod(width, 3 * self.n_heads)
    assert rest == 0, rest
    scale = 1.0 / math.sqrt(math.sqrt(head_dim))

    # Split into heads and channels
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(bs, length, self.n_heads, head_dim)
    k = k.reshape(bs, length, self.n_heads, head_dim)
    v = v.reshape(bs, length, self.n_heads, head_dim)

    dtype = jnp.bfloat16 if self.attn_implementation == "cudnn" else q.dtype
    a = jax.nn.dot_product_attention(
        q.astype(dtype),
        k.astype(dtype),
        v.astype(dtype),
        scale=scale,
        implementation=self.attn_implementation,
    ).astype(q.dtype)
    return a.reshape(bs, length, self.n_heads * head_dim)


class AttentionBlock(nnx.Module):

  def __init__(
      self,
      channels: int,
      *,
      num_heads: int = 1,
      attn_implementation: Optional[Literal["xla", "cudnn"]] = None,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.channels = channels
    self.num_heads = num_heads
    self.norm = normalization(
        channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
    )
    self.qkv = conv_nd(
        1,
        channels,
        channels * 3,
        1,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    # split qkv before split heads
    self.attention = QKVAttention(
        self.num_heads, attn_implementation=attn_implementation
    )

    self.proj_out = conv_nd(
        1,
        channels,
        channels,
        1,
        zero_init=True,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    # [B, H, W, C]
    b, *spatial, c = x.shape
    x = x.reshape(b, -1, c)  # [B, H * W, C]
    qkv = self.qkv(self.norm(x))
    h = self.attention(qkv)
    h = self.proj_out(h)
    return (x + h).reshape(b, *spatial, c)


class UNet(nnx.Module):
  """UNet model with attention and timestep embedding.

  Args:
    shape: Input shape ``[height, width, channels]``.
    model_channels: Number of model channels.
    num_res_blocks: Number of residual blocks.
    attention_resolutions: Resolutions at which to add self-attention.
    out_channels: Number of output channels. If :obj:`None`, use input channels.
    dropout: Dropout rate.
    channel_mult: Multiplier for ``model_channels`` for each resolution.
    time_embed_dim: Dimensionality of the time embedding.
      If :obj:`None`, use ``4 * model_channels``.
      If :class:`float`, use ``int(time_embed_dim * model_channels)``.
    conv_resample: If :obj:`False`, don't use convolution for upsampling and use
      average pooling for downsampling instead of using convolution.
    num_heads: Number of attention heads for the input and middle blocks.
    num_heads_upsample: Number of attention heads for the output blocks.
      If :obj:`None`, use ``num_heads``.
    resblock_updown: Whether to use residual blocks for up/downsampling.
    num_classes: Number of classes.
    dtype: Data type for computation.
    param_dtype: Data type for parameters.
    attn_implementation: Attention implementation for
      :func:`~jax.nn.dot_product_attention`.
    rngs: Random number generator for initialization.
  """

  def __init__(
      self,
      *,
      shape: Tuple[int, int, int],
      model_channels: int,
      num_res_blocks: int,
      attention_resolutions: Tuple[int, ...],
      out_channels: Optional[int] = None,
      dropout: float = 0.0,
      channel_mult: Tuple[float, ...] = (1, 2, 4, 8),
      time_embed_dim: Optional[Union[int, float]] = None,
      conv_resample: bool = True,
      num_heads: int = 1,
      num_heads_upsample: Optional[int] = None,
      resblock_updown: bool = False,
      num_classes: Optional[int] = None,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      attn_implementation: Optional[Literal["xla", "cudnn"]] = None,
      rngs: nnx.Rngs,
  ):
    super().__init__()

    image_size, _, in_channels = shape
    out_channels = out_channels or in_channels
    attention_resolutions = tuple(
        image_size // res for res in attention_resolutions
    )
    num_heads_upsample = num_heads_upsample or num_heads

    self.dtype = dtype
    self.in_channels = in_channels
    self.model_channels = model_channels
    self.num_res_blocks = num_res_blocks
    self.attention_resolutions = attention_resolutions
    self.dropout = dropout
    self.channel_mult = channel_mult
    self.conv_resample = conv_resample
    self.num_heads = num_heads
    self.num_heads_upsample = num_heads_upsample

    # Time embedding
    if time_embed_dim is None:
      time_embed_dim = model_channels * 4
    elif isinstance(time_embed_dim, float):
      time_embed_dim = int(time_embed_dim * model_channels)
    assert isinstance(time_embed_dim, int), time_embed_dim

    self.time_embed = TimestepEmbedSequential(
        nnx.Linear(
            model_channels,
            time_embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
        nnx.silu,
        nnx.Linear(
            time_embed_dim,
            time_embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
    )

    # condition embedding
    if num_classes is not None:
      self.label_emb = nnx.Embed(
          num_embeddings=num_classes, features=time_embed_dim, rngs=rngs
      )
    else:
      self.label_emb = None

    # Input blocks
    self.input_blocks = nnx.List() if hasattr(nnx, "List") else []
    input_block_chans = [model_channels]
    ch = int(channel_mult[0] * model_channels)
    ds = 1

    # First input block (just convolution)
    self.input_blocks.append(
        TimestepEmbedSequential(
            conv_nd(
                2,
                in_channels,
                ch,
                3,
                padding=1,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        )
    )

    # Rest of input blocks
    for level, mult in enumerate(channel_mult):
      for _ in range(num_res_blocks):
        layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=int(mult * model_channels),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        ]
        ch = int(mult * model_channels)

        if ds in attention_resolutions:
          layers.append(
              AttentionBlock(
                  ch,
                  num_heads=num_heads,
                  attn_implementation=attn_implementation,
                  dtype=dtype,
                  param_dtype=param_dtype,
                  rngs=rngs,
              )
          )

        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

      # Downsample if not last level
      if level != len(channel_mult) - 1:
        out_ch = ch
        self.input_blocks.append(
            TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=out_ch,
                    down=True,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ) if resblock_updown else Downsample(
                    ch,
                    conv_resample,
                    out_channels=out_ch,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
        )
        ch = out_ch
        input_block_chans.append(ch)
        ds *= 2

    self.middle_block = TimestepEmbedSequential(
        ResBlock(
            ch,
            time_embed_dim,
            dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
        AttentionBlock(
            ch,
            num_heads=num_heads,
            attn_implementation=attn_implementation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
        ResBlock(
            ch,
            time_embed_dim,
            dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
    )

    # Output blocks
    self.output_blocks = nnx.List() if hasattr(nnx, "List") else []
    for level, mult in list(enumerate(channel_mult))[::-1]:
      for i in range(num_res_blocks + 1):
        ich = input_block_chans.pop()
        layers = [
            ResBlock(
                ch + ich,
                time_embed_dim,
                dropout,
                out_channels=int(model_channels * mult),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        ]
        ch = int(model_channels * mult)

        if ds in attention_resolutions:
          layers.append(
              AttentionBlock(
                  ch,
                  num_heads=num_heads_upsample,
                  attn_implementation=attn_implementation,
                  dtype=dtype,
                  param_dtype=param_dtype,
                  rngs=rngs,
              )
          )

        if level and i == num_res_blocks:
          out_ch = ch
          layers.append(
              ResBlock(
                  ch,
                  time_embed_dim,
                  dropout,
                  out_channels=out_ch,
                  up=True,
                  dtype=dtype,
                  param_dtype=param_dtype,
                  rngs=rngs,
              ) if resblock_updown else Upsample(
                  ch,
                  conv_resample,
                  out_channels=out_ch,
                  dtype=dtype,
                  param_dtype=param_dtype,
                  rngs=rngs,
              )
          )
          ds //= 2

        self.output_blocks.append(TimestepEmbedSequential(*layers))

    self.out = TimestepEmbedSequential(
        normalization(ch, dtype=dtype, param_dtype=param_dtype, rngs=rngs),
        nnx.silu,
        conv_nd(
            2,
            ch,
            out_channels,
            3,
            padding=1,
            zero_init=True,
            # dtype=dtype,  # don't cast computations to (bf)float16
            param_dtype=param_dtype,
            rngs=rngs,
        ),
    )

  def __call__(
      self,
      t: jax.Array,
      x: jax.Array,
      cond: Optional[jax.Array] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ) -> jax.Array:
    """Compute the velocity.

    Args:
      t: Time array of shape ``[batch,]``.
      x: Image of shape ``[batch, height, width, channels]``.
      cond: Class condition array of shape ``[batch,]``.
      rngs: Random number generator for dropout.

    Returns:
      The velocity array of shape ``[batch, height, width, channels]``.
    """
    emb = self.time_embed(timestep_embedding(t, self.model_channels), rngs=rngs)
    # TODO(michalk8): generalize for different types of conditions
    if self.label_emb is not None:
      assert cond is not None, "Please provide a condition."
      # emb is cast to `self.dtype` inside each submodule
      emb = emb + self.label_emb(cond)
    h = x.astype(self.dtype)
    hs = []
    for module in self.input_blocks:
      h = module(h, emb, rngs=rngs)
      hs.append(h)
    h = self.middle_block(h, emb, rngs=rngs)
    for module in self.output_blocks:
      h = jnp.concatenate([h, hs.pop()], axis=-1)
      h = module(h, emb, rngs=rngs)
    h = h.astype(x.dtype)
    # output's compute dtype
    return self.out(h, rngs=rngs)
