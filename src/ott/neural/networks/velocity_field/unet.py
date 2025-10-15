import abc
import math
from typing import Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from flax import nnx

__all__ = [
    "timestep_embedding",
    "GroupNorm32",
    "conv_nd",
    "normalization",
    "TimestepEmbedSequential",
    "Upsample",
    "Downsample",
    "ResBlock",
    "QKVAttention",
    "AttentionBlock",
    "UNetModel",
    "UNetModelWrapper",
]


def timestep_embedding(
    timesteps: jax.Array, dim: int, max_period: int = 10000
) -> jax.Array:
  """Create sinusoidal timestep embeddings.

  :param timesteps: a 1-D Tensor of N indices, one per batch element.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an [N x dim] Tensor of positional embeddings.
  """
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
    dims: Union[int, Tuple[int, ...]],
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
  """Create a 1D, 2D, or 3D convolution module.
  """
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
  """Make a standard normalization layer."""
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
  """Any module where forward() takes timestep embeddings as a second argument."""

  @abc.abstractmethod
  def __call__(
      self,
      x: jax.Array,
      emb: jax.Array,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ) -> jax.Array:
    """Apply the module to `x` given `emb` timestep embeddings."""
    raise NotImplementedError


class TimestepEmbedSequential(nnx.Module):
  """A sequential module that passes timestep embeddings to the children that support it."""

  def __init__(self, *layers: nnx.Module):
    super().__init__()
    self.layers = list(layers)

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
  """An upsampling layer with an optional convolution."""

  def __init__(
      self,
      channels: int,
      use_conv: bool,
      dims: int = 2,
      *,
      out_channels: Optional[int] = None,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    assert dims == 2, dims
    self.channels = channels
    self.out_channels = out_channels or channels
    self.use_conv = use_conv
    self.dims = dims
    if use_conv:
      self.conv = conv_nd(
          dims,
          self.channels,
          self.out_channels,
          3,
          padding=1,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )

  def __call__(self, x: jax.Array) -> jax.Array:
    assert x.shape[-1] == self.channels
    shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
    x = jax.image.resize(x, shape, method="nearest")
    if self.use_conv:
      x = self.conv(x)
    return x


class Downsample(nnx.Module):
  """A downsampling layer with an optional convolution."""

  def __init__(
      self,
      channels: int,
      use_conv: bool,
      dims: int = 2,
      *,
      out_channels: Optional[int] = None,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.channels = channels
    self.out_channels = out_channels or channels
    self.use_conv = use_conv
    self.dims = dims

    strides = 2 if dims != 3 else (1, 2, 2)
    if use_conv:
      self.op = conv_nd(
          dims,
          self.channels,
          self.out_channels,
          3,
          strides=strides,
          padding=1,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
    else:
      raise NotImplementedError("Average pooling is not implemented.")

  def __call__(self, x: jax.Array) -> jax.Array:
    assert x.shape[-1] == self.channels
    return self.op(x)


class ResBlock(TimestepBlock):
  """A residual block that can optionally change the number of channels."""

  def __init__(
      self,
      channels: int,
      emb_channels: int,
      dropout: float,
      *,
      out_channels: Optional[int] = None,
      use_conv: bool = False,
      use_scale_shift_norm: bool = False,
      dims: int = 2,
      use_checkpoint: bool = False,
      up: bool = False,
      down: bool = False,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    assert not use_checkpoint, "Checkpointing is not implemented."
    super().__init__()
    self.channels = channels
    self.emb_channels = emb_channels
    self.dropout = dropout
    self.out_channels = out_channels or channels
    self.use_conv = use_conv
    self.use_checkpoint = use_checkpoint
    self.use_scale_shift_norm = use_scale_shift_norm
    self.updown = up or down

    self.in_norm = normalization(
        channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
    )
    self.in_act = nnx.silu
    self.in_conv = conv_nd(
        dims,
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
          dims=dims,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
      self.x_upd = Upsample(
          channels,
          use_conv=False,
          dims=dims,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
    elif down:
      self.h_upd = Downsample(
          channels,
          use_conv=False,
          dims=dims,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
      )
      self.x_upd = Downsample(
          channels,
          use_conv=False,
          dims=dims,
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
        (2 * self.out_channels) if use_scale_shift_norm else self.out_channels,
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
        dims,
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
          dims,
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
          dims,
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
    """Apply the block to a Tensor, conditioned on a timestep embedding."""
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
    while len(emb_out.shape) < len(h.shape):
      emb_out = jnp.expand_dims(emb_out, axis=1)
    if self.use_scale_shift_norm:
      scale, shift = jnp.split(emb_out, 2, axis=1)
      h = self.out_norm(h) * (1 + scale) + shift
      h = self.out_act(h)
      h = self.out_dropout(h, rngs=rngs)
      h = self.out_conv(h)
    else:
      h = h + emb_out
      h = self.out_norm(h)
      h = self.out_act(h)
      h = self.out_dropout(h, rngs=rngs)
      h = self.out_conv(h)

    return self.skip_connection(x) + h


class QKVAttention(nnx.Module):
  """A module which performs QKV attention and splits in a different order."""

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

    attn_dtype = jnp.bfloat16 if self.attn_implementation == "cudnn" else q.dtype
    a = jax.nn.dot_product_attention(
        q.astype(attn_dtype),
        k.astype(attn_dtype),
        v.astype(attn_dtype),
        scale=scale,
        implementation=self.attn_implementation,
    ).astype(q.dtype)
    a = a.reshape(bs, length, self.n_heads * head_dim)
    return a


class AttentionBlock(nnx.Module):
  """An attention block that allows spatial positions to attend to each other."""

  def __init__(
      self,
      channels: int,
      *,
      num_heads: int = 1,
      num_head_channels: int = -1,
      use_checkpoint: bool = False,
      attn_implementation: Optional[Literal["xla", "cudnn"]] = None,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    assert not use_checkpoint, "Checkpointing is not implemented."
    super().__init__()
    self.channels = channels
    if num_head_channels == -1:
      self.num_heads = num_heads
    else:
      assert channels % num_head_channels == 0
      self.num_heads = channels // num_head_channels
    self.use_checkpoint = use_checkpoint

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


class UNetModel(nnx.Module):
  """The full UNet model with attention and timestep embedding."""

  def __init__(
      self,
      *,
      image_size: int,
      in_channels: int,
      model_channels: int,
      out_channels: int,
      num_res_blocks: int,
      attention_resolutions: Tuple[int, ...],
      dropout: float = 0.0,
      channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
      time_embed_dim: Optional[int] = None,
      conv_resample: bool = True,
      dims: int = 2,
      use_checkpoint: bool = False,
      num_heads: int = 1,
      num_head_channels: int = -1,
      num_heads_upsample: int = -1,
      use_scale_shift_norm: bool = False,
      resblock_updown: bool = False,
      attn_implementation: Optional[Literal["xla", "cudnn"]] = None,
      dtype: Optional[jnp.dtype] = None,
      num_classes: Optional[int] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    if num_heads_upsample == -1:
      num_heads_upsample = num_heads

    self.dtype = dtype
    self.image_size = image_size
    self.in_channels = in_channels
    self.model_channels = model_channels
    self.out_channels = out_channels
    self.num_res_blocks = num_res_blocks
    self.attention_resolutions = attention_resolutions
    self.dropout = dropout
    self.channel_mult = channel_mult
    self.conv_resample = conv_resample
    self.use_checkpoint = use_checkpoint
    self.num_heads = num_heads
    self.num_head_channels = num_head_channels
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

    # Condition embedding
    self.num_classes = num_classes
    if self.num_classes is not None:
      self.label_emb = nnx.Embed(
          num_embeddings=self.num_classes, features=time_embed_dim, rngs=rngs
      )

    # Input blocks
    self.input_blocks = []
    input_block_chans = [model_channels]
    ch = int(channel_mult[0] * model_channels)
    ds = 1

    # First input block (just convolution)
    self.input_blocks.append(
        TimestepEmbedSequential(
            conv_nd(
                dims,
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
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
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
                  use_checkpoint=use_checkpoint,
                  num_heads=num_heads,
                  num_head_channels=num_head_channels,
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
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=True,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ) if resblock_updown else Downsample(
                    ch,
                    conv_resample,
                    dims=dims,
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
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
        AttentionBlock(
            ch,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            attn_implementation=attn_implementation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
        ResBlock(
            ch,
            time_embed_dim,
            dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        ),
    )

    # Output blocks
    self.output_blocks = []
    for level, mult in list(enumerate(channel_mult))[::-1]:
      for i in range(num_res_blocks + 1):
        ich = input_block_chans.pop()
        layers = [
            ResBlock(
                ch + ich,
                time_embed_dim,
                dropout,
                out_channels=int(model_channels * mult),
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
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
                  use_checkpoint=use_checkpoint,
                  num_heads=num_heads_upsample,
                  num_head_channels=num_head_channels,
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
                  dims=dims,
                  use_checkpoint=use_checkpoint,
                  use_scale_shift_norm=use_scale_shift_norm,
                  up=True,
                  dtype=dtype,
                  param_dtype=param_dtype,
                  rngs=rngs,
              ) if resblock_updown else Upsample(
                  ch,
                  conv_resample,
                  dims=dims,
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
            dims,
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
      *,
      y: Optional[jax.Array] = None,
      rngs: Optional[nnx.Rngs] = None,
  ) -> jax.Array:
    emb = self.time_embed(timestep_embedding(t, self.model_channels), rngs=rngs)
    if self.num_classes is not None:
      emb = emb + self.label_emb(jnp.squeeze(y))
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
    return self.out(h, rngs=rngs)


class UNetModelWrapper(UNetModel):
  """A wrapper for UNetModel that handles default channel multipliers based on image size."""

  @classmethod
  def create(
      cls,
      *,
      shape: Tuple[int, ...],
      model_channels: int,
      num_res_blocks: int,
      channel_mult: Union[Tuple[int, ...], None] = None,
      learn_sigma: bool = False,
      use_checkpoint: bool = False,
      attention_resolutions: Tuple[int, ...] = (16,),
      conv_resample: bool = True,
      num_heads: int = 1,
      num_head_channels: int = -1,
      num_heads_upsample: int = -1,
      use_scale_shift_norm: bool = False,
      dropout: float = 0.0,
      resblock_updown: bool = False,
      dtype: Optional[jnp.dtype] = None,
      param_dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs,
      **kwargs: Any,
  ) -> UNetModel:
    image_size = shape[0]
    if channel_mult is None:
      if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
      elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
      elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
      elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
      elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
      elif image_size == 28:
        channel_mult = (1, 2, 2)
      else:
        raise ValueError(f"Unsupported image size: {image_size}")
    else:
      channel_mult = tuple(channel_mult)

    attention_ds = []
    for res in attention_resolutions:
      attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=shape[-1],
        model_channels=model_channels,
        out_channels=(shape[-1] if not learn_sigma else shape[-1] * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=conv_resample,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
        **kwargs,
    )
