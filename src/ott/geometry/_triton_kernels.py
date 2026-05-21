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
"""Flash-style Sinkhorn kernels via Triton.

Two fused GPU kernels for :class:`~ott.geometry.low_rank.LRCGeometry`:

* :func:`apply_lse_kernel_triton` — fused row-wise logsumexp (both
  ``vec=None`` and signed weighted ``vec!=None`` cases).
* :func:`apply_kernel_triton` — kernel-vector product ``K @ v``.

Both kernels use ``tl.dot`` with ``BD = next_power_of_2(max(d, 16))``
so tensor cores are engaged regardless of ``d``.  Block sizes ``BM`` and
``BN`` are chosen adaptively to fill the GPU's shared memory (queried at
runtime so the same code runs on V100, A100, H100, B200, and consumer
GPUs without any manual configuration).

JAX integration uses :func:`jax.pure_callback` (works on every JAX
version).  On GPU, JAX passes DLPack device arrays so no host copy
occurs.  When inputs are sharded, the callback runs once per device on
the local shard; the explicit ``pmax``/``psum`` handle cross-device
aggregation.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "apply_lse_kernel_triton",
    "apply_kernel_triton",
]

_FALLBACK_SRAM = 96 * 1024  # V100 minimum


# ---------------------------------------------------------------------------
# Triton kernel definitions
# ---------------------------------------------------------------------------

def _get_triton():
  try:
    import triton
    import triton.language as tl
    return triton, tl
  except ImportError as exc:
    raise ImportError("pip install triton") from exc


def _build_triton_kernels():
  triton, tl = _get_triton()

  @triton.jit
  def sinkhorn_lse(
      c2s_ptr, gs_ptr,
      c1s_ptr, fs_ptr,
      vec_ptr,          # [N]  weights; ignored when HAS_VEC=False
      bias_s,
      out_m_ptr,        # [M]  running max (partial logsumexp)
      out_l_ptr,        # [M]  running sum  (unweighted)
      out_acc_ptr,      # [M]  running sum  (weighted, only when HAS_VEC=True)
      M, N, D,
      stride_c2m, stride_c2d,
      stride_c1n, stride_c1d,
      BM: tl.constexpr,
      BN: tl.constexpr,
      BD: tl.constexpr,
      HAS_VEC: tl.constexpr,  # compile-time flag; eliminates acc when False
  ):
    j0    = tl.program_id(0) * BM
    j_off = j0 + tl.arange(0, BM)
    d_off = tl.arange(0, BD)
    # Load query tile once; stays in SRAM for the full inner loop.
    c2 = tl.load(
        c2s_ptr + j_off[:, None] * stride_c2m + d_off[None, :] * stride_c2d,
        mask=(j_off[:, None] < M) & (d_off[None, :] < D), other=0.,
    )
    g = tl.load(gs_ptr + j_off, mask=j_off < M, other=0.)

    m   = tl.full((BM,), float("-inf"), tl.float32)
    l   = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM,), tl.float32)   # live only when HAS_VEC=True

    for i0 in range(0, N, BN):
      i_off = i0 + tl.arange(0, BN)
      c1 = tl.load(
          c1s_ptr + i_off[:, None] * stride_c1n + d_off[None, :] * stride_c1d,
          mask=(i_off[:, None] < N) & (d_off[None, :] < D), other=0.,
      )
      f = tl.load(fs_ptr + i_off, mask=i_off < N, other=float("-inf"))

      # tl.dot engages tensor cores; [BM, BN] logit stays in registers.
      dot   = tl.dot(c2, tl.trans(c1), out_dtype=tl.float32)   # [BM, BN]
      logit = g[:, None] + f[None, :] - dot - bias_s
      logit = tl.where(i_off[None, :] < N, logit, float("-inf"))

      m_new = tl.maximum(m, tl.max(logit, axis=1))              # [BM]
      scale = tl.exp(m - m_new)
      exp_s = tl.exp(logit - m_new[:, None])                    # [BM, BN]
      l     = scale * l + tl.sum(exp_s, axis=1)

      if HAS_VEC:
        v   = tl.load(vec_ptr + i_off, mask=i_off < N, other=0.)
        acc = scale * acc + tl.sum(exp_s * v[None, :], axis=1)

      m = m_new

    tl.store(out_m_ptr   + j_off, m,   mask=j_off < M)
    tl.store(out_l_ptr   + j_off, l,   mask=j_off < M)
    tl.store(out_acc_ptr + j_off, acc, mask=j_off < M)

  @triton.jit
  def sinkhorn_ker(
      c1s_ptr, c2s_ptr, vec_ptr,
      bias_s,
      out_m_ptr, out_acc_ptr,
      M, N, D,
      stride_c1n, stride_c1d,
      stride_c2m, stride_c2d,
      BN: tl.constexpr,
      BM: tl.constexpr,
      BD: tl.constexpr,
  ):
    i0    = tl.program_id(0) * BN
    i_off = i0 + tl.arange(0, BN)
    d_off = tl.arange(0, BD)
    c1 = tl.load(
        c1s_ptr + i_off[:, None] * stride_c1n + d_off[None, :] * stride_c1d,
        mask=(i_off[:, None] < N) & (d_off[None, :] < D), other=0.,
    )
    m   = tl.full((BN,), float("-inf"), tl.float32)
    acc = tl.zeros((BN,), tl.float32)
    for j0 in range(0, M, BM):
      j_off = j0 + tl.arange(0, BM)
      c2 = tl.load(
          c2s_ptr + j_off[:, None] * stride_c2m + d_off[None, :] * stride_c2d,
          mask=(j_off[:, None] < M) & (d_off[None, :] < D), other=0.,
      )
      v = tl.load(vec_ptr + j_off, mask=j_off < M, other=0.)
      dot   = tl.dot(c1, tl.trans(c2), out_dtype=tl.float32)
      logit = -(dot + bias_s)
      logit = tl.where(j_off[None, :] < M, logit, float("-inf"))
      m_new = tl.maximum(m, tl.max(logit, axis=1))
      scale = tl.exp(m - m_new)
      exp_s = tl.exp(logit - m_new[:, None])
      acc   = scale * acc + tl.sum(exp_s * v[None, :], axis=1)
      m     = m_new
    tl.store(out_m_ptr   + i_off, m,   mask=i_off < N)
    tl.store(out_acc_ptr + i_off, acc, mask=i_off < N)

  return sinkhorn_lse, sinkhorn_ker


_KERNELS: dict = {}


def _triton_kernels():
  if not _KERNELS:
    lse, ker = _build_triton_kernels()
    _KERNELS["lse"] = lse
    _KERNELS["ker"] = ker
  return _KERNELS["lse"], _KERNELS["ker"]


# ---------------------------------------------------------------------------
# Adaptive block-size selection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _libcuda_handle():
  lib = ctypes.util.find_library("cuda")
  return ctypes.CDLL(lib) if lib else None


@lru_cache(maxsize=8)
def _gpu_sram_bytes(device_id: int = 0) -> int:
  """Query max configurable shared memory per block at runtime.

  Tries Triton's driver, then the CUDA driver API, then falls back to
  96 KB (V100 minimum).

  Typical values:  V100 96 KB · A100 164 KB · H100 228 KB · B200 ≥256 KB.
  """
  try:
    import triton.runtime.driver as _trd
    props = _trd.active.get_device_properties(device_id)
    for key in ("max_shared_mem", "shared_memory_per_block",
                "smem_per_block", "smem"):
      if key in props:
        return int(props[key])
  except Exception:
    pass
  try:
    lib = _libcuda_handle()
    if lib is not None:
      val = ctypes.c_int(0)
      if lib.cuDeviceGetAttribute(ctypes.byref(val), 97, device_id) == 0:
        if val.value > 0:
          return int(val.value)
  except Exception:
    pass
  return _FALLBACK_SRAM


@lru_cache(maxsize=64)
def _choose_block_sizes(d: int, device_id: int = 0) -> Tuple[int, int, int]:
  """Return ``(BM, BN, BD)`` tuned to the actual GPU's shared memory.

  * ``BD = next_power_of_2(max(d, 16))`` — tensor-core K-dimension.
  * ``BM = BN`` is the largest power-of-2 ≥ 16 such that the
    **query tile** ``[BM, BD]`` and the **logit output** ``[BM, BN]``
    fit in 80 % of the GPU's shared memory:
    ``(BM * BD + BM * BM) * 4 bytes ≤ 0.8 × SRAM``.
    (The key tile ``[BN, BD]`` is streamed per iteration by ``tl.dot``'s
    internal tiling and need not be counted separately.)

  Result is cached per ``(d, device_id)``; the SRAM query runs once.
  """
  triton, _ = _get_triton()
  BD     = max(triton.next_power_of_2(d), 16)
  budget = int(_gpu_sram_bytes(device_id) * 0.8)
  BM = 256
  while BM >= 16:
    # c2 tile [BM, BD] + logit output [BM, BM]
    if (BM * BD + BM * BM) * 4 <= budget:
      break
    BM //= 2
  BM = max(BM, 16)
  return BM, BM, BD


# ---------------------------------------------------------------------------
# JAX integration via pure_callback
# ---------------------------------------------------------------------------

def _to_torch(arr):
  import torch
  try:
    return torch.from_dlpack(arr)
  except Exception:
    return torch.as_tensor(np.asarray(arr)).cuda()


def _lse_callback(c2_s, g_s, c1_s, f_s, vec_s, bias_s_arr, BM, BN, BD,
                  has_vec: bool):
  """Per-device Triton call; returns partial ``(m, l, acc)``."""
  import torch
  triton, _ = _get_triton()
  lse_k, _ = _triton_kernels()

  c2t  = _to_torch(c2_s)
  gt   = _to_torch(g_s)
  c1t  = _to_torch(c1_s)
  ft   = _to_torch(f_s)
  vect = _to_torch(vec_s)
  bs   = float(_to_torch(bias_s_arr)[0])

  M, D = c2t.shape
  N    = c1t.shape[0]
  bm   = min(BM, triton.next_power_of_2(M))
  bn   = min(BN, triton.next_power_of_2(N))

  out_m   = torch.empty(M, dtype=torch.float32, device=c2t.device)
  out_l   = torch.empty(M, dtype=torch.float32, device=c2t.device)
  out_acc = torch.empty(M, dtype=torch.float32, device=c2t.device)

  lse_k[(triton.cdiv(M, bm),)](
      c2t, gt, c1t, ft, vect, bs,
      out_m, out_l, out_acc,
      M, N, D,
      int(c2t.stride(0)), int(c2t.stride(1)),
      int(c1t.stride(0)), int(c1t.stride(1)),
      BM=bm, BN=bn, BD=BD, HAS_VEC=has_vec,
  )
  try:
    return out_m, out_l, out_acc
  except Exception:
    return out_m.cpu().numpy(), out_l.cpu().numpy(), out_acc.cpu().numpy()


def _ker_callback(c1_s, c2_s, vec, bias_s_arr, BN, BM, BD):
  import torch
  triton, _ = _get_triton()
  _, ker_k = _triton_kernels()

  c1t  = _to_torch(c1_s)
  c2t  = _to_torch(c2_s)
  vect = _to_torch(vec)
  bs   = float(_to_torch(bias_s_arr)[0])

  N, D = c1t.shape
  M    = c2t.shape[0]
  bn   = min(BN, triton.next_power_of_2(N))
  bm   = min(BM, triton.next_power_of_2(M))

  out_m   = torch.empty(N, dtype=torch.float32, device=c1t.device)
  out_acc = torch.empty(N, dtype=torch.float32, device=c1t.device)

  ker_k[(triton.cdiv(N, bn),)](
      c1t, c2t, vect, bs,
      out_m, out_acc,
      M, N, D,
      int(c1t.stride(0)), int(c1t.stride(1)),
      int(c2t.stride(0)), int(c2t.stride(1)),
      BN=bn, BM=bm, BD=BD,
  )
  try:
    return out_m, out_acc
  except Exception:
    return out_m.cpu().numpy(), out_acc.cpu().numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lse_kernel_triton(
    cost_1: jnp.ndarray,
    cost_2: jnp.ndarray,
    f: jnp.ndarray,
    g: jnp.ndarray,
    eps: float,
    bias: float = 0.0,
    vec: Optional[jnp.ndarray] = None,
    axis: int = 0,
    axis_name: Optional[str] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  r"""Fused logsumexp via Triton (both ``vec=None`` and ``vec!=None``).

  For ``axis=0`` computes, for every target point :math:`j`:

  * ``vec=None``:
    :math:`\text{result}[j] = \varepsilon\,
    \mathrm{logsumexp}_i\!\bigl((f_i+g_j-C_{ij})/\varepsilon\bigr) - g_j`
  * ``vec!=None``:
    :math:`\text{result}[j] = \varepsilon\,
    \log\!\bigl|\sum_i \exp\!\bigl((f_i+g_j-C_{ij})/\varepsilon\bigr)
    v_i\bigr| - g_j`,
    with :math:`\text{sign}[j] = \mathrm{sign}(\cdot)`.

  Block sizes are chosen adaptively from the GPU's shared memory.

  Args:
    cost_1: ``[n, d]`` LRC factor.
    cost_2: ``[m, d]`` LRC factor.
    f: source potentials ``[n]``.
    g: target potentials ``[m]``.
    eps: entropic regularisation.
    bias: constant cost offset.
    vec: optional weight vector ``[n]`` (source) for ``axis=0``.
    axis: ``0`` → result ``[m]``; ``1`` → result ``[n]``.
    axis_name: JAX collective axis for multi-GPU aggregation.

  Returns:
    ``(w_res, w_sgn)`` matching
    :meth:`~ott.geometry.geometry.Geometry.apply_lse_kernel`.
  """
  has_vec = vec is not None
  if axis == 1:
    cost_1, cost_2 = cost_2, cost_1
    f, g = g, f

  n, d  = cost_1.shape
  m_dim = cost_2.shape[0]
  BM, BN, BD = _choose_block_sizes(d)

  sqrt_eps = jnp.sqrt(eps)
  c1_s   = (cost_1 / sqrt_eps).astype(jnp.float32)
  c2_s   = (cost_2 / sqrt_eps).astype(jnp.float32)
  f_s    = (f / eps).astype(jnp.float32)
  g_s    = (g / eps).astype(jnp.float32)
  bias_s = jnp.array([bias / eps], dtype=jnp.float32)
  # vec is NOT pre-scaled (it's a multiplicative weight, not inside exp).
  vec_s  = (vec if has_vec else jnp.zeros((n,), dtype=cost_1.dtype)
            ).astype(jnp.float32)

  out_s = jax.ShapeDtypeStruct((m_dim,), jnp.float32)

  partial_m, partial_l, partial_acc = jax.pure_callback(
      lambda c2, gs, c1, fs, vs, bs: _lse_callback(
          c2, gs, c1, fs, vs, bs, BM, BN, BD, has_vec
      ),
      (out_s, out_s, out_s),
      c2_s, g_s, c1_s, f_s, vec_s, bias_s,
      vectorized=False,
  )

  # Multi-GPU: combine partial statistics with two tiny collectives.
  # Both l and acc share the same rescaling factor exp(m_k - m_global).
  if axis_name is not None:
    m_g   = jax.lax.pmax(partial_m, axis_name=axis_name)
    scale = jnp.exp(partial_m - m_g)
    l_g   = jax.lax.psum(partial_l   * scale, axis_name=axis_name)
    acc_g = jax.lax.psum(partial_acc * scale, axis_name=axis_name)
  else:
    m_g, l_g, acc_g = partial_m, partial_l, partial_acc

  dtype  = cost_1.dtype
  g_orig = (g_s * eps).astype(dtype)
  remove = jnp.where(jnp.isfinite(g_orig), g_orig, 0.0)

  if not has_vec:
    w_res = eps * (m_g + jnp.log(l_g)).astype(dtype) - remove
    w_sgn = jnp.array([1.0], dtype=dtype)
  else:
    # acc_g[j] = sum_i exp(logit[j,i] - m_g[j]) * vec[i]
    # log|sum exp(logit) * vec| = m_g + log|acc_g|
    abs_acc = jnp.abs(acc_g)
    log_abs = jnp.where(abs_acc > 0.0, jnp.log(abs_acc), 0.0)
    w_res   = jnp.where(abs_acc > 0.0, eps * (m_g + log_abs), -jnp.inf)
    w_res   = w_res.astype(dtype) - remove
    w_sgn   = jnp.sign(acc_g).astype(dtype)

  return w_res, w_sgn


def apply_kernel_triton(
    cost_1: jnp.ndarray,
    cost_2: jnp.ndarray,
    vec: jnp.ndarray,
    eps: float,
    bias: float = 0.0,
    axis: int = 1,
    axis_name: Optional[str] = None,
) -> jnp.ndarray:
  r"""Fused kernel-vector product via Triton.

  Computes :math:`(K\mathbf{v})[i]=\sum_j K_{ij}v_j` (``axis=1``) or
  the transpose (``axis=0``).

  Args:
    cost_1: ``[n, d]``.
    cost_2: ``[m, d]``.
    vec: scaling vector.
    eps: entropic regularisation.
    bias: constant cost offset.
    axis: ``1`` → ``K @ vec``; ``0`` → ``K^T @ vec``.
    axis_name: JAX collective axis for multi-GPU aggregation.
  """
  if axis == 0:
    cost_1, cost_2 = cost_2, cost_1

  n, d = cost_1.shape
  BM, BN, BD = _choose_block_sizes(d)

  sqrt_eps = jnp.sqrt(eps)
  c1_s   = (cost_1 / sqrt_eps).astype(jnp.float32)
  c2_s   = (cost_2 / sqrt_eps).astype(jnp.float32)
  vec_f  = vec.astype(jnp.float32)
  bias_s = jnp.array([bias / eps], dtype=jnp.float32)

  out_s = jax.ShapeDtypeStruct((n,), jnp.float32)

  partial_m, partial_acc = jax.pure_callback(
      lambda c1, c2, v, bs: _ker_callback(c1, c2, v, bs, BN, BM, BD),
      (out_s, out_s),
      c1_s, c2_s, vec_f, bias_s,
      vectorized=False,
  )

  if axis_name is not None:
    m_g   = jax.lax.pmax(partial_m, axis_name=axis_name)
    acc_g = jax.lax.psum(
        partial_acc * jnp.exp(partial_m - m_g), axis_name=axis_name
    )
  else:
    m_g, acc_g = partial_m, partial_acc

  return (jnp.exp(m_g) * acc_g).astype(cost_1.dtype)
