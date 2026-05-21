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
"""Sinkhorn kernels via JAX Pallas — scan-over-tiles design.

Each public function implements one Sinkhorn update step.  The cost
matrix is *never* materialised: a JAX ``lax.scan`` iterates over
source / target tiles in Python-space, and each scan step launches one
small Pallas kernel that processes a single ``[BM, BD] × [BN, BD]`` tile
pair.  This bounds the per-invocation GPU memory to ``O(BM × BD)``
bytes regardless of ``n`` or ``m``.

Pre-scaling convention
----------------------
All inputs are pre-scaled before the kernel call so that the kernel
body never receives a traced ``eps`` scalar (which would prevent
``float(eps)`` conversions and trigger re-compilation):

  c1_scaled  = cost_1 / sqrt(eps)    [n, BD]
  c2_scaled  = cost_2 / sqrt(eps)    [m, BD]
  f_scaled   = f / eps               [n]
  g_scaled   = g / eps               [m]
  bias_scaled = bias / eps           scalar

With these definitions the logit becomes:
  logit[j, i] = g_scaled[j] + f_scaled[i]
                - dot(c2_scaled[j], c1_scaled[i])    # = -C[i,j]/eps
                - bias_scaled
             = (g[j] + f[i] - C[i,j] - bias) / eps

Padding convention
------------------
- Source (key) rows beyond ``n`` get ``f_scaled = _F_PAD`` (a large
  negative constant).  This makes their logit ≈ ``_F_PAD`` ≪ 0, so
  ``exp(logit) ≈ 0`` and they contribute nothing to the logsumexp.
- Target (query) rows beyond ``m`` are handled by stripping the output
  after the scan (their carry values are never read).
- For ``apply_kernel_pallas``, target tiles can be partially valid; a
  per-step boolean mask is passed to set padding logits to ``-inf``.

Block-size selection
--------------------
``BD = next_power_of_2(max(d, 16))`` so that ``jnp.dot`` inside the
kernel lowers to ``tl.dot`` with a power-of-two K dimension (required
for tensor-core engagement).  ``BM = BN`` is chosen so the query tile
``[BM, BD]`` and the logit tile ``[BM, BN]`` together fit in 80 % of
the GPU's configurable shared memory (queried at runtime).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
from functools import lru_cache
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Lazy import: importing this module must never fail even when Pallas is
# absent (e.g. CPU-only JAX builds used during package installation).
def _get_pallas():
  try:
    from jax.experimental import pallas as pl
    return pl
  except ImportError as exc:
    raise ImportError(
        "JAX Pallas is required. Install via: pip install jax[cuda]"
    ) from exc


__all__ = ["apply_lse_kernel_pallas", "apply_kernel_pallas"]

# Padding sentinel for f_scaled: large enough that exp(logit_padded) = 0
# in float32 without overflowing, regardless of eps (never use float(eps)).
_F_PAD = jnp.float32(-1e9)


# ---------------------------------------------------------------------------
# GPU shared-memory query and block-size selection
# ---------------------------------------------------------------------------

_FALLBACK_SRAM = 96 * 1024  # V100 minimum in bytes


@lru_cache(maxsize=8)
def _libcuda():
  lib = ctypes.util.find_library("cuda")
  return ctypes.CDLL(lib) if lib else None


@lru_cache(maxsize=8)
def _gpu_sram_bytes(device_id: int = 0) -> int:
  """Max configurable shared memory per SM, queried at runtime.

  Tries Triton's driver (preferred), then the CUDA driver API, then
  falls back to 96 KB (V100 minimum).  Values: V100 96 KB, A100 164 KB,
  H100 228 KB, B200 ≥256 KB.
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
    lib = _libcuda()
    if lib is not None:
      val = ctypes.c_int(0)
      # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
      if lib.cuDeviceGetAttribute(ctypes.byref(val), 97, device_id) == 0:
        if val.value > 0:
          return int(val.value)
  except Exception:
    pass
  return _FALLBACK_SRAM


@lru_cache(maxsize=64)
def _choose_block_sizes(d: int, device_id: int = 0) -> Tuple[int, int, int]:
  """Return ``(BM, BN, BD)`` adapted to the GPU's shared memory.

  Constraints enforced:
  * ``BD = 2^ceil(log2(max(d, 16)))`` — power-of-two K for ``tl.dot``.
  * ``BM = BN`` — largest power-of-two ≥ 16 such that the query tile
    ``[BM, BD]`` plus the logit output ``[BM, BN]`` fit in SRAM budget:
    ``(BM * BD + BM * BM) * 4 bytes ≤ 0.8 × SRAM``.
  """
  BD = max(int(2 ** np.ceil(np.log2(max(d, 16)))), 16)
  budget = int(_gpu_sram_bytes(device_id) * 0.8)
  BM = 256
  while BM >= 16:
    if (BM * BD + BM * BM) * 4 <= budget:
      break
    BM //= 2
  return max(BM, 16), max(BM, 16), BD


def _pad_dim0(arr: jnp.ndarray, size: int, fill=0.0) -> jnp.ndarray:
  """Pad ``arr`` along axis 0 to ``size`` using ``fill``."""
  diff = size - arr.shape[0]
  if diff == 0:
    return arr
  return jnp.pad(arr, [(0, diff)] + [(0, 0)] * (arr.ndim - 1),
                 constant_values=fill)


# ---------------------------------------------------------------------------
# Pallas kernel bodies — one tile pair per invocation, no fori_loop
# ---------------------------------------------------------------------------

def _lse_kernel(
    # Inputs via BlockSpec (loaded by Pallas per grid invocation):
    c2s_ref,      # [BM, BD]  pre-scaled target (query) tile
    g_s_ref,      # [BM]      g / eps for this target tile
    # Inputs via BlockSpec (same for all invocations in this scan step):
    c1s_ref,      # [BN, BD]  pre-scaled source (key) tile
    f_s_ref,      # [BN]      f / eps for this source tile (or _F_PAD)
    vec_ref,      # [BN]      weight vector (zeros if not has_vec)
    bias_s_ref,   # [1]       bias / eps
    # Outputs via BlockSpec:
    m_out_ref,    # [BM]      partial max of logits
    ell_out_ref,  # [BM]      partial ∑ exp(logit - m)
    acc_out_ref,  # [BM]      partial ∑ exp(logit - m) * vec  (0 if not has_vec)
    *,
    has_vec: bool,  # Python bool, evaluated at JAX trace time → dead-code elim
):
  """Compute partial logsumexp for ONE (target-tile, source-tile) pair.

  The logit formula (with pre-scaled inputs) is:
    logit[j, i] = g_s[j] + f_s[i] - dot(c2s[j], c1s[i]) - bias_s
                = (g[j] + f[i] - C[i,j] - bias) / eps

  Output: partial (m, ell, acc) to be combined by the outer scan.
  """
  c2s = c2s_ref[...].astype(jnp.float32)    # [BM, BD]
  g_s = g_s_ref[...].astype(jnp.float32)    # [BM]
  c1s = c1s_ref[...].astype(jnp.float32)    # [BN, BD]
  f_s = f_s_ref[...].astype(jnp.float32)    # [BN]
  bs  = bias_s_ref[0].astype(jnp.float32)   # scalar

  # jnp.dot([BM, BD] @ [BD, BN]) — Pallas lowers to tl.dot with K=BD (power of 2).
  dot   = jnp.dot(c2s, c1s.T)                          # [BM, BN]
  logit = g_s[:, None] + f_s[None, :] - dot - bs       # [BM, BN]
  # Padded source entries have f_s ≈ _F_PAD ≪ 0 → logit ≪ 0 → exp ≈ 0.

  m     = logit.max(axis=1)                             # [BM]
  exp_s = jnp.exp(logit - m[:, None])                  # [BM, BN]
  ell   = exp_s.sum(axis=1)                             # [BM]

  # has_vec is a Python bool: the branch below is eliminated at trace time
  # when has_vec=False, so vec_ref is never loaded in that case.
  if has_vec:
    v   = vec_ref[...].astype(jnp.float32)             # [BN]
    acc = (exp_s * v[None, :]).sum(axis=1)              # [BM]
  else:
    acc = jnp.zeros_like(m)

  m_out_ref[...]   = m
  ell_out_ref[...] = ell
  acc_out_ref[...] = acc


def _ker_kernel(
    c1s_ref,       # [BN, BD]  pre-scaled source (query) tile
    c2s_ref,       # [BM, BD]  pre-scaled target (key) tile
    vec_ref,       # [BM]      target scaling vector for this tile
    bias_s_ref,    # [1]       bias / eps
    valid_ref,     # [BM]      bool: which target entries are real (not padding)
    m_out_ref,     # [BN]      partial max of logits
    acc_out_ref,   # [BN]      partial ∑ exp(logit - m) * vec
):
  """Compute partial K@v for ONE (source-tile, target-tile) pair.

  Logit: logit[i, j] = -(c1s[i] · c2s[j] + bias_s) = -C[i,j]/eps
  Padded target entries are masked to -inf via ``valid_ref``.
  """
  c1s   = c1s_ref[...].astype(jnp.float32)     # [BN, BD]
  c2s   = c2s_ref[...].astype(jnp.float32)     # [BM, BD]
  v     = vec_ref[...].astype(jnp.float32)      # [BM]
  bs    = bias_s_ref[0].astype(jnp.float32)     # scalar
  valid = valid_ref[...].astype(jnp.bool_)      # [BM]

  dot   = jnp.dot(c1s, c2s.T)                            # [BN, BM]
  logit = -(dot + bs)                                     # [BN, BM]
  # Mask out padded target entries so they don't inflate the running max.
  logit = jnp.where(valid[None, :], logit, -jnp.inf)     # [BN, BM]

  m     = logit.max(axis=1)                               # [BN]
  exp_s = jnp.exp(logit - m[:, None])                    # [BN, BM]
  acc   = (exp_s * v[None, :]).sum(axis=1)                # [BN]

  m_out_ref[...]   = m
  acc_out_ref[...] = acc


# ---------------------------------------------------------------------------
# Logsumexp combination rule (Flash-Attention online algorithm)
# ---------------------------------------------------------------------------

def _combine_lse(m_run, ell_run, acc_run, pm, pe, pa):
  """Combine running ``(m, ell, acc)`` with one tile's partial ``(pm, pe, pa)``.

  Online logsumexp:
    m_new = max(m_run, pm)
    ell_new = exp(m_run - m_new) * ell_run + exp(pm - m_new) * pe
    acc_new = exp(m_run - m_new) * acc_run + exp(pm - m_new) * pa
  """
  m_new   = jnp.maximum(m_run, pm)
  s_run   = jnp.exp(m_run - m_new)
  s_tile  = jnp.exp(pm  - m_new)
  ell_new = s_run * ell_run + s_tile * pe
  acc_new = s_run * acc_run + s_tile * pa
  return m_new, ell_new, acc_new


# ---------------------------------------------------------------------------
# Public: apply_lse_kernel_pallas
# ---------------------------------------------------------------------------

def apply_lse_kernel_pallas(
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
  r"""Fused logsumexp for Sinkhorn LSE updates.

  For ``axis=0`` computes, for every target point :math:`j`:

  .. math::

    \text{result}[j] =
    \varepsilon\,\mathrm{logsumexp}_i
    \!\bigl((f_i + g_j - C_{ij})/\varepsilon\bigr) - g_j

  where :math:`C = \text{cost\_1}\,\text{cost\_2}^\top + \text{bias}`.

  Args:
    cost_1: ``[n, d]``.
    cost_2: ``[m, d]``.
    f: source potentials ``[n]``.
    g: target potentials ``[m]``.
    eps: entropic regularisation (may be a traced JAX scalar).
    bias: constant cost offset.
    vec: optional weight vector for signed weighted logsumexp.
    axis: ``0`` → sum over source, result ``[m]``; ``1`` → vice versa.
    axis_name: JAX collective axis for multi-GPU ``pmax``/``psum``.

  Returns:
    ``(w_res, w_sgn)`` matching
    :meth:`~ott.geometry.geometry.Geometry.apply_lse_kernel`.
  """
  pl = _get_pallas()
  has_vec = vec is not None

  # Axis normalisation: after this point we always sum over cost_1 (source)
  # and produce a result indexed by cost_2 (target).
  if axis == 1:
    cost_1, cost_2 = cost_2, cost_1
    f, g = g, f

  n, d  = cost_1.shape
  m_dim = cost_2.shape[0]
  BM, BN, BD = _choose_block_sizes(d)

  # Pre-scale: absorb eps into all arrays so that the Pallas kernel body
  # contains no reference to the (possibly traced) scalar eps.
  sqrt_eps  = jnp.sqrt(eps)
  c1s       = (cost_1 / sqrt_eps).astype(jnp.float32)   # [n, BD] after pad
  c2s       = (cost_2 / sqrt_eps).astype(jnp.float32)   # [m, BD] after pad
  f_s       = (f / eps).astype(jnp.float32)              # [n]
  g_s       = (g / eps).astype(jnp.float32)              # [m]
  bias_s    = jnp.array([jnp.float32(bias) / jnp.float32(eps)], dtype=jnp.float32)
  # vec is NOT divided by eps (it is a multiplicative weight, not inside exp).
  vec_s = (vec if has_vec else jnp.zeros((n,), jnp.float32)).astype(jnp.float32)

  # Round up to multiples of BN/BM for the tiled reshape.
  S_pad = ((n + BN - 1) // BN) * BN     # padded source count
  T_pad = ((m_dim + BM - 1) // BM) * BM # padded target count

  # Pad source arrays along the first axis (extra rows) and second axis
  # (extra columns to reach BD).
  # Padded f_s rows use _F_PAD (very negative): logit_padded ≈ _F_PAD ≪ 0
  # so exp(logit_padded) ≈ 0 — they contribute nothing to the logsumexp.
  # Note: _F_PAD is a float32 constant; we do NOT divide by eps here because
  # eps may be a traced JAX value and float(traced) raises a concrete-value error.
  c1s_pad = jnp.pad(c1s, [(0, S_pad - n), (0, BD - d)])              # [S_pad, BD]
  f_s_pad = _pad_dim0(f_s,   S_pad, fill=float(_F_PAD))               # [S_pad]
  vec_pad = _pad_dim0(vec_s, S_pad, fill=0.0)                          # [S_pad]

  # Pad target arrays (outputs for padded rows will be discarded).
  c2s_pad = jnp.pad(c2s, [(0, T_pad - m_dim), (0, BD - d)])          # [T_pad, BD]
  g_s_pad = _pad_dim0(g_s, T_pad, fill=0.0)                            # [T_pad]

  # Reshape source into tiles: each scan step sees one [BN, BD] source tile.
  n_tiles   = S_pad // BN
  c1s_tiles = c1s_pad.reshape(n_tiles, BN, BD)  # [n_tiles, BN, BD]
  f_s_tiles = f_s_pad.reshape(n_tiles, BN)       # [n_tiles, BN]
  vec_tiles = vec_pad.reshape(n_tiles, BN)        # [n_tiles, BN]

  # Build the Pallas call once (outside scan_step) so the compiled kernel
  # is reused across all scan steps and across repeated calls.
  out_s = jax.ShapeDtypeStruct((T_pad,), jnp.float32)
  lse_pallas = pl.pallas_call(
      functools.partial(_lse_kernel, has_vec=has_vec),
      out_shape=[out_s, out_s, out_s],
      grid=(T_pad // BM,),
      in_specs=[
          # c2s_pad: one [BM, BD] query-tile per grid invocation.
          pl.BlockSpec(lambda j: (j, 0), (BM, BD)),
          # g_s_pad: one [BM] slice per grid invocation.
          pl.BlockSpec(lambda j: (j,),   (BM,)),
          # c1s_tile: ALL grid invocations see the SAME [BN, BD] source tile
          # (one tile per scan step, broadcast across the target grid).
          pl.BlockSpec(lambda j: (0, 0), (BN, BD)),
          # f_s_tile: same broadcast pattern.
          pl.BlockSpec(lambda j: (0,),   (BN,)),
          # vec_tile: same.
          pl.BlockSpec(lambda j: (0,),   (BN,)),
          # bias_s: scalar [1], shared.
          None,
      ],
      out_specs=[
          pl.BlockSpec(lambda j: (j,), (BM,)),
          pl.BlockSpec(lambda j: (j,), (BM,)),
          pl.BlockSpec(lambda j: (j,), (BM,)),
      ],
  )

  def scan_step(carry, tile_data):
    """One scan step: call Pallas for one source tile, combine with running stats."""
    m_run, ell_run, acc_run = carry
    c1s_tile, f_s_tile, vec_tile = tile_data
    pm, pe, pa = lse_pallas(c2s_pad, g_s_pad, c1s_tile, f_s_tile,
                             vec_tile, bias_s)
    return _combine_lse(m_run, ell_run, acc_run, pm, pe, pa), None

  init = (
      jnp.full((T_pad,), -jnp.inf, jnp.float32),  # running max
      jnp.zeros((T_pad,), jnp.float32),             # running ∑ exp(logit - m)
      jnp.zeros((T_pad,), jnp.float32),             # running ∑ exp(logit - m)*vec
  )
  (m_f, ell_f, acc_f), _ = jax.lax.scan(
      scan_step, init, (c1s_tiles, f_s_tiles, vec_tiles)
  )

  # Strip padding and recover original dtype.
  dtype  = cost_1.dtype
  m_out  = m_f[:m_dim].astype(dtype)
  ell_out = ell_f[:m_dim].astype(dtype)
  acc_out = acc_f[:m_dim].astype(dtype)
  # g_s_pad was g / eps for valid rows; recover g for the "remove" subtraction.
  g_out  = (g_s_pad[:m_dim] * eps).astype(dtype)
  remove = jnp.where(jnp.isfinite(g_out), g_out, 0.0)

  # Multi-GPU: each device ran the kernel on its local source shard, producing
  # partial (m, ell, acc).  Combine with two collective ops.
  if axis_name is not None:
    m_g    = jax.lax.pmax(m_out, axis_name=axis_name)
    scale  = jnp.exp(m_out - m_g)
    ell_g  = jax.lax.psum(ell_out * scale, axis_name=axis_name)
    acc_g  = jax.lax.psum(acc_out * scale, axis_name=axis_name)
  else:
    m_g, ell_g, acc_g = m_out, ell_out, acc_out

  if not has_vec:
    # Standard logsumexp: result[j] = eps * (m[j] + log(ell[j])) - g[j]
    w_res = eps * (m_g + jnp.log(ell_g)) - remove
    w_sgn = jnp.array([1.0], dtype=dtype)
  else:
    # Signed weighted logsumexp.
    # acc_g[j] = ∑_i exp(logit[j,i] - m[j]) * vec[i]
    # log|∑_i exp(logit[j,i]) * vec[i]| = m[j] + log|acc_g[j]|
    abs_acc = jnp.abs(acc_g)
    log_abs = jnp.where(abs_acc > 0.0, jnp.log(abs_acc), 0.0)
    w_res   = jnp.where(abs_acc > 0.0, eps * (m_g + log_abs), -jnp.inf)
    w_res   = w_res - remove
    w_sgn   = jnp.sign(acc_g).astype(dtype)

  return w_res, w_sgn


# ---------------------------------------------------------------------------
# Public: apply_kernel_pallas
# ---------------------------------------------------------------------------

def apply_kernel_pallas(
    cost_1: jnp.ndarray,
    cost_2: jnp.ndarray,
    vec: jnp.ndarray,
    eps: float,
    bias: float = 0.0,
    axis: int = 1,
    axis_name: Optional[str] = None,
) -> jnp.ndarray:
  r"""Fused kernel-vector product for Sinkhorn scaling updates.

  Computes :math:`(K\mathbf{v})[i] = \sum_j K_{ij}\,v_j` (``axis=1``)
  or the transpose (``axis=0``), where
  :math:`K_{ij} = \exp(-C_{ij}/\varepsilon)`.

  Args:
    cost_1: ``[n, d]``.
    cost_2: ``[m, d]``.
    vec: scaling vector, length ``m`` (``axis=1``) or ``n`` (``axis=0``).
    eps: entropic regularisation (may be a traced JAX scalar).
    bias: constant cost offset.
    axis: ``1`` → ``K @ vec``; ``0`` → ``K^T @ vec``.
    axis_name: JAX collective axis for multi-GPU aggregation.
  """
  pl = _get_pallas()

  # Axis normalisation: after this point cost_1 = source (query),
  # cost_2 = target (key), vec is over target.
  if axis == 0:
    cost_1, cost_2 = cost_2, cost_1

  n, d  = cost_1.shape
  m_dim = cost_2.shape[0]
  BM, BN, BD = _choose_block_sizes(d)

  sqrt_eps = jnp.sqrt(eps)
  c1s      = (cost_1 / sqrt_eps).astype(jnp.float32)
  c2s      = (cost_2 / sqrt_eps).astype(jnp.float32)
  vec_f    = vec.astype(jnp.float32)
  bias_s   = jnp.array([jnp.float32(bias) / jnp.float32(eps)], dtype=jnp.float32)

  N_pad = ((n + BN - 1) // BN) * BN       # padded source count
  M_pad = ((m_dim + BM - 1) // BM) * BM   # padded target count

  # Pad source along both axes; output rows beyond n are stripped at the end.
  c1s_pad = jnp.pad(c1s, [(0, N_pad - n), (0, BD - d)])          # [N_pad, BD]

  # Pad target along both axes; vec is padded with 0 (zero contribution).
  c2s_pad = jnp.pad(c2s, [(0, M_pad - m_dim), (0, BD - d)])      # [M_pad, BD]
  vec_pad = _pad_dim0(vec_f, M_pad, fill=0.0)                      # [M_pad]

  # Per-tile valid-entry count for the target dimension.
  # Most tiles have BM valid entries; the last may have fewer.
  # This information is passed to the kernel so it can mask padded target
  # entries to -inf (preventing them from inflating the running max).
  n_tiles = M_pad // BM
  tile_valid_counts = jnp.array(
      [min(BM, max(0, m_dim - i * BM)) for i in range(n_tiles)],
      dtype=jnp.int32,
  )  # [n_tiles]

  # Reshape target into tiles.
  c2s_tiles  = c2s_pad.reshape(n_tiles, BM, BD)  # [n_tiles, BM, BD]
  vec_tiles  = vec_pad.reshape(n_tiles, BM)        # [n_tiles, BM]
  # Boolean validity masks per tile: True where the target entry is real.
  valid_masks = (
      jnp.arange(BM)[None, :] < tile_valid_counts[:, None]
  ).astype(jnp.bool_)  # [n_tiles, BM]

  out_s = jax.ShapeDtypeStruct((N_pad,), jnp.float32)
  ker_pallas = pl.pallas_call(
      _ker_kernel,
      out_shape=[out_s, out_s],
      grid=(N_pad // BN,),
      in_specs=[
          # c1s_pad: one [BN, BD] source-tile per grid invocation.
          pl.BlockSpec(lambda i: (i, 0), (BN, BD)),
          # c2s_tile: ALL grid invocations see the SAME target tile.
          pl.BlockSpec(lambda i: (0, 0), (BM, BD)),
          # vec_tile: same.
          pl.BlockSpec(lambda i: (0,),   (BM,)),
          # bias_s: scalar [1].
          None,
          # valid_mask: [BM] bool, same for all source invocations.
          pl.BlockSpec(lambda i: (0,),   (BM,)),
      ],
      out_specs=[
          pl.BlockSpec(lambda i: (i,), (BN,)),
          pl.BlockSpec(lambda i: (i,), (BN,)),
      ],
  )

  def scan_step(carry, tile_data):
    """One scan step: call Pallas for one target tile, combine with running stats."""
    m_run, acc_run = carry
    c2s_tile, vec_tile, valid_mask = tile_data
    pm, pa = ker_pallas(c1s_pad, c2s_tile, vec_tile, bias_s, valid_mask)
    m_new   = jnp.maximum(m_run, pm)
    s_run   = jnp.exp(m_run - m_new)
    s_tile  = jnp.exp(pm   - m_new)
    acc_new = s_run * acc_run + s_tile * pa
    return (m_new, acc_new), None

  init = (
      jnp.full((N_pad,), -jnp.inf, jnp.float32),
      jnp.zeros((N_pad,), jnp.float32),
  )
  (m_f, acc_f), _ = jax.lax.scan(
      scan_step, init, (c2s_tiles, vec_tiles, valid_masks)
  )

  # Strip padding and recover original dtype.
  m_out   = m_f[:n].astype(cost_1.dtype)
  acc_out = acc_f[:n].astype(cost_1.dtype)

  # Multi-GPU combination.
  if axis_name is not None:
    m_g   = jax.lax.pmax(m_out, axis_name=axis_name)
    acc_g = jax.lax.psum(
        acc_out * jnp.exp(m_out - m_g), axis_name=axis_name
    )
  else:
    m_g, acc_g = m_out, acc_out

  # (K @ vec)[i] = exp(m[i]) * acc[i]
  return jnp.exp(m_g) * acc_g
