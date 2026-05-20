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
"""Tests for Pallas Sinkhorn kernels (GPU only).

Run with::

    pytest tests/geometry/pallas_kernels_test.py -v

Tests are skipped automatically when Pallas is unavailable or no GPU is
present.
"""
import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import costs, geometry, low_rank, pointcloud

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


def _pallas_available() -> bool:
  try:
    import jax.experimental.pallas as pl  # noqa: F401

    from ott.geometry._pallas_kernels import (  # noqa: F401
        apply_kernel_pallas,
        apply_lse_kernel_pallas,
    )

    # Actually trigger compilation to check GPU support.
    return jax.default_backend() == "gpu"
  except Exception:
    return False


pytestmark = pytest.mark.skipif(
    not _pallas_available(),
    reason="Pallas not available or no GPU detected",
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_lrc(rng, n, m, r, eps=0.1, bias=0.0):
  rngs = jax.random.split(rng, 4)
  c1 = jax.random.normal(rngs[0], (n, r))
  c2 = jax.random.normal(rngs[1], (m, r))
  f = jax.random.normal(rngs[2], (n,))
  g = jax.random.normal(rngs[3], (m,))
  lrc = low_rank.LRCGeometry(c1, c2, bias=bias, epsilon=eps)
  return lrc, f, g


def _ref_lse(lrc, f, g, eps, vec=None, axis=0):
  """Dense reference: materialise cost matrix, then logsumexp."""
  C = lrc.cost_matrix  # [n, m]
  bi = lrc.bias
  if axis == 0:
    # sum over source i, result for target j
    logit = (f[:, None] + g[None, :] - C - bi) / eps  # [n, m]
    if vec is None:
      from jax.scipy.special import logsumexp
      lse = logsumexp(logit, axis=0)  # [m]
      return eps * lse - g, jnp.array([1.0])
    else:
      from ott.math import utils as mu
      res, sgn = mu.logsumexp(logit, b=vec[:, None].T, axis=0, return_sign=True)
      # wait, vec is [n], sum over i
      # manual: for each j, logsumexp_i(logit[:,j], b=vec)
      import jax.numpy as jnp

      def one_j(logit_col):  # [n]
        from ott.math import utils as mu
        r, s = mu.logsumexp(logit_col, b=vec, return_sign=True)
        return r, s

      res_j, sgn_j = jax.vmap(one_j)(logit.T)  # vmap over j
      return eps * res_j - jnp.where(jnp.isfinite(g), g, 0.0), sgn_j
  else:
    logit = (f[:, None] + g[None, :] - C - bi) / eps  # [n, m]
    if vec is None:
      from jax.scipy.special import logsumexp
      lse = logsumexp(logit, axis=1)  # [n]
      return eps * lse - f, jnp.array([1.0])
    else:
      import jax.numpy as jnp

      def one_i(logit_row):  # [m]
        from ott.math import utils as mu
        r, s = mu.logsumexp(logit_row, b=vec, return_sign=True)
        return r, s

      res_i, sgn_i = jax.vmap(one_i)(logit)  # vmap over i
      return eps * res_i - jnp.where(jnp.isfinite(f), f, 0.0), sgn_i


def _ref_kernel(lrc, vec, eps, axis=1):
  K = jnp.exp(-lrc.cost_matrix / eps)
  return K @ vec if axis == 1 else K.T @ vec


# ---------------------------------------------------------------------------
# apply_lse_kernel_pallas
# ---------------------------------------------------------------------------


class TestApplyLSEKernelPallas:

  @pytest.mark.parametrize("axis", [0, 1])
  @pytest.mark.parametrize("r", [32, 64, 100])
  def test_lse_no_vec_matches_dense(self, rng: jax.Array, axis: int, r: int):
    """apply_lse_kernel_pallas (vec=None) matches dense reference."""
    from ott.geometry._pallas_kernels import apply_lse_kernel_pallas
    n, m, eps = 128, 192, 0.1
    lrc, f, g = _make_lrc(rng, n, m, r, eps)

    ref, _ = _ref_lse(lrc, f, g, eps, vec=None, axis=axis)
    got, sgn = apply_lse_kernel_pallas(
        lrc.cost_1, lrc.cost_2, f, g, eps, lrc.bias, None, axis
    )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )
    assert float(sgn[0]) == pytest.approx(1.0)

  @pytest.mark.parametrize("axis", [0, 1])
  def test_lse_with_vec_matches_dense(self, rng: jax.Array, axis: int):
    """Signed weighted logsumexp matches dense reference."""
    from ott.geometry._pallas_kernels import apply_lse_kernel_pallas
    n, m, r, eps = 128, 192, 64, 0.1
    lrc, f, g = _make_lrc(rng, n, m, r, eps)
    rngs = jax.random.split(rng, 2)
    # positive vec — sign should be +1 everywhere
    vec_src = jnp.abs(jax.random.normal(rngs[0], (n,))) + 0.1
    vec_tgt = jnp.abs(jax.random.normal(rngs[1], (m,))) + 0.1
    vec = vec_src if axis == 0 else vec_tgt

    ref, ref_sgn = _ref_lse(lrc, f, g, eps, vec=vec, axis=axis)
    got, got_sgn = apply_lse_kernel_pallas(
        lrc.cost_1, lrc.cost_2, f, g, eps, lrc.bias, vec, axis
    )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_array_equal(
        np.sign(np.asarray(got_sgn)), np.sign(np.asarray(ref_sgn))
    )

  def test_lse_nonmultiple_sizes(self, rng: jax.Array):
    """Padding handles n, m not divisible by block size."""
    from ott.geometry._pallas_kernels import apply_lse_kernel_pallas
    n, m, r, eps = 100, 150, 37, 0.1  # 100 % 64 != 0
    lrc, f, g = _make_lrc(rng, n, m, r, eps)

    ref, _ = _ref_lse(lrc, f, g, eps, vec=None, axis=0)
    got, _ = apply_lse_kernel_pallas(
        lrc.cost_1, lrc.cost_2, f, g, eps, lrc.bias, None, 0, BM=64, BN=64
    )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )

  def test_lse_nonzero_bias(self, rng: jax.Array):
    """Constant bias in LRCGeometry is handled correctly."""
    from ott.geometry._pallas_kernels import apply_lse_kernel_pallas
    n, m, r, eps = 128, 128, 64, 0.1
    lrc, f, g = _make_lrc(rng, n, m, r, eps, bias=0.5)

    ref, _ = _ref_lse(lrc, f, g, eps, vec=None, axis=0)
    got, _ = apply_lse_kernel_pallas(
        lrc.cost_1, lrc.cost_2, f, g, eps, lrc.bias, None, 0
    )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )

  @pytest.mark.parametrize("axis", [0, 1])
  def test_lrcgeometry_apply_lse_kernel(self, rng: jax.Array, axis: int):
    """LRCGeometry.apply_lse_kernel auto-uses Pallas and matches dense."""
    n, m, r, eps = 128, 192, 64, 0.1
    lrc, f, g = _make_lrc(rng, n, m, r, eps)

    ref_geom = geometry.Geometry(lrc.cost_matrix, epsilon=eps)
    ref, _ = ref_geom.apply_lse_kernel(f, g, eps, axis=axis)
    got, _ = lrc.apply_lse_kernel(f, g, eps, axis=axis)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )

  @pytest.mark.parametrize("axis", [0, 1])
  def test_pointcloud_negdotp_apply_lse_kernel(self, rng: jax.Array, axis: int):
    """PointCloud (NegDotProduct, online) delegates to Pallas."""
    n, m, d, eps = 128, 192, 64, 0.1
    rngs = jax.random.split(rng, 4)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d))
    f = jax.random.normal(rngs[2], (n,))
    g = jax.random.normal(rngs[3], (m,))

    pc_ref = pointcloud.PointCloud(
        x, y, cost_fn=costs.NegDotProduct(), batch_size=None, epsilon=eps
    )
    ref, _ = pc_ref.apply_lse_kernel(f, g, eps, axis=axis)

    pc_pallas = pointcloud.PointCloud(
        x, y, cost_fn=costs.NegDotProduct(), batch_size=n, epsilon=eps
    )
    got, _ = pc_pallas.apply_lse_kernel(f, g, eps, axis=axis)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# apply_kernel_pallas
# ---------------------------------------------------------------------------


class TestApplyKernelPallas:

  @pytest.mark.parametrize("axis", [0, 1])
  @pytest.mark.parametrize("r", [32, 64, 100])
  def test_kernel_matches_dense(self, rng: jax.Array, axis: int, r: int):
    """apply_kernel_pallas matches dense K @ vec."""
    from ott.geometry._pallas_kernels import apply_kernel_pallas
    n, m, eps = 128, 192, 0.1
    lrc, _, _ = _make_lrc(rng, n, m, r, eps)
    rngs = jax.random.split(rng, 1)
    vec_size = n if axis == 0 else m
    vec = jax.random.normal(rngs[0], (vec_size,))

    ref = _ref_kernel(lrc, vec, eps, axis)
    got = apply_kernel_pallas(lrc.cost_1, lrc.cost_2, vec, eps, lrc.bias, axis)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )

  def test_kernel_nonmultiple_sizes(self, rng: jax.Array):
    """Masking + padding handles non-divisible sizes correctly."""
    from ott.geometry._pallas_kernels import apply_kernel_pallas
    n, m, r, eps = 100, 150, 37, 0.1
    lrc, _, _ = _make_lrc(rng, n, m, r, eps)
    rngs = jax.random.split(rng, 1)
    vec = jax.random.normal(rngs[0], (m,))

    ref = _ref_kernel(lrc, vec, eps, axis=1)
    got = apply_kernel_pallas(
        lrc.cost_1, lrc.cost_2, vec, eps, lrc.bias, axis=1, BN=64, BM=64
    )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )

  def test_kernel_nonzero_bias(self, rng: jax.Array):
    from ott.geometry._pallas_kernels import apply_kernel_pallas
    n, m, r, eps = 128, 128, 64, 0.1
    lrc, _, _ = _make_lrc(rng, n, m, r, eps, bias=0.5)
    rngs = jax.random.split(rng, 1)
    vec = jax.random.normal(rngs[0], (m,))

    ref = _ref_kernel(lrc, vec, eps, axis=1)
    got = apply_kernel_pallas(
        lrc.cost_1, lrc.cost_2, vec, eps, lrc.bias, axis=1
    )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )

  @pytest.mark.parametrize("axis", [0, 1])
  def test_lrcgeometry_apply_kernel(self, rng: jax.Array, axis: int):
    """LRCGeometry.apply_kernel auto-uses Pallas."""
    n, m, r, eps = 128, 192, 64, 0.1
    lrc, _, _ = _make_lrc(rng, n, m, r, eps)
    rngs = jax.random.split(rng, 1)
    vec_size = n if axis == 0 else m
    vec = jax.random.normal(rngs[0], (vec_size,))

    ref = _ref_kernel(lrc, vec, eps, axis)
    got = lrc.apply_kernel(vec, eps=eps, axis=axis)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-4, atol=1e-4
    )

  @pytest.mark.parametrize("axis", [0, 1])
  def test_pointcloud_negdotp_apply_kernel(self, rng: jax.Array, axis: int):
    """PointCloud (NegDotProduct, online) delegates apply_kernel to Pallas."""
    n, m, d, eps = 128, 192, 64, 0.1
    rngs = jax.random.split(rng, 2)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d))
    vec_size = n if axis == 0 else m
    vec = jax.random.normal(jax.random.key(99), (vec_size,))

    pc_ref = pointcloud.PointCloud(
        x, y, cost_fn=costs.NegDotProduct(), batch_size=None, epsilon=eps
    )
    ref = pc_ref.apply_kernel(vec, eps=eps, axis=axis)

    pc_pallas = pointcloud.PointCloud(
        x, y, cost_fn=costs.NegDotProduct(), batch_size=n, epsilon=eps
    )
    got = pc_pallas.apply_kernel(vec, eps=eps, axis=axis)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Sinkhorn end-to-end
# ---------------------------------------------------------------------------


class TestSinkhornPallas:

  @pytest.mark.parametrize("lse_mode", [True, False])
  def test_sinkhorn_matches_reference(self, rng: jax.Array, lse_mode: bool):
    """Full Sinkhorn solve via Pallas matches the dense reference."""
    from ott.solvers import linear

    n, m, d, eps = 128, 192, 64, 0.1
    rngs = jax.random.split(rng, 2)
    x = jax.random.normal(rngs[0], (n, d))
    y = jax.random.normal(rngs[1], (m, d))

    geom_ref = pointcloud.PointCloud(
        x, y, cost_fn=costs.NegDotProduct(), batch_size=None, epsilon=eps
    )
    geom_pal = pointcloud.PointCloud(
        x, y, cost_fn=costs.NegDotProduct(), batch_size=n, epsilon=eps
    )

    out_ref = linear.solve(geom_ref, lse_mode=lse_mode)
    out_pal = linear.solve(geom_pal, lse_mode=lse_mode)

    np.testing.assert_allclose(
        float(out_pal.reg_ot_cost),
        float(out_ref.reg_ot_cost),
        rtol=1e-3,
        atol=1e-3,
    )
