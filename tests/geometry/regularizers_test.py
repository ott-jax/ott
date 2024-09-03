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

import lineax as lx

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from ott.geometry import regularizers


def _proj(matrix: jnp.ndarray, nu: float = 1.0) -> jnp.ndarray:
  assert nu > 0.0, nu
  u, _, v_h = jnp.linalg.svd(matrix, full_matrices=False)
  return u.dot(v_h) * jnp.sqrt(nu)


class TestProximalOperator:
  D = 19

  @pytest.mark.parametrize("tau", [0.7, 1.0, 3.31])
  @pytest.mark.parametrize(("reg", "lam"), [(regularizers.L1(), 0.142),
                                            (regularizers.SqL2(), 2.1)])
  def test_moreau_envelope(
      self,
      rng: jax.Array,
      tau: float,
      reg: regularizers.ProximalOperator,
      lam: Optional[float],
  ):
    tol = 1e-5
    x = jax.random.normal(rng, (32, self.D))
    if lam is not None:
      reg = regularizers.PostComposition(reg, alpha=lam)

    actual = jax.vmap(jax.grad(lambda x: reg.moreau_envelope(x, tau)))
    expected = jax.vmap(lambda x: (1.0 / tau) * (x - reg.prox(x, tau)))

    np.testing.assert_allclose(expected(x), actual(x), rtol=tol, atol=tol)

  @pytest.mark.parametrize(("rho", "use_a", "tau"), [(1.0, False, 0.2),
                                                     (2.1, True, 1.0)])
  def test_regularization(
      self, rng: jax.Array, rho: float, use_a: bool, tau: float
  ):
    rng_x, rng_a, rng_moreau = jax.random.split(rng, 3)
    x = jax.random.normal(rng_x, (self.D,))
    a = jax.random.normal(rng_a, (self.D,)) if use_a else jnp.zeros(self.D)

    l1 = regularizers.L1()
    reg = regularizers.Regularization(l1, a=a if use_a else None, rho=rho)

    expected = l1(x) + (0.5 * rho) * jnp.sum((x - a) ** 2)
    actual = reg(x)

    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)
    self.test_moreau_envelope(rng_moreau, tau=tau, reg=reg, lam=None)

  @pytest.mark.parametrize(("nu", "use_b", "reg", "lam"),
                           [(0.25, True, regularizers.SqL2(), 2.1),
                            (1.0, False, regularizers.L1(), 1.3)])
  def test_orthogonal(
      self,
      rng: jax.Array,
      nu: float,
      use_b: bool,
      reg: regularizers.ProximalOperator,
      lam: float,
  ):
    k = 12
    rng_x, rng_A, rng_b, rng_moreau = jax.random.split(rng, 4)
    x = jax.random.normal(rng_x, (self.D,))

    reg = regularizers.PostComposition(reg, alpha=lam)
    A = _proj(jax.random.normal(rng_A, (k, self.D)), nu=nu)
    b = jax.random.normal(rng_b, (k,)) if use_b else jnp.zeros(k)
    orth = regularizers.Orthogonal(reg, A=A, b=b if use_b else None, nu=nu)

    expected = orth(x)
    actual = reg(A @ x + b)

    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)
    assert orth.is_fully_orthogonal == (nu == 1.0)
    for tau in [0.1, 0.5, 1.0]:
      self.test_moreau_envelope(rng_moreau, tau=tau, reg=orth, lam=None)


class TestQuadratic:

  @pytest.mark.parametrize(
      "is_orthogonal", [False, True], ids=["noorth", "orth"]
  )
  @pytest.mark.parametrize(
      "is_complement", [False, True], ids=["nocomp", "comp"]
  )
  @pytest.mark.parametrize("is_factor", [False, True], ids=["nofac", "fac"])
  def test_quad_properties(
      self,
      rng: jax.Array,
      is_orthogonal: bool,
      is_complement: bool,
      is_factor: bool,
  ):

    def loss(reg: regularizers.ProximalOperator, x: jnp.ndarray) -> float:
      return jnp.mean(jax.vmap(reg)(x))

    def test_properties(reg: regularizers.ProximalOperator) -> None:
      assert reg.is_orthogonal == is_orthogonal
      assert reg.is_complement == is_complement
      assert reg.is_factor == is_factor
      if reg.is_complement:
        assert isinstance(reg.A_comp, lx.AbstractLinearOperator)
      else:
        assert reg.A_comp is None

    is_square = not is_factor and not is_complement
    k, d = (17, 17) if is_square else (5, 17)
    rng_A, rng_x = jax.random.split(rng, 2)
    A = jax.random.normal(rng_A, (k, d))
    x = jax.random.normal(rng_x, (13, d))
    if is_orthogonal:
      A = _proj(A)

    reg = regularizers.Quadratic(
        A,
        is_orthogonal=is_orthogonal,
        is_complement=is_complement,
        is_factor=is_factor,
    )
    grad_reg = jax.jit(jax.grad(loss))(reg, x)
    grad_A = grad_reg.A.as_matrix()

    test_properties(reg)
    test_properties(grad_reg)

    np.testing.assert_array_equal(jnp.isfinite(grad_A), True)
    with pytest.raises(AssertionError, match="Not equal"):
      # check that the gradients are not close to 0
      np.testing.assert_allclose(grad_A, 0.0, rtol=1e-3, atol=1e-3)

  @pytest.mark.parametrize("d", [17, 64])
  def test_pythagorean_identity(self, rng: jax.Array, d: int):
    rng_A, rng_x = jax.random.split(rng, 2)

    A = _proj(jax.random.normal(rng_A, (d // 2, d)))
    x = jax.random.normal(rng_x, (d,))

    reg = regularizers.Quadratic(A=A, is_factor=True)
    reg_c = regularizers.Quadratic(A=A, is_complement=True, is_factor=True)

    expected = 0.5 * (x ** 2).sum()
    actual = reg(x) + reg_c(x)

    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize("is_orthogonal", [False, True])
  @pytest.mark.parametrize(("is_complement", "use_b"), [(False, False),
                                                        (True, False),
                                                        (True, True)])
  @pytest.mark.parametrize("tau", [0.1, 1.0, 10.0])
  def test_orth_use_b(
      self,
      rng: jax.Array,
      is_orthogonal: bool,
      is_complement: bool,
      use_b: bool,
      tau: float,
  ):
    d = 31
    is_factor = not is_complement
    rng_A, rng_b, rng_x = jax.random.split(rng, 3)

    A = jax.random.normal(rng_A, (d // 3, d))
    if is_orthogonal:
      A = _proj(A)
    x = jax.random.normal(rng_x, (d,))
    b = jax.random.normal(rng_b, (d,)) * 10.0 + 100.0 if use_b else jnp.zeros(d)

    reg_orth = regularizers.Quadratic(
        A,
        b=b if use_b else None,
        is_factor=is_factor,
        is_complement=is_complement,
        is_orthogonal=is_orthogonal,
    )

    iden = lx.IdentityLinearOperator(reg_orth.Q.out_structure())
    A, y = iden + tau * reg_orth.Q, x - tau * b
    expected = lx.linear_solve(A, y).value
    actual = reg_orth.prox(x, tau)

    np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-3)

  @pytest.mark.parametrize("tau", [0.05, 1.0, 2.1])
  @pytest.mark.parametrize("use_b", [False, True])
  def test_Q_is_identity(self, rng: jax.Array, use_b: bool, tau: float):
    d = 7
    rng_b, rng_x = jax.random.split(rng)
    x = jax.random.normal(rng_x, (d,))
    b = jax.random.normal(rng_b, (d,)) if use_b else jnp.zeros(d)
    reg = regularizers.Quadratic(b=b if use_b else None)

    expected_norm = 0.5 * (x ** 2).sum() + jnp.dot(x, b)
    expected_prox = (1.0 / (1.0 + tau)) * x - (tau / (1.0 + tau)) * b

    assert not reg.is_factor
    assert not reg.is_complement
    np.testing.assert_allclose(expected_norm, reg(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        expected_prox, reg.prox(x, tau), rtol=1e-5, atol=1e-5
    )

  @pytest.mark.parametrize("lam", [0.2, 7.1])
  @pytest.mark.parametrize("is_complement", [False, True])
  def test_l2(self, rng: jax.Array, lam: float, is_complement: bool):
    k, d = 7, 12
    rng_A, rng_x = jax.random.split(rng, 2)
    A = jax.random.normal(rng_A, (k, d))
    x = jax.random.normal(rng_x, (d,))

    l2 = regularizers.SqL2(A, is_complement=is_complement)
    l2 = regularizers.PostComposition(l2, alpha=lam)
    reg = l2.f.f

    A_ = (reg.A_comp if is_complement else reg.A).as_matrix()
    expected_norm1 = 0.5 * lam * jnp.dot(A_ @ x, A_ @ x)
    expected_norm2 = 0.5 * lam * jnp.dot(x, reg.Q.as_matrix() @ x)

    assert reg.A.as_matrix().shape == (k, d)
    assert reg.Q.as_matrix().shape == (d, d)
    assert reg.is_factor
    assert reg.is_complement == is_complement
    if is_complement:
      assert reg.A_comp.as_matrix().shape == (d, d)

    np.testing.assert_allclose(expected_norm1, l2(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(expected_norm2, l2(x), rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize(("lam", "tau"), [(0.5, 1.0), (11.2, 0.1)])
  @pytest.mark.parametrize(("is_complement", "is_orthogonal"), [(False, False),
                                                                (True, False),
                                                                (True, True)])
  def test_l2_moreau_envelope(
      self, rng: jax.Array, lam: float, tau: float, is_complement: bool,
      is_orthogonal: bool
  ):
    n, k, d = 21, 5, 15
    rng_A, rng_x = jax.random.split(rng, 2)
    A = jax.random.normal(rng_A, (k, d))
    if is_orthogonal:
      A = _proj(A)
    x = jax.random.normal(rng_x, (n, d))

    reg = regularizers.SqL2(
        A, is_complement=is_complement, is_orthogonal=is_orthogonal
    )
    reg = regularizers.PostComposition(reg, alpha=lam)

    actual = jax.vmap(jax.grad(lambda x: reg.moreau_envelope(x, tau)))
    expected = jax.vmap(lambda x: (1.0 / tau) * (x - reg.prox(x, tau)))

    np.testing.assert_allclose(expected(x), actual(x), rtol=1e-4, atol=1e-4)

  @pytest.mark.parametrize(("is_complement", "is_orthogonal"), [(False, True),
                                                                (True, False),
                                                                (True, True)])
  @pytest.mark.parametrize("tau", [0.05, 1.0, 13.0])
  def test_matrix_inversion_lemma(
      self, rng: jax.Array, is_complement: bool, is_orthogonal: bool, tau: float
  ):
    d = 33
    rng_A, rng_b, rng_x = jax.random.split(rng, 3)
    A = jax.random.normal(rng_A, (7, d))
    if is_orthogonal:
      A = _proj(A)
    b = jax.random.normal(rng_b, (d,))
    x = jax.random.normal(rng_x, (d,))

    reg = regularizers.Quadratic(
        A,
        b=b,
        is_orthogonal=is_orthogonal,
        is_factor=True,
        is_complement=is_complement,
    )
    iden = lx.IdentityLinearOperator(reg.Q.out_structure())

    expected = lx.linear_solve(iden + tau * reg.Q, x - tau * b).value
    actual = reg.prox(x, tau)

    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)


class TestSqKOverlap:

  @pytest.mark.parametrize("d", [32, 64, 126])
  def test_matches_l1(self, rng: jax.Array, d: int):
    x = jax.random.normal(rng, (d,))
    sq_kovp = regularizers.SqKOverlap(k=1)

    expected = 0.5 * jnp.linalg.norm(x, ord=1) ** 2
    actual = sq_kovp(x)

    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("d", [4, 8, 16])
  def test_matches_l2(self, rng: jax.Array, d: int):
    x = jax.random.normal(rng, (d,))
    l2 = regularizers.SqL2()
    sq_kovp = regularizers.SqKOverlap(k=d)

    expected = 0.5 * jnp.linalg.norm(x, ord=2) ** 2
    actual = sq_kovp(x)

    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        l2.prox(x), sq_kovp.prox(x), rtol=1e-5, atol=1e-5
    )
