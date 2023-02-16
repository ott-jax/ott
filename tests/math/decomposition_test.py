from typing import Any

import pytest

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from ott.math import decomposition

sksparse = pytest.importorskip("sksparse")


@pytest.mark.fast
class TestDecomposition:

  def test_dense_cholesky_solver(self, rng: jnp.ndarray):
    N = 10
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    A = jax.random.normal(keys[0], (N, N))
    b = jax.random.normal(keys[1], (N,))
    B = jnp.dot(A, A.transpose())
    solver = decomposition.DenseCholeskySolver(B)
    L = solver.L
    assert L is not None
    assert jnp.allclose(L, jnp.tril(L))
    assert jnp.allclose(B, jnp.dot(L, L.transpose()))
    assert jnp.allclose(solver.A, B)
    solution = solver.solve(b)
    assert jnp.allclose(jnp.dot(L, solution), b)

  @pytest.mark.parametrize("sparse_format", [sparse.BCOO.fromdense])
  def test_sparse_cholesky_solver(self, rng: jnp.ndarray, sparse_format: Any):
    N = 10
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    A = jax.random.normal(keys[0], (N, N), dtype=jnp.float64)
    b = jax.random.normal(keys[1], (N,))
    B = jnp.dot(A, A.transpose())
    B = sparse_format(B)
    solver = decomposition.SparseCholeskySolver(B)
    L = solver.L
    assert L is None
    out = list(solver._FACTOR_CACHE.values())[0]  # get cholmod.Factor
    assert jnp.allclose(out.apply_P(b), b)
    solution = solver.solve(b)
    assert jnp.allclose(out.solve_A(b), solution)
