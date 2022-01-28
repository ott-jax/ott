# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Linear algebra utility methods for optimal transport of Gaussian mixtures."""

from typing import Callable, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp


def get_mean_and_var(
    points: jnp.ndarray,  # (n, d)
    weights: jnp.ndarray,  # (n,)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Get the mean and variance of a weighted set of points."""
  weights_sum = jnp.sum(weights, axis=-1)  # (1,)
  mean = (
      # matmul((1, n), (n, d)) -> (1, d)
      jnp.matmul(weights, points) / weights_sum)
  # center points
  centered = points - mean[None, :]  # (n, d) - (1, d)
  var = (
      # matmul((1, n), (n, d)) -> (1, d)
      jnp.matmul(weights, centered ** 2.) / weights_sum)
  return mean, var


def get_mean_and_cov(
    points: jnp.ndarray,  # (n, d)
    weights: jnp.ndarray,  # (n,)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Get the mean and covariance of a weighted set of points."""
  weights_sum = jnp.sum(weights, axis=-1, keepdims=True)  # (1,)
  mean = (
      # matmul((1, n), (n, d)) -> (1, d)
      jnp.matmul(weights, points) / weights_sum)
  # center points
  centered = points - mean[None, :]  # (n, d) - (1, d)
  cov = (
      jnp.matmul(
          # (1, n)           (d, n)
          weights[None, :] * jnp.swapaxes(centered, axis1=-2, axis2=-1),
          # (n, d)
          centered) /
      weights_sum)
  return mean, cov


def flat_to_tril(x: jnp.ndarray, size: int) -> jnp.ndarray:
  """Map flat values to lower triangular matrices.

  Args:
    x: flat values
    size: size of lower triangular matrices. x should have shape
      (..., size(size+1)/2), and the final matrices should have shape
      (..., size, size).

  Returns:
    Lower triangular matrices.
  """
  m = jnp.zeros(tuple(list(x.shape[:-1]) + [size, size]), dtype=x.dtype)
  tril = jnp.tril_indices(size)
  return m.at[..., tril[0], tril[1]].set(x)


def tril_to_flat(m: jnp.ndarray) -> jnp.ndarray:
  """Flatten lower triangular matrices.

  Args:
    m: lower triangular matrices of shape (..., size, size)

  Returns:
    A vector of shape (..., size (size+1) // 2)
  """
  size = m.shape[-1]
  tril = jnp.tril_indices(size)
  return m[..., tril[0], tril[1]]


def apply_to_diag(
    m: jnp.ndarray,
    fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
  """Apply a function to the diagonal of a matrix."""
  size = m.shape[-1]
  diag = jnp.diagonal(m, axis1=-2, axis2=-1)
  ind = jnp.arange(size, dtype=jnp.int32)
  return m.at[..., ind, ind].set(fn(diag))


def matrix_powers(
    m: jnp.ndarray,
    powers: Iterable[float],
) -> List[jnp.ndarray]:
  """Raise a real, symmetric matrix to multiple powers."""
  eigs, q = jnp.linalg.eigh(m)
  qt = jnp.swapaxes(q, axis1=-2, axis2=-1)
  ret = []
  for power in powers:
    ret.append(jnp.matmul(jnp.expand_dims(eigs ** power, -2) * q, qt))
  return ret


def invmatvectril(
    m: jnp.ndarray,
    x: jnp.ndarray,
    lower: bool = True
) -> jnp.ndarray:
  """Multiply x by the inverse of a triangular matrix.

  Args:
    m: triangular matrix, shape (d, d)
    x: array of points, shape (n, d)
    lower: if True, m is lower triangular; otherwise m is upper triangular

  Returns:
    m^{-1} x
  """
  return jnp.transpose(
      jax.scipy.linalg.solve_triangular(
          m, jnp.transpose(x), lower=lower))


def get_random_orthogonal(
    key: jnp.ndarray,
    dim: int,
    dtype: Optional[jnp.dtype] = None
) -> jnp.ndarray:
  """Get a random orthogonal matrix with the specified dimension."""
  m = jax.random.normal(
      key=key,
      shape=[dim, dim],
      dtype=dtype)
  q, _ = jnp.linalg.qr(m)
  return q
