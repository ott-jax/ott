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

# Lint as: python3
"""Tests for the differentiability of reg_ot_cost w.r.t weights/locations."""
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import sinkhorn
from ott.geometry import geometry, pointcloud


class TestSinkhornJacobian:

  @pytest.mark.fast.with_args(
      "lse_mode,shape_data",
      [(True, (7, 9)), (False, (11, 5))],
      only_fast=0,
  )
  def test_autograd_sinkhorn(
      self, rng: jnp.ndarray, lse_mode: bool, shape_data: Tuple[int, int]
  ):
    """Test gradient w.r.t. probability weights."""
    n, m = shape_data
    d = 3
    eps = 1e-3  # perturbation magnitude
    keys = jax.random.split(rng, 5)
    x = jax.random.uniform(keys[0], (n, d))
    y = jax.random.uniform(keys[1], (m, d))
    a = jax.random.uniform(keys[2], (n,)) + eps
    b = jax.random.uniform(keys[3], (m,)) + eps
    # Adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    def reg_ot(a, b):
      return sinkhorn.sinkhorn(
          pointcloud.PointCloud(x, y, epsilon=0.1), a=a, b=b, lse_mode=lse_mode
      ).reg_ot_cost

    reg_ot_and_grad = jax.jit(jax.value_and_grad(reg_ot))
    _, grad_reg_ot = reg_ot_and_grad(a, b)
    delta = jax.random.uniform(keys[4], (n,))
    delta = delta * (a > 0)  # ensures only perturbing non-zero coords.
    delta = delta - jnp.sum(delta) / jnp.sum(a > 0)  # center perturbation
    delta = delta * (a > 0)  # ensures only perturbing non-zero coords.
    reg_ot_delta_plus = reg_ot(a + eps * delta, b)
    reg_ot_delta_minus = reg_ot(a - eps * delta, b)
    delta_dot_grad = jnp.nansum(delta * grad_reg_ot)

    assert not jnp.any(jnp.isnan(delta_dot_grad))
    np.testing.assert_allclose(
        delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
        rtol=1e-03,
        atol=1e-02
    )

  @pytest.mark.parametrize(
      "lse_mode,shape_data", [(True, (7, 9)), (False, (11, 5))]
  )
  def test_gradient_sinkhorn_geometry(
      self, rng: jnp.ndarray, lse_mode: bool, shape_data: Tuple[int, int]
  ):
    """Test gradient w.r.t. cost matrix."""
    n, m = shape_data
    keys = jax.random.split(rng, 2)
    cost_matrix = jnp.abs(jax.random.normal(keys[0], (n, m)))
    delta = jax.random.normal(keys[1], (n, m))
    delta = delta / jnp.sqrt(jnp.vdot(delta, delta))
    eps = 1e-3  # perturbation magnitude

    def loss_fn(cm):
      a = jnp.ones(cm.shape[0]) / cm.shape[0]
      b = jnp.ones(cm.shape[1]) / cm.shape[1]
      geom = geometry.Geometry(cm, epsilon=0.5)
      out = sinkhorn.sinkhorn(geom, a, b, lse_mode=lse_mode)
      return out.reg_ot_cost, (geom, out.f, out.g)

    # first calculation of gradient
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    (loss_value, aux), grad_loss = loss_and_grad(cost_matrix)
    custom_grad = jnp.sum(delta * grad_loss)

    assert not jnp.isnan(loss_value)
    np.testing.assert_array_equal(grad_loss.shape, cost_matrix.shape)
    np.testing.assert_array_equal(jnp.isnan(grad_loss), False)

    # second calculation of gradient
    transport_matrix = aux[0].transport_from_potentials(aux[1], aux[2])
    grad_x = transport_matrix
    other_grad = jnp.sum(delta * grad_x)

    # third calculation of gradient
    loss_delta_plus, _ = loss_fn(cost_matrix + eps * delta)
    loss_delta_minus, _ = loss_fn(cost_matrix - eps * delta)
    finite_diff_grad = (loss_delta_plus - loss_delta_minus) / (2 * eps)

    np.testing.assert_allclose(custom_grad, other_grad, rtol=1e-02, atol=1e-02)
    np.testing.assert_allclose(
        custom_grad, finite_diff_grad, rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        other_grad, finite_diff_grad, rtol=1e-02, atol=1e-02
    )
    np.testing.assert_array_equal(jnp.isnan(custom_grad), False)

  @pytest.mark.fast.with_args(
      "lse_mode,implicit_differentiation,min_iter,max_iter,epsilon",
      [
          (True, True, 0, 2000, 1e-3),
          (True, True, 1000, 1000, 1e-3),
          (True, False, 1000, 1000, 1e-2),
          (True, False, 0, 2000, 1e-2),
          (False, True, 0, 2000, 1e-2),
      ],
      ids=[
          "lse-implicit", "lse-implicit-force_scan", "lse-backprop-force_scan",
          "lse-backprop", "scan-implicit"
      ],
      only_fast=[0, 1],
  )
  def test_gradient_sinkhorn_euclidean(
      self,
      rng: jnp.ndarray,
      lse_mode: bool,
      implicit_differentiation: bool,
      min_iter: int,
      max_iter: int,
      epsilon: float,
  ):
    """Test gradient w.r.t. locations x of reg-ot-cost."""
    # TODO(cuturi): ensure scaling mode works with backprop.
    d = 3
    n, m = 11, 13
    keys = jax.random.split(rng, 4)
    x = jax.random.normal(keys[0], (n, d)) / 10
    y = jax.random.normal(keys[1], (m, d)) / 10

    a = jax.random.uniform(keys[2], (n,))
    b = jax.random.uniform(keys[3], (m,))
    # Adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    def loss_fn(x, y):
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
      out = sinkhorn.sinkhorn(
          geom,
          a,
          b,
          lse_mode=lse_mode,
          implicit_differentiation=implicit_differentiation,
          min_iterations=min_iter,
          max_iterations=max_iter,
          jit=False
      )
      return out.reg_ot_cost, (geom, out.f, out.g)

    delta = jax.random.normal(keys[0], (n, d))
    delta = delta / jnp.sqrt(jnp.vdot(delta, delta))
    eps = 1e-5  # perturbation magnitude

    # first calculation of gradient
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_value, aux), grad_loss = loss_and_grad(x, y)
    custom_grad = jnp.sum(delta * grad_loss)

    assert not jnp.isnan(loss_value)
    np.testing.assert_array_equal(grad_loss.shape, x.shape)
    np.testing.assert_array_equal(jnp.isnan(grad_loss), False)
    # second calculation of gradient
    tm = aux[0].transport_from_potentials(aux[1], aux[2])
    tmp = 2 * tm[:, :, None] * (x[:, None, :] - y[None, :, :])
    grad_x = jnp.sum(tmp, 1)
    other_grad = jnp.sum(delta * grad_x)

    # third calculation of gradient
    loss_delta_plus, _ = loss_fn(x + eps * delta, y)
    loss_delta_minus, _ = loss_fn(x - eps * delta, y)
    finite_diff_grad = (loss_delta_plus - loss_delta_minus) / (2 * eps)

    np.testing.assert_allclose(custom_grad, other_grad, rtol=1e-02, atol=1e-02)
    np.testing.assert_allclose(
        custom_grad, finite_diff_grad, rtol=1e-02, atol=1e-02
    )
    np.testing.assert_allclose(
        other_grad, finite_diff_grad, rtol=1e-02, atol=1e-02
    )
    np.testing.assert_array_equal(jnp.isnan(custom_grad), False)

  def test_autoepsilon_differentiability(self, rng: jnp.ndarray):
    cost = jax.random.uniform(rng, (15, 17))

    def reg_ot_cost(c: jnp.ndarray):
      geom = geometry.Geometry(c, epsilon=None)  # auto epsilon
      return sinkhorn.sinkhorn(geom).reg_ot_cost

    gradient = jax.grad(reg_ot_cost)(cost)
    np.testing.assert_array_equal(jnp.isnan(gradient), False)

  def test_differentiability_with_jit(self, rng: jnp.ndarray):
    cost = jax.random.uniform(rng, (15, 17))

    def reg_ot_cost(c: jnp.ndarray):
      geom = geometry.Geometry(c, epsilon=1e-2)
      return sinkhorn.sinkhorn(geom, jit=True).reg_ot_cost

    gradient = jax.grad(reg_ot_cost)(cost)
    np.testing.assert_array_equal(jnp.isnan(gradient), False)
