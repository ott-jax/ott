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
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ott.core import sinkhorn
from ott.geometry import geometry, grid, pointcloud
from ott.tools import transport


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


@pytest.mark.fast
class TestSinkhornGradGrid:

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_diff_sinkhorn_x_grid_x_perturbation(
      self, rng: jnp.ndarray, lse_mode: bool
  ):
    """Test gradient w.r.t. probability weights."""
    eps = 1e-3  # perturbation magnitude
    keys = jax.random.split(rng, 6)
    x = (
        jnp.array([.0, 1.0]), jnp.array([.3, .4,
                                         .7]), jnp.array([1.0, 1.3, 2.4, 3.7])
    )
    grid_size = tuple(xs.shape[0] for xs in x)
    a = jax.random.uniform(keys[0], grid_size) + 1.0
    b = jax.random.uniform(keys[1], grid_size) + 1.0
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)

    def reg_ot(x):
      geom = grid.Grid(x=x, epsilon=1.0)
      return sinkhorn.sinkhorn(
          geom, a=a, b=b, threshold=0.1, lse_mode=lse_mode
      ).reg_ot_cost

    reg_ot_and_grad = jax.value_and_grad(reg_ot)
    _, grad_reg_ot = reg_ot_and_grad(x)
    delta = [jax.random.uniform(keys[i], (g,)) for i, g in enumerate(grid_size)]

    x_p_delta = [(xs + eps * delt) for xs, delt in zip(x, delta)]
    x_m_delta = [(xs - eps * delt) for xs, delt in zip(x, delta)]

    # center perturbation
    reg_ot_delta_plus = reg_ot(x_p_delta)
    reg_ot_delta_minus = reg_ot(x_m_delta)
    delta_dot_grad = jnp.sum(
        jnp.array([
            jnp.sum(delt * gr, axis=None)
            for delt, gr in zip(delta, grad_reg_ot)
        ])
    )
    np.testing.assert_allclose(
        delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
        rtol=1e-03,
        atol=1e-02
    )

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_diff_sinkhorn_x_grid_weights_perturbation(
      self, rng: jnp.ndarray, lse_mode: bool
  ):
    """Test gradient w.r.t. probability weights."""
    eps = 1e-4  # perturbation magnitude
    keys = jax.random.split(rng, 3)
    # yapf: disable
    x = (
        jnp.asarray([.0, 1.0]),
        jnp.asarray([.3, .4, .7]),
        jnp.asarray([1.0, 1.3, 2.4, 3.7])
    )
    # yapf: enable
    grid_size = tuple(xs.shape[0] for xs in x)
    a = jax.random.uniform(keys[0], grid_size) + 1
    b = jax.random.uniform(keys[1], grid_size) + 1
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)
    geom = grid.Grid(x=x, epsilon=1)

    def reg_ot(a, b):
      return sinkhorn.sinkhorn(
          geom, a=a, b=b, threshold=0.001, lse_mode=lse_mode
      ).reg_ot_cost

    reg_ot_and_grad = jax.value_and_grad(reg_ot)
    _, grad_reg_ot = reg_ot_and_grad(a, b)
    delta = jax.random.uniform(keys[2], grid_size).ravel()
    delta = delta - jnp.mean(delta)

    # center perturbation
    reg_ot_delta_plus = reg_ot(a + eps * delta, b)
    reg_ot_delta_minus = reg_ot(a - eps * delta, b)
    delta_dot_grad = jnp.sum(delta * grad_reg_ot)
    np.testing.assert_allclose(
        delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
        rtol=1e-03,
        atol=1e-02
    )


class TestSinkhornJacobianPreconditioning:

  @pytest.mark.fast.with_args(
      lse_mode=[True, False],
      tau_a=[1.0, .94],
      tau_b=[1.0, .91],
      shape=[(18, 19), (27, 18), (275, 414)],
      arg=[0, 1],
      only_fast=[0, -1],
  )
  def test_potential_jacobian_sinkhorn(
      self, rng: jnp.ndarray, lse_mode: bool, tau_a: float, tau_b: float,
      shape: Tuple[int, int], arg: int
  ):
    """Test Jacobian of optimal potential w.r.t. weights and locations."""
    n, m = shape
    dim = 3
    rngs = jax.random.split(rng, 7)
    x = jax.random.uniform(rngs[0], (n, dim))
    y = jax.random.uniform(rngs[1], (m, dim))
    a = jax.random.uniform(rngs[2], (n,)) + .2
    b = jax.random.uniform(rngs[3], (m,)) + .2
    a = a / (0.5 * n) if tau_a < 1.0 else a / jnp.sum(a)
    b = b / (0.5 * m) if tau_b < 1.0 else b / jnp.sum(b)
    random_dir = jax.random.uniform(rngs[4], (n,)) / n
    # center projection direction so that < potential , random_dir>
    # is invariant w.r.t additive shifts.
    random_dir = random_dir - jnp.mean(random_dir)
    delta_a = jax.random.uniform(rngs[5], (n,))
    if tau_a == 1.0:
      delta_a = delta_a - jnp.mean(delta_a)
    delta_x = jax.random.uniform(rngs[6], (n, dim))

    # As expected, lse_mode False has a harder time with small epsilon when
    # differentiating.
    epsilon = 0.01 if lse_mode else 0.1

    def loss_from_potential(
        a, x, precondition_fun=None, linear_solve_kwargs=None
    ):
      if linear_solve_kwargs is None:
        linear_solve_kwargs = {}
      out = transport.solve(
          x,
          y,
          epsilon=epsilon,
          a=a,
          b=b,
          tau_a=tau_a,
          tau_b=tau_b,
          lse_mode=lse_mode,
          precondition_fun=precondition_fun,
          **linear_solve_kwargs
      )
      return jnp.sum(random_dir * out.solver_output.f)

    # Compute implicit gradient
    loss_imp_no_precond = jax.jit(
        jax.value_and_grad(
            functools.partial(
                loss_from_potential,
                precondition_fun=lambda x: x,
                linear_solve_kwargs={'implicit_solver_symmetric': True}
            ),
            argnums=arg
        )
    )

    loss_imp_log_precond = jax.jit(
        jax.value_and_grad(loss_from_potential, argnums=arg)
    )

    _, g_imp_np = loss_imp_no_precond(a, x)
    imp_dif_np = jnp.sum(g_imp_np * (delta_a if arg == 0 else delta_x))

    _, g_imp_lp = loss_imp_log_precond(a, x)
    imp_dif_lp = jnp.sum(g_imp_lp * (delta_a if arg == 0 else delta_x))

    # Compute finite difference
    perturb_scale = 1e-4
    a_p = a + perturb_scale * delta_a if arg == 0 else a
    x_p = x if arg == 0 else x + perturb_scale * delta_x
    a_m = a - perturb_scale * delta_a if arg == 0 else a
    x_m = x if arg == 0 else x - perturb_scale * delta_x

    val_p, _ = loss_imp_no_precond(a_p, x_p)
    val_m, _ = loss_imp_no_precond(a_m, x_m)
    fin_dif = (val_p - val_m) / (2 * perturb_scale)
    np.testing.assert_allclose(fin_dif, imp_dif_lp, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(fin_dif, imp_dif_np, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(imp_dif_np, imp_dif_lp, atol=1e-2, rtol=1e-2)

    # center both if balanced problem testing gradient w.r.t weights
    if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
      g_imp_np = g_imp_np - jnp.mean(g_imp_np)
      g_imp_lp = g_imp_lp - jnp.mean(g_imp_lp)

    np.testing.assert_allclose(g_imp_np, g_imp_lp, atol=1e-2, rtol=1e-2)
