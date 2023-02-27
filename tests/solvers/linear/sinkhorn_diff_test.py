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
import functools
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ott.geometry import costs, geometry, grid, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear import sinkhorn


class TestSinkhornImplicit:
  """Check implicit and autodiff match for Sinkhorn."""

  @pytest.fixture(autouse=True)
  def initialize(self, rng: jax.random.PRNGKeyArray):
    self.dim = 3
    self.n = 38
    self.m = 73
    self.rng, *rngs = jax.random.split(rng, 10)
    self.rngs = rngs
    self.x = jax.random.uniform(rngs[0], (self.n, self.dim))
    self.y = jax.random.uniform(rngs[1], (self.m, self.dim))
    a = jax.random.uniform(rngs[2], (self.n,)) + .1
    b = jax.random.uniform(rngs[3], (self.m,)) + .1
    self.a = a / jnp.sum(a)
    self.b = b / jnp.sum(b)

  @pytest.mark.parametrize(("lse_mode", "threshold", "pcg"),
                           [(False, 1e-6, False), (True, 1e-4, True)])
  def test_implicit_differentiation_versus_autodiff(
      self, lse_mode: bool, threshold: float, pcg: bool
  ):
    epsilon = 0.05

    def loss_g(a: jnp.ndarray, x: jnp.ndarray, implicit: bool = True) -> float:
      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      geom = geometry.Geometry(
          cost_matrix=jnp.sum(x ** 2, axis=1)[:, jnp.newaxis] +
          jnp.sum(self.y ** 2, axis=1)[jnp.newaxis, :] -
          2 * jnp.dot(x, self.y.T),
          epsilon=epsilon
      )
      prob = linear_problem.LinearProblem(
          geom, a=a, b=self.b, tau_a=0.9, tau_b=0.87
      )
      solver = sinkhorn.Sinkhorn(
          threshold=threshold, lse_mode=lse_mode, implicit_diff=implicit_diff
      )
      return solver(prob).reg_ot_cost

    def loss_pcg(
        a: jnp.ndarray, x: jnp.ndarray, implicit: bool = True
    ) -> float:
      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      geom = pointcloud.PointCloud(x, self.y, epsilon=epsilon)
      prob = linear_problem.LinearProblem(
          geom, a=a, b=self.b, tau_a=1.0, tau_b=0.95
      )
      solver = sinkhorn.Sinkhorn(
          threshold=threshold, lse_mode=lse_mode, implicit_diff=implicit_diff
      )
      return solver(prob).reg_ot_cost

    loss = loss_pcg if pcg else loss_g

    loss_and_grad_imp = jax.jit(
        jax.value_and_grad(lambda a, x: loss(a, x, True), argnums=(0, 1))
    )
    loss_and_grad_auto = jax.jit(
        jax.value_and_grad(lambda a, x: loss(a, x, False), argnums=(0, 1))
    )

    loss_value_imp, grad_loss_imp = loss_and_grad_imp(self.a, self.x)
    loss_value_auto, grad_loss_auto = loss_and_grad_auto(self.a, self.x)

    np.testing.assert_allclose(loss_value_imp, loss_value_auto)
    eps = 1e-3

    # test gradient w.r.t. a works and gradient implicit ~= gradient autodiff
    delta = jax.random.uniform(self.rngs[4], (self.n,)) / 10
    delta = delta - jnp.mean(delta)  # center perturbation
    reg_ot_delta_plus = loss(self.a + eps * delta, self.x)
    reg_ot_delta_minus = loss(self.a - eps * delta, self.x)
    delta_dot_grad = jnp.sum(delta * grad_loss_imp[0])
    np.testing.assert_allclose(
        delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
        rtol=1e-02,
        atol=1e-02
    )
    # note how we removed gradients below. This is because gradients are only
    # determined up to additive constant here (the primal variable is in the
    # simplex).
    np.testing.assert_allclose(
        grad_loss_imp[0] - jnp.mean(grad_loss_imp[0]),
        grad_loss_auto[0] - jnp.mean(grad_loss_auto[0]),
        rtol=1e-02,
        atol=1e-02
    )

    # test gradient w.r.t. x works and gradient implicit ~= gradient autodiff
    delta = jax.random.uniform(self.rngs[4], (self.n, self.dim))
    reg_ot_delta_plus = loss(self.a, self.x + eps * delta)
    reg_ot_delta_minus = loss(self.a, self.x - eps * delta)
    delta_dot_grad = jnp.sum(delta * grad_loss_imp[1])
    np.testing.assert_allclose(
        delta_dot_grad, (reg_ot_delta_plus - reg_ot_delta_minus) / (2 * eps),
        rtol=1e-02,
        atol=1e-02
    )
    np.testing.assert_allclose(
        grad_loss_imp[1], grad_loss_auto[1], rtol=1e-02, atol=1e-02
    )


class TestSinkhornJacobian:

  @pytest.mark.fast.with_args(
      "lse_mode,shape_data",
      [(True, (7, 9)), (False, (11, 5))],
      only_fast=0,
  )
  def test_autograd_sinkhorn(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool, shape_data: Tuple[int,
                                                                            int]
  ):
    """Test gradient w.r.t. probability weights."""
    n, m = shape_data
    d = 3
    eps = 1e-3  # perturbation magnitude
    rngs = jax.random.split(rng, 5)
    x = jax.random.uniform(rngs[0], (n, d))
    y = jax.random.uniform(rngs[1], (m, d))
    a = jax.random.uniform(rngs[2], (n,)) + eps
    b = jax.random.uniform(rngs[3], (m,)) + eps
    # Adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)

    def reg_ot(a: jnp.ndarray, b: jnp.ndarray) -> float:
      geom = pointcloud.PointCloud(x, y, epsilon=1e-1)
      prob = linear_problem.LinearProblem(geom, a=a, b=b)
      # TODO: fails with `jit=True`, investigate
      solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, jit=False)
      return solver(prob).reg_ot_cost

    reg_ot_and_grad = jax.jit(jax.grad(reg_ot))
    grad_reg_ot = reg_ot_and_grad(a, b)
    delta = jax.random.uniform(rngs[4], (n,))
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

  @pytest.mark.parametrize(("lse_mode", "shape_data"), [(True, (7, 9)),
                                                        (False, (11, 5))])
  def test_gradient_sinkhorn_geometry(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool, shape_data: Tuple[int,
                                                                            int]
  ):
    """Test gradient w.r.t. cost matrix."""
    n, m = shape_data
    rngs = jax.random.split(rng, 2)
    cost_matrix = jnp.abs(jax.random.normal(rngs[0], (n, m)))
    delta = jax.random.normal(rngs[1], (n, m))
    delta = delta / jnp.sqrt(jnp.vdot(delta, delta))
    eps = 1e-3  # perturbation magnitude

    def loss_fn(cm: jnp.ndarray):
      a = jnp.ones(cm.shape[0]) / cm.shape[0]
      b = jnp.ones(cm.shape[1]) / cm.shape[1]
      geom = geometry.Geometry(cm, epsilon=0.5)
      prob = linear_problem.LinearProblem(geom, a=a, b=b)
      solver = sinkhorn.Sinkhorn(lse_mode=lse_mode)
      out = solver(prob)
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
      "lse_mode,implicit,min_iter,max_iter,epsilon,cost_fn",
      [
          (True, True, 0, 2000, 1e-3, costs.Euclidean()),
          (True, True, 1000, 1000, 1e-3, costs.Euclidean()),
          (True, False, 1000, 1000, 1e-2, costs.SqEuclidean()),
          (True, False, 0, 2000, 1e-2, costs.SqEuclidean()),
          (False, True, 0, 2000, 1e-2, costs.Euclidean()),
      ],
      ids=[
          "lse-implicit", "lse-implicit-force_scan", "lse-backprop-force_scan",
          "lse-backprop", "scan-implicit"
      ],
      only_fast=[0, 1],
  )
  def test_gradient_sinkhorn_euclidean(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool, implicit: bool,
      min_iter: int, max_iter: int, epsilon: float, cost_fn: costs.CostFn
  ):
    """Test gradient w.r.t. locations x of reg-ot-cost."""
    # TODO(cuturi): ensure scaling mode works with backprop.
    d = 3
    n, m = 11, 13
    rngs = jax.random.split(rng, 4)
    x = jax.random.normal(rngs[0], (n, d)) / 10
    y = jax.random.normal(rngs[1], (m, d)) / 10

    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (m,))
    # Adding zero weights to test proper handling
    a = a.at[0].set(0)
    b = b.at[3].set(0)
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    # Adding some near-zero distances to test proper handling with p_norm=1.
    y = y.at[0].set(x[0, :] + 1e-3)

    def loss_fn(x: jnp.ndarray,
                y: jnp.ndarray) -> Tuple[float, sinkhorn.SinkhornOutput]:
      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon, cost_fn=cost_fn)
      prob = linear_problem.LinearProblem(geom, a, b)
      solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode,
          min_iterations=min_iter,
          max_iterations=max_iter,
          # TODO(cuturi): figure out why implicit diff breaks when `jit=True`
          jit=False,
          implicit_diff=implicit_diff,
      )
      out = solver(prob)
      return out.reg_ot_cost, out

    delta = jax.random.normal(rngs[0], (n, d))
    delta = delta / jnp.sqrt(jnp.vdot(delta, delta))
    eps = 1e-5  # perturbation magnitude

    # first calculation of gradient
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    (loss_value, out), grad_loss = loss_and_grad(x, y)
    custom_grad = jnp.sum(delta * grad_loss)

    assert not jnp.isnan(loss_value)
    np.testing.assert_array_equal(grad_loss.shape, x.shape)
    np.testing.assert_array_equal(jnp.isnan(grad_loss), False)
    # second calculation of gradient
    tm = out.matrix
    if isinstance(cost_fn, costs.SqEuclidean):
      tmp = 2 * tm[:, :, None] * (x[:, None, :] - y[None, :, :])
      grad_x = jnp.sum(tmp, 1)
      other_grad = jnp.sum(delta * grad_x)
    if isinstance(cost_fn, costs.Euclidean):
      tmp = tm[:, :, None] * (x[:, None, :] - y[None, :, :])
      norms = jnp.linalg.norm(x[:, None] - y[None, :], axis=-1)
      tmp /= norms[:, :, None] + 1e-8  # to stabilize when computed by hand
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

  def test_autoepsilon_differentiability(self, rng: jax.random.PRNGKeyArray):
    cost = jax.random.uniform(rng, (15, 17))

    def reg_ot_cost(c: jnp.ndarray) -> float:
      geom = geometry.Geometry(c, epsilon=None)  # auto epsilon
      prob = linear_problem.LinearProblem(geom)
      return sinkhorn.Sinkhorn()(prob).reg_ot_cost

    gradient = jax.grad(reg_ot_cost)(cost)
    np.testing.assert_array_equal(jnp.isnan(gradient), False)

  @pytest.mark.fast()
  def test_differentiability_with_jit(self, rng: jax.random.PRNGKeyArray):

    def reg_ot_cost(c: jnp.ndarray) -> float:
      geom = geometry.Geometry(c, epsilon=1e-2)
      prob = linear_problem.LinearProblem(geom)
      return sinkhorn.Sinkhorn()(prob).reg_ot_cost

    cost = jax.random.uniform(rng, (15, 17))
    gradient = jax.jit(jax.grad(reg_ot_cost))(cost)
    np.testing.assert_array_equal(jnp.isnan(gradient), False)

  @pytest.mark.fast.with_args(
      lse_mode=[True, False],
      tau_a=[1.0, .94],
      tau_b=[1.0, .92],
      shape=[(237, 153)],
      arg=[0, 1],
      axis=[0, 1],
      only_fast=0,
  )
  def test_apply_transport_jacobian(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool, tau_a: float,
      tau_b: float, shape: Tuple[int, int], arg: int, axis: int
  ):
    """Tests Jacobian of application of OT to vector, w.r.t.

    a/x.

    Args:
      lse_mode: most tests here are intended to be run in lse_mode=True, kernel
        mode (False) is tested with looser convergence settings.
      tau_a: loosen up 1st marginal constraint when <1.0
      tau_b: loosen up 2nd marginal constraint when <1.0
      shape: size for point clouds n, m.
      arg: test jacobian w.r.t. either weight vectors a or locations x
      axis: test the jacobian of the application of the (right) application of
        transport to arbitrary vec (axis=0) or the left (axis=1).
    """
    n, m = shape
    dim = 4
    rngs = jax.random.split(rng, 9)
    x = jax.random.uniform(rngs[0], (n, dim)) / dim
    y = jax.random.uniform(rngs[1], (m, dim)) / dim
    a = jax.random.uniform(rngs[2], (n,)) + .2
    b = jax.random.uniform(rngs[3], (m,)) + .2
    a = a / (0.5 * n) if tau_a < 1.0 else a / jnp.sum(a)
    b = b / (0.5 * m) if tau_b < 1.0 else b / jnp.sum(b)
    vec = jax.random.uniform(rngs[4], (m if axis else n,)) - .5

    delta_a = jax.random.uniform(rngs[5], (n,))
    if tau_a == 1.0:
      delta_a = delta_a - jnp.mean(delta_a)
    delta_x = jax.random.uniform(rngs[6], (n, dim))

    # lse_mode=False is unstable for small epsilon when differentiating as a
    # general rule, even more so when using backprop.
    epsilon = 0.01 if lse_mode else 0.1

    def apply_ot(a: jnp.ndarray, x: jnp.ndarray, implicit: bool) -> jnp.ndarray:
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
      prob = linear_problem.LinearProblem(geom, a, b, tau_a=tau_a, tau_b=tau_b)

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, implicit_diff=implicit_diff)
      out = solver(prob)

      return out.apply(vec, axis=axis)

    delta = delta_x if arg else delta_a
    # Compute implicit jacobian
    jac_apply_imp = jax.jit(
        jax.jacrev(lambda a, x: apply_ot(a, x, True), argnums=arg)
    )
    j_imp = jac_apply_imp(a, x)
    # Apply jacobian to perturbation tensor (here vector or matrix)
    imp_dif = jnp.sum(
        j_imp * delta[jnp.newaxis, ...],
        axis=tuple(range(1, 1 + len(delta.shape)))
    )

    if lse_mode:  # only check unrolling if using lse_mode, too unstable else.
      # Compute backprop (unrolling) jacobian
      jac_apply_back = jax.jit(
          jax.jacrev(lambda a, x: apply_ot(a, x, False), argnums=arg)
      )
      j_back = jac_apply_back(a, x)
      # Apply jacobian to perturbation tensor (here vector or matrix)
      back_dif = jnp.sum(
          j_back * delta[jnp.newaxis, ...],
          axis=tuple(range(1, 1 + len(delta.shape)))
      )

    # Compute finite difference
    perturb_scale = 1e-5
    a_p = a + perturb_scale * delta_a if arg == 0 else a
    x_p = x if arg == 0 else x + perturb_scale * delta_x
    a_m = a - perturb_scale * delta_a if arg == 0 else a
    x_m = x if arg == 0 else x - perturb_scale * delta_x

    app_p = apply_ot(a_p, x_p, False)
    app_m = apply_ot(a_m, x_m, True)
    fin_dif = (app_p - app_m) / (2 * perturb_scale)

    # Set tolerance depending on lse_mode (False is more loose)
    atol = 1e-2 if lse_mode else 1e-1
    # Check finite differences match with application of (implicit) Jacobian.
    np.testing.assert_allclose(fin_dif, imp_dif, atol=atol, rtol=1e-1)

    # Check unrolling jacobian when using lse_mode.
    if lse_mode:
      np.testing.assert_allclose(fin_dif, back_dif, atol=atol, rtol=1e-1)

      # Check Jacobian matrices match loosely.
      # Orthogonalize j_imp, j_back w.r.t. 1 if balanced problem,
      # and testing jacobian w.r.t weights
      if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
        j_imp = j_imp - jnp.mean(j_imp, axis=1)[:, None]
        j_back = j_back - jnp.mean(j_imp, axis=1)[:, None]
      np.testing.assert_allclose(j_imp, j_back, atol=atol, rtol=1e-1)

  @pytest.mark.fast.with_args(
      lse_mode=[True, False],
      tau_a=[1.0, .93],
      tau_b=[1.0, .91],
      shape=[(12, 15), (27, 18)],
      arg=[0, 1],
      only_fast=0,
  )
  def test_potential_jacobian_sinkhorn(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool, tau_a: float,
      tau_b: float, shape: Tuple[int, int], arg: int
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

    def loss_from_potential(a: jnp.ndarray, x: jnp.ndarray, implicit: bool):
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
      prob = linear_problem.LinearProblem(geom, a, b, tau_a=tau_a, tau_b=tau_b)

      implicit_diff = implicit_lib.ImplicitDiff() if implicit else None
      solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, implicit_diff=implicit_diff)

      out = solver(prob)

      return jnp.sum(random_dir * out.f)

    # Compute implicit gradient
    loss_imp = jax.jit(
        jax.value_and_grad(
            lambda a, x: loss_from_potential(a, x, True), argnums=arg
        )
    )
    _, g_imp = loss_imp(a, x)
    imp_dif = jnp.sum(g_imp * (delta_a if arg == 0 else delta_x))
    # Compute backprop (unrolling) gradient

    loss_back = jax.jit(
        jax.grad(lambda a, x: loss_from_potential(a, x, False), argnums=arg)
    )
    g_back = loss_back(a, x)
    back_dif = jnp.sum(g_back * (delta_a if arg == 0 else delta_x))

    # Compute finite difference
    perturb_scale = 1e-4
    a_p = a + perturb_scale * delta_a if arg == 0 else a
    x_p = x if arg == 0 else x + perturb_scale * delta_x
    a_m = a - perturb_scale * delta_a if arg == 0 else a
    x_m = x if arg == 0 else x - perturb_scale * delta_x

    val_p, _ = loss_imp(a_p, x_p)
    val_m, _ = loss_imp(a_m, x_m)
    fin_dif = (val_p - val_m) / (2 * perturb_scale)

    np.testing.assert_allclose(fin_dif, back_dif, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(fin_dif, imp_dif, atol=1e-2, rtol=1e-2)

    # center g_imp, g_back if balanced problem testing gradient w.r.t weights
    if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
      g_imp = g_imp - jnp.mean(g_imp)
      g_back = g_back - jnp.mean(g_back)
    np.testing.assert_allclose(g_imp, g_back, atol=5e-2, rtol=1e-2)


@pytest.mark.fast()
class TestSinkhornGradGrid:

  @pytest.mark.parametrize("lse_mode", [False, True])
  def test_diff_sinkhorn_x_grid_x_perturbation(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool
  ):
    """Test gradient w.r.t. probability weights."""
    eps = 1e-3  # perturbation magnitude
    rngs = jax.random.split(rng, 6)
    x = (
        jnp.array([.0, 1.0]), jnp.array([.3, .4,
                                         .7]), jnp.array([1.0, 1.3, 2.4, 3.7])
    )
    grid_size = tuple(xs.shape[0] for xs in x)
    a = jax.random.uniform(rngs[0], grid_size) + 1.0
    b = jax.random.uniform(rngs[1], grid_size) + 1.0
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)

    def reg_ot(x: List[jnp.ndarray]) -> float:
      geom = grid.Grid(x=x, epsilon=1.0)
      prob = linear_problem.LinearProblem(geom, a=a, b=b)
      solver = sinkhorn.Sinkhorn(threshold=1e-1, lse_mode=lse_mode)
      return solver(prob).reg_ot_cost

    reg_ot_and_grad = jax.grad(reg_ot)
    grad_reg_ot = reg_ot_and_grad(x)
    delta = [jax.random.uniform(rngs[i], (g,)) for i, g in enumerate(grid_size)]

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
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool
  ):
    """Test gradient w.r.t. probability weights."""
    eps = 1e-4  # perturbation magnitude
    rngs = jax.random.split(rng, 3)
    # yapf: disable
    x = (
        jnp.asarray([.0, 1.0]),
        jnp.asarray([.3, .4, .7]),
        jnp.asarray([1.0, 1.3, 2.4, 3.7])
    )
    # yapf: enable
    grid_size = tuple(xs.shape[0] for xs in x)
    a = jax.random.uniform(rngs[0], grid_size) + 1
    b = jax.random.uniform(rngs[1], grid_size) + 1
    a = a.ravel() / jnp.sum(a)
    b = b.ravel() / jnp.sum(b)
    geom = grid.Grid(x=x, epsilon=1)

    def reg_ot(a: jnp.ndarray, b: jnp.ndarray) -> float:
      prob = linear_problem.LinearProblem(geom, a, b)
      solver = sinkhorn.Sinkhorn(threshold=1e-3, lse_mode=lse_mode)
      return solver(prob).reg_ot_cost

    reg_ot_and_grad = jax.grad(reg_ot)
    grad_reg_ot = reg_ot_and_grad(a, b)
    delta = jax.random.uniform(rngs[2], grid_size).ravel()
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
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool, tau_a: float,
      tau_b: float, shape: Tuple[int, int], arg: int
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
        a: jnp.ndarray,
        x: jnp.ndarray,
        precondition_fun: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        symmetric: bool = False
    ) -> float:
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
      prob = linear_problem.LinearProblem(geom, a, b, tau_a=tau_a, tau_b=tau_b)

      implicit_diff = implicit_lib.ImplicitDiff(
          symmetric=symmetric, precondition_fun=precondition_fun
      )
      solver = sinkhorn.Sinkhorn(lse_mode=lse_mode, implicit_diff=implicit_diff)

      out = solver(prob)

      return jnp.sum(random_dir * out.f)

    # Compute implicit gradient
    loss_imp_no_precond = jax.jit(
        jax.value_and_grad(
            functools.partial(
                loss_from_potential,
                precondition_fun=lambda x: x,
                symmetric=True,
            ),
            argnums=arg
        )
    )

    loss_imp_log_precond = jax.jit(jax.grad(loss_from_potential, argnums=arg))

    _, g_imp_np = loss_imp_no_precond(a, x)
    imp_dif_np = jnp.sum(g_imp_np * (delta_a if arg == 0 else delta_x))

    g_imp_lp = loss_imp_log_precond(a, x)
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


class TestSinkhornHessian:

  @pytest.mark.fast.with_args(
      lse_mode=[True, False],
      tau_a=[1.0, .93],
      tau_b=[1.0, .91],
      shape=[(12, 15)],
      arg=[0, 1],
      only_fast=-1
  )
  def test_hessian_sinkhorn(
      self, rng: jax.random.PRNGKeyArray, lse_mode: bool, tau_a: float,
      tau_b: float, shape: Tuple[int, int], arg: int
  ):
    """Test hessian w.r.t. weights and locations."""
    # TODO(cuturi): reinstate this flag to True when JAX bug fixed.
    test_back = False

    n, m = shape
    dim = 3
    rngs = jax.random.split(rng, 6)
    x = jax.random.uniform(rngs[0], (n, dim))
    y = jax.random.uniform(rngs[1], (m, dim))
    a = jax.random.uniform(rngs[2], (n,)) + .1
    b = jax.random.uniform(rngs[3], (m,)) + .1
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    epsilon = 0.1
    ridge = 1e-5

    def loss(a: jnp.ndarray, x: jnp.ndarray, implicit: bool = True):
      geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
      prob = linear_problem.LinearProblem(geom, a, b, tau_a, tau_b)
      implicit_diff = (
          None if not implicit else
          implicit_lib.ImplicitDiff(ridge_kernel=ridge, ridge_identity=ridge)
      )
      solver = sinkhorn.Sinkhorn(
          lse_mode=lse_mode,
          threshold=1e-4,
          use_danskin=False,
          implicit_diff=implicit_diff,
      )
      return solver(prob).reg_ot_cost

    delta_a = jax.random.uniform(rngs[4], (n,))
    delta_a = delta_a - jnp.mean(delta_a)
    delta_x = jax.random.uniform(rngs[5], (n, dim))

    hess_loss_imp = jax.jit(
        jax.hessian(lambda a, x: loss(a, x, True), argnums=arg)
    )
    hess_imp = hess_loss_imp(a, x)

    # Test that Hessians produced with either backprop or implicit do match.
    if test_back:
      hess_loss_back = jax.jit(
          jax.hessian(lambda a, x: loss(a, x, False), argnums=arg)
      )
      hess_back = hess_loss_back(a, x)
    # In the balanced case, when studying differentiability w.r.t
    # weights, both Hessians must be the same,
    # but only need to be so on the orthogonal space to 1s.
    # For that reason we remove that contribution and check the
    # resulting matrices are equal.
    if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
      hess_imp -= jnp.mean(hess_imp, axis=1)[:, None]
      if test_back:
        hess_back -= jnp.mean(hess_back, axis=1)[:, None]

    # Uniform equality is difficult to obtain numerically on the
    # entire matrices. We switch to relative 1-norm of difference.
    if test_back:
      dif_norm = jnp.sum(jnp.abs(hess_imp - hess_back))
      rel_dif_norm = dif_norm / jnp.sum(jnp.abs(hess_imp))
      assert rel_dif_norm < 0.1

    eps = 1e-3
    for impl in [True, False] if test_back else [True]:
      grad_ = jax.jit(
          jax.grad(functools.partial(loss, implicit=impl), argnums=arg)
      )
      grad_init = grad_(a, x)

      # Depending on variable tested, perturb either a or x.
      a_p = a + eps * delta_a if arg == 0 else a
      x_p = x if arg == 0 else x + eps * delta_x

      # Perturbed gradient.
      grad_pert = grad_(a_p, x_p)
      grad_dif = (grad_pert - grad_init) / eps
      # Apply hessian to perturbation
      if arg == 0:
        hess_delta = jnp.matmul(hess_imp, delta_a)
      else:
        # Here tensordot is needed because Hessian is 4D, delta_x is 2D.
        hess_delta = jnp.tensordot(hess_imp, delta_x)

      if tau_a == 1.0 and tau_b == 1.0 and arg == 0:
        hess_delta -= jnp.mean(hess_delta)
        grad_dif -= jnp.mean(grad_dif)

      # No rtol here because many of these values can be close to 0.
      np.testing.assert_allclose(grad_dif, hess_delta, atol=0.1, rtol=0.1)
