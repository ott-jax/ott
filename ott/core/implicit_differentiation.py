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

"""Functions entering the implicit differentiation of Sinkhorn."""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ott.core import dataclasses
from ott.core import problems
from ott.core import unbalanced_functions


@dataclasses.register_pytree_node
class ImplicitDiff:
  """Implicit differentiation of Sinkhorn algorithm.

  Attributes:
    implicit_solver_fun: Callable, should return (solution, ...)
    ridge_kernel: promotes zero-sum solutions. only used if tau_a = tau_b = 1.0
    ridge_identity: handles rank deficient transport matrices (this happens
      typically when rows/cols in cost/kernel matrices are colinear, or,
      equivalently when two points from either measure are close).
    symmetric: flag used to figure out whether the linear system solved in the
      implicit function theorem is symmetric or not. This happens when either
      ``a == b`` or the precondition_fun is the identity. False by default, and,
      at the moment, needs to be set manually by the user in the more favorable
      case where the system is guaranteed to be symmetric.
  """
  solver_fun: Callable = jax.scipy.sparse.linalg.cg  # pylint: disable=g-bare-generic
  ridge_kernel: float = 0.0
  ridge_identity: float = 0.0
  symmetric: bool = False
  precondition_fun: Optional[Callable[[float], float]] = None

  def solve(
      self,
      gr: Tuple[np.ndarray],
      ot_prob: problems.LinearProblem,
      f: np.ndarray,
      g: np.ndarray,
      lse_mode: bool):
    r"""Applies minus inverse of [hessian ``reg_ot_cost`` w.r.t ``f``, ``g``].

    This function is used to carry out implicit differentiation of ``sinkhorn``
    outputs, notably optimal potentials ``f`` and ``g``. That differentiation
    requires solving a linear system, using (and inverting) the Jacobian of
    (preconditioned) first-order conditions w.r.t. the reg-OT problem.

    Given a ``precondition_fun``, written here for short as :math:`h`,
    the first order conditions for the dual energy
    :math:`E(K, \epsilon, a, b, f, g) :=- <a,\phi_a^{*}(-f)> + <b,
    \phi_b^{*}(-g)> - \langle\exp^{f/\epsilon}, K
      \exp^{g/\epsilon}>`

    form the basis of the Sinkhorn algorithm. To differentiate optimal solutions
    to that problem, we exploit the fact that :math:`h(\nabla E = 0)` and
    differentiate that identity to recover variations (Jacobians) of optimal
    solutions :math:`f^\star, g^\star$` as a function of changes in the inputs.
    The Jacobian of :math:`h(\nabla_{f,g} E = 0)` is a linear operator which, if
    it were to be instantiated as a matrix, would be of size
    :math:`(n+m) \times (n+m)`. When :math:`h` is the identity, that matrix is
    the Hessian of :math:`E`, is symmetric and negative-definite
    (:math:`E` is concave) and is structured as :math:`[A, B; B^T, D]`. More
    generally, for other functions :math:`h`, the Jacobian of these
    preconditioned
    first order conditions is no longer symmetric (except if ``a==b``), and
    has now a structure as :math:`[A, B; C, D]`. That system can
    be still inverted more generic solvers. By default, :math:`h = \epsilon
    \log`,
    as proposed in https://arxiv.org/pdf/2002.03229.pdf.

    In both cases :math:`A` and :math:`D` are diagonal matrices, equal to the
    row and
    column marginals respectively, multiplied by the derivatives of :math:`h`
    evaluated at those marginals, corrected (if handling the unbalanced case)
    by the second derivative of the part of the objective that ties potentials
    to the marginals (terms in ``phi_star``). When :math:`h` is the identity,
    :math:`B` and :math:`B^T` are equal respectively to the OT matrix and its
    transpose, i.e. :math:`n \times m` and :math:`m \times n` matrices.
    When :math:`h` is not the identity, :math:`B` (resp. :math:`C`) is equal
    to the OT matrix (resp. its transpose), rescaled on the left by the
    application elementwise of :math:`h'` to the row (respectively column)
    marginal sum of the transport.

    Note that we take great care in not instantiatiating these transport
    matrices, to rely instead on calls to the ``app_transport`` method from the
    ``Geometry`` object ``geom`` (which will either use potentials or scalings,
    depending on ``lse_mode``)

    The Jacobian's diagonal + off-diagonal blocks structure allows to exploit
    Schur complements. Depending on the sizes involved, it is better to
    instantiate the Schur complement of the first or of the second diagonal
    block.

    In either case, the Schur complement is rank deficient, with a 0 eigenvalue
    for the vector of ones in the balanced case, which is why we add a ridge on
    that subspace to enforce solutions have zero sum.

    The Schur complement can also be rank deficient if two lines or columns of T
    are colinear. This will typically happen it two rows or columns of the cost
    or kernel matrix are numerically close. To avoid this, we add a more global
    ``ridge_identity * z`` regularizer to achieve better conditioning.

    These linear systems are solved using the user defined
    ``implicit_solver_fun``,
    which is set by default to ``cg``. When the system is symmetric (as detected
    by the corresponding flag ``symmetric``), ``cg`` is applied directly. When
    it
    is not, normal equations are used (i.e. the Schur complement is multiplied
    by
    its transpose before solving the system).

    Args:
      gr: 2-uple, (vector of size ``n``, vector of size ``m``).
      ot_prob: the instantiation of the regularizad transport problem.
      f: potential, w.r.t marginal a.
      g: potential, w.r.t marginal b.
      lse_mode: bool, log-sum-exp mode if True, kernel else.

    Returns:
      A tuple of two vectors, of the same size as ``gr``.
    """
    geom = ot_prob.geom
    marginal_a, marginal_b, app_transport = (
        ot_prob.get_transport_functions(lse_mode))

    # elementwise vmap apply of derivative of precondition_fun. No vmapping
    # can be problematic here.
    if self.precondition_fun is None:
      precond_fun = lambda x: geom.epsilon * jnp.log(x)
    else:
      precond_fun = self.precondition_fun
    derivative = jax.vmap(jax.grad(precond_fun))

    n, m = geom.shape
    # pylint: disable=g-long-lambda
    vjp_fg = lambda z: app_transport(
        f, g, z * derivative(marginal_b(f, g)), axis=1) / geom.epsilon
    vjp_gf = lambda z: app_transport(
        f, g, z * derivative(marginal_a(f, g)), axis=0) / geom.epsilon

    if not self.symmetric:
      vjp_fgt = lambda z: app_transport(
          f, g, z, axis=0) * derivative(marginal_b(f, g)) / geom.epsilon
      vjp_gft = lambda z: app_transport(
          f, g, z, axis=1) * derivative(marginal_a(f, g)) / geom.epsilon

    diag_hess_a = (
        marginal_a(f, g) * derivative(marginal_a(f, g)) / geom.epsilon +
        unbalanced_functions.diag_jacobian_of_marginal_fit(
            ot_prob.a, f, ot_prob.tau_a, geom.epsilon, derivative))
    diag_hess_b = (
        marginal_b(f, g) * derivative(marginal_b(f, g)) / geom.epsilon +
        unbalanced_functions.diag_jacobian_of_marginal_fit(
            ot_prob.b, g, ot_prob.tau_b, geom.epsilon, derivative))

    n, m = geom.shape
    # Remove ridge on kernel space if problem is balanced.
    ridge_kernel = jnp.where(ot_prob.is_balanced, self.ridge_kernel, 0.0)

    # Forks on using Schur complement of either A or D, depending on size.
    if n > m:  #  if n is bigger, run m x m linear system.
      inv_vjp_ff = lambda z: z / diag_hess_a
      vjp_gg = lambda z: z * diag_hess_b
      schur_ = lambda z: vjp_gg(z) - vjp_gf(inv_vjp_ff(vjp_fg(z)))
      g0, g1 = vjp_gf(inv_vjp_ff(gr[0])), gr[1]

      if self.symmetric:
        schur = lambda z: (
            schur_(z) + ridge_kernel * jnp.sum(z) + self.ridge_identity * z)
      else:
        schur_t = lambda z: vjp_gg(z) - vjp_fgt(inv_vjp_ff(vjp_gft(z)))
        g0, g1 = schur_t(g0), schur_t(g1)
        schur = lambda z: (schur_t(schur_(z)) + ridge_kernel * jnp.sum(z)
                           + self.ridge_identity * z)

      sch_f = self.solver_fun(schur, g0)[0]
      sch_g = self.solver_fun(schur, g1)[0]
      vjp_gr_f = inv_vjp_ff(gr[0] + vjp_fg(sch_f) - vjp_fg(sch_g))
      vjp_gr_g = -sch_f + sch_g
    else:
      vjp_ff = lambda z: z * diag_hess_a
      inv_vjp_gg = lambda z: z / diag_hess_b
      schur_ = lambda z: vjp_ff(z) - vjp_fg(inv_vjp_gg(vjp_gf(z)))
      g0, g1 = vjp_fg(inv_vjp_gg(gr[1])), gr[0]

      if self.symmetric:
        schur = lambda z: (schur_(z) + self.ridge_kernel * jnp.sum(z)
                           + self.ridge_identity * z)
      else:
        schur_t = lambda z: vjp_ff(z) - vjp_gft(inv_vjp_gg(vjp_fgt(z)))
        g0, g1 = schur_t(g0), schur_t(g1)
        schur = lambda z: (schur_t(schur_(z)) + self.ridge_kernel * jnp.sum(z)
                           + self.ridge_identity * z)
      # pylint: enable=g-long-lambda
      sch_g = self.solver_fun(schur, g0)[0]
      sch_f = self.solver_fun(schur, g1)[0]
      vjp_gr_g = inv_vjp_gg(gr[1] + vjp_gf(sch_g) - vjp_gf(sch_f))
      vjp_gr_f = -sch_g + sch_f

    return jnp.concatenate((-vjp_gr_f, -vjp_gr_g))

  def first_order_conditions(
      self, prob, f: jnp.ndarray, g: jnp.ndarray, lse_mode: bool):
    r"""Computes vector of first order conditions for the reg-OT problem.

    The output of this vector should be close to zero at optimality.
    Upon completion of the Sinkhorn forward pass, its norm (using the norm
    parameter defined using ``norm_error``) should be below the threshold
    parameter.

    This error will be itself assumed to be close to zero when using implicit
    differentiation.

    Args:
      prob: definition of the linear optimal transport problem.
      f: jnp.ndarray, first potential
      g: jnp.ndarray, second potential
      lse_mode: bool

    Returns:
      a jnp.ndarray of size (size of ``n + m``) quantifying deviation to
      optimality for variables ``f`` and ``g``.
    """
    geom = prob.geom
    marginal_a, marginal_b, _ = prob.get_transport_functions(lse_mode)
    grad_a = unbalanced_functions.grad_of_marginal_fit(prob.a, f, prob.tau_a,
                                                       geom.epsilon)
    grad_b = unbalanced_functions.grad_of_marginal_fit(prob.b, g, prob.tau_b,
                                                       geom.epsilon)
    if self.precondition_fun is None:
      precond_fun = lambda x: geom.epsilon * jnp.log(x)
    else:
      precond_fun = self.precondition_fun

    result_a = jnp.where(
        prob.a > 0, precond_fun(marginal_a(f, g)) - precond_fun(grad_a), 0.0)
    result_b = jnp.where(
        prob.b > 0, precond_fun(marginal_b(f, g)) - precond_fun(grad_b), 0.0)
    return jnp.concatenate((result_a, result_b))

  def gradient(self, prob, f, g, lse_mode, gr) -> problems.LinearProblem:
    """Applies vjp to recover gradient in reverse mode differentiation."""
    # Applies first part of vjp to gr: inverse part of implicit function theorem
    vjp_gr = self.solve(gr, prob, f, g, lse_mode)

    # Instantiates vjp of first order conditions of the objective, as a
    # function of geom, a and b parameters (against which we differentiate)
    foc_prob = lambda prob: self.first_order_conditions(prob, f, g, lse_mode)

    # Carries pullback onto original inputs, here geom, a and b.
    _, pull_prob = jax.vjp(foc_prob, prob)
    return pull_prob(vjp_gr)
