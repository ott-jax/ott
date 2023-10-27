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
import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from ott import utils
from ott.math import unbalanced_functions as uf

if TYPE_CHECKING:
  from ott.problems.linear import linear_problem

LinOp_t = Callable[[jnp.ndarray], jnp.ndarray]
Solver_t = Callable[[LinOp_t, jnp.ndarray, Optional[LinOp_t], bool],
                    jnp.ndarray]

__all__ = ["ImplicitDiff", "solve_jax_cg"]


@utils.register_pytree_node
class ImplicitDiff:
  """Implicit differentiation of Sinkhorn algorithm.

  Args:
    solver: Callable to compute the solution to a linear problem. The callable
      expects a linear function, a vector, optionally another linear function
      that implements the transpose of that function, and a boolean flag to
      specify symmetry. This solver is by default one of :class:`lineax.CG` or
      :class:`lineax.NormalCG` solvers, if the package can be imported, as
      described in :func:`~ott.solvers.linear.lineax_implicit.solve_lineax`.
      The :mod:`jax` alternative is described in
      :func:`~ott.solvers.linear.implicit_differentiation.solve_jax_cg`.
      Note that `lineax` solvers handle better poorly conditioned problems,
      which arise typically when differentiating the solutions of balanced OT
      problems (when ``tau_a==tau_b==1.0``). Relying on
      :func:`~ott.solvers.linear.implicit_differentiation.solve_jax_cg`
      for such cases might require hand-tuning ridge parameters,
      in particular ``ridge_kernel`` and ``ridge_identity`` as described in its
      doc. These parameters can be passed using ``solver_kwargs`` below.
    solver_kwargs: keyword arguments passed on to the solver.
    symmetric: flag used to figure out whether the linear system solved in the
      implicit function theorem is symmetric or not. This happens when
      ``tau_a==tau_b``, and when ``a == b``, or the precondition_fun
      is the identity. The flag is False by default, and is also tested against
      ``tau_a==tau_b``. It needs to be set manually by the user in the more
      favorable case where the system is guaranteed to be symmetric.
    precondition_fun: Function used to precondition, on both sides, the linear
      system derived from first-order conditions of the regularized OT problem.
      That linear system typically involves an equality between marginals (or
      simple transform of these marginals when the problem is unbalanced) and
      another function of the potentials. When that function is specified, that
      function is applied on both sides of the equality, before being further
      differentiated to provide the Jacobians needed for implicit function
      theorem differentiation.
  """

  solver: Optional[Solver_t] = None
  solver_kwargs: Optional[Dict[str, Any]] = None
  symmetric: bool = False
  precondition_fun: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

  def solve(
      self,
      gr: Tuple[jnp.ndarray, jnp.ndarray],
      ot_prob: "linear_problem.LinearProblem",
      f: jnp.ndarray,
      g: jnp.ndarray,
      lse_mode: bool,
  ) -> jnp.ndarray:
    r"""Apply minus inverse of [hessian ``reg_ot_cost`` w.r.t. ``f``, ``g``].

    This function is used to carry out implicit differentiation of ``sinkhorn``
    outputs, notably optimal potentials ``f`` and ``g``. That differentiation
    requires solving a linear system, using (and inverting) the Jacobian of
    (preconditioned) first-order conditions w.r.t. the reg-OT problem.

    Given a ``precondition_fun``, written here for short as :math:`h`,
    the first order conditions for the dual energy

    .. math::

      E(K, \epsilon, a, b, f, g) :=- <a,\phi_a^{*}(-f)> + <b,
      \phi_b^{*}(-g)> - \langle\exp^{f/\epsilon}, K \exp^{g/\epsilon}>

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
    \log`, as proposed in :cite:`cuturi:20a`.

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

    Note that we take great care in not instantiating these transport
    matrices, to rely instead on calls to the ``app_transport`` method from the
    ``Geometry`` object ``geom`` (which will either use potentials or scalings,
    depending on ``lse_mode``)

    The Jacobian's diagonal + off-diagonal blocks structure allows to exploit
    Schur complements. Depending on the sizes involved, it is better to
    instantiate the Schur complement of the first or of the second diagonal
    block.

    These linear systems are solved using the user-defined ``solver``, using
    by default :mod:`lineax` solvers when available, or falling back on
    :mod:`jax` when not.

    Args:
      gr: 2-tuple, (vector of size ``n``, vector of size ``m``).
      ot_prob: the instantiation of the regularized transport problem.
      f: potential, w.r.t marginal a.
      g: potential, w.r.t marginal b.
      lse_mode: bool, log-sum-exp mode if True, kernel else.

    Returns:
      A tuple of two vectors, of the same size as ``gr``.
    """
    solver = _get_solver() if self.solver is None else self.solver
    solver_kwargs = {} if self.solver_kwargs is None else self.solver_kwargs
    geom = ot_prob.geom
    marginal_a, marginal_b, app_transport = (
        ot_prob.get_transport_functions(lse_mode)
    )
    if self.precondition_fun is None:
      precond_fun = lambda x: geom.epsilon * jnp.log(x)
      symmetric = False
    else:
      precond_fun = self.precondition_fun
      symmetric = self.symmetric

    derivative = jax.vmap(jax.grad(precond_fun))

    n, m = geom.shape
    # pylint: disable=g-long-lambda
    vjp_fg = lambda z: app_transport(
        f, g, z * derivative(marginal_b(f, g)), axis=1
    ) / geom.epsilon
    vjp_gf = lambda z: app_transport(
        f, g, z * derivative(marginal_a(f, g)), axis=0
    ) / geom.epsilon

    if not symmetric:
      vjp_fgt = lambda z: app_transport(
          f, g, z, axis=0
      ) * derivative(marginal_b(f, g)) / geom.epsilon
      vjp_gft = lambda z: app_transport(
          f, g, z, axis=1
      ) * derivative(marginal_a(f, g)) / geom.epsilon

    diag_hess_a = (
        marginal_a(f, g) * derivative(marginal_a(f, g)) / geom.epsilon +
        uf.diag_jacobian_of_marginal_fit(
            ot_prob.a, f, ot_prob.tau_a, geom.epsilon, derivative
        )
    )
    diag_hess_b = (
        marginal_b(f, g) * derivative(marginal_b(f, g)) / geom.epsilon +
        uf.diag_jacobian_of_marginal_fit(
            ot_prob.b, g, ot_prob.tau_b, geom.epsilon, derivative
        )
    )
    n, m = geom.shape
    # TODO(cuturi) consider materializing linear operator schur if size allows.
    # Forks on using Schur complement of either A or D, depending on size.
    if n > m:  #  if n is bigger, run m x m linear system.
      inv_vjp_ff = lambda z: z / diag_hess_a
      vjp_gg = lambda z: z * diag_hess_b
      schur = lambda z: vjp_gg(z) - vjp_gf(inv_vjp_ff(vjp_fg(z)))
      if not symmetric:
        schur_t = lambda z: vjp_gg(z) - vjp_fgt(inv_vjp_ff(vjp_gft(z)))
      else:
        schur_t = None
      res = gr[1] - vjp_gf(inv_vjp_ff(gr[0]))
      sch = solver(schur, res, schur_t, symmetric, **solver_kwargs)
      vjp_gr_f = inv_vjp_ff(gr[0] - vjp_fg(sch))
      vjp_gr_g = sch
    else:
      vjp_ff = lambda z: z * diag_hess_a
      inv_vjp_gg = lambda z: z / diag_hess_b
      schur = lambda z: vjp_ff(z) - vjp_fg(inv_vjp_gg(vjp_gf(z)))

      if not symmetric:
        schur_t = lambda z: vjp_ff(z) - vjp_gft(inv_vjp_gg(vjp_fgt(z)))
      else:
        schur_t = None
      res = gr[0] - vjp_fg(inv_vjp_gg(gr[1]))
      sch = solver(schur, res, schur_t, symmetric, **solver_kwargs)
      vjp_gr_g = inv_vjp_gg(gr[1] - vjp_gf(sch))
      vjp_gr_f = sch

    return jnp.concatenate((-vjp_gr_f, -vjp_gr_g))

  def first_order_conditions(
      self, prob, f: jnp.ndarray, g: jnp.ndarray, lse_mode: bool
  ):
    r"""Compute vector of first order conditions for the reg-OT problem.

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
    grad_a = uf.grad_of_marginal_fit(prob.a, f, prob.tau_a, geom.epsilon)
    grad_b = uf.grad_of_marginal_fit(prob.b, g, prob.tau_b, geom.epsilon)
    if self.precondition_fun is None:
      precond_fun = lambda x: geom.epsilon * jnp.log(x)
    else:
      precond_fun = self.precondition_fun

    result_a = jnp.where(
        prob.a > 0,
        precond_fun(marginal_a(f, g)) - precond_fun(grad_a), 0.0
    )
    result_b = jnp.where(
        prob.b > 0,
        precond_fun(marginal_b(f, g)) - precond_fun(grad_b), 0.0
    )
    return jnp.concatenate((result_a, result_b))

  def gradient(
      self, prob: "linear_problem.LinearProblem", f: jnp.ndarray,
      g: jnp.ndarray, lse_mode: bool, gr: Tuple[jnp.ndarray, jnp.ndarray]
  ) -> "linear_problem.LinearProblem":
    """Apply VJP to recover gradient in reverse mode differentiation."""
    # Applies first part of vjp to gr: inverse part of implicit function theorem
    vjp_gr = self.solve(gr, prob, f, g, lse_mode)

    # Instantiates vjp of first order conditions of the objective, as a
    # function of geom, a and b parameters (against which we differentiate)
    foc_prob = lambda prob: self.first_order_conditions(prob, f, g, lse_mode)

    # Carries pullback onto original inputs, here geom, a and b.
    _, pull_prob = jax.vjp(foc_prob, prob)
    return pull_prob(vjp_gr)

  def replace(self, **kwargs: Any) -> "ImplicitDiff":  # noqa: D102
    return dataclasses.replace(self, **kwargs)


def solve_jax_cg(
    lin: LinOp_t,
    b: jnp.ndarray,
    lin_t: Optional[LinOp_t] = None,
    symmetric: bool = False,
    ridge_identity: float = 0.0,
    ridge_kernel: float = 0.0,
    **kwargs: Any
) -> jnp.ndarray:
  """Wrapper around JAX native linear solvers.

  Args:
    lin: Linear operator
    b: vector. Returned `x` is such that `lin(x)=b`
    lin_t: Linear operator, corresponding to transpose of `lin`.
    symmetric: whether `lin` is symmetric.
    ridge_kernel: promotes zero-sum solutions. Only use if `tau_a = tau_b = 1.0`
    ridge_identity: handles rank deficient transport matrices (this happens
      typically when rows/cols in cost/kernel matrices are collinear, or,
      equivalently when two points from either measure are close).
    kwargs: arguments passed to :func:`~jax.scipy.sparse.linalg.cg`
  """
  op = lin if symmetric else lambda x: lin_t(lin(x))
  if ridge_kernel > 0.0 or ridge_identity > 0.0:
    lin_reg = lambda x: op(x) + ridge_kernel * jnp.sum(x) + ridge_identity * x
  else:
    lin_reg = op
  return jax.scipy.sparse.linalg.cg(lin_reg, b, **kwargs)[0]


def _get_solver() -> Solver_t:
  """Get lineax solver when possible, default to jax.scipy else."""
  try:
    from ott.solvers.linear import lineax_implicit
    return lineax_implicit.solve_lineax
  except ImportError:
    return solve_jax_cg
