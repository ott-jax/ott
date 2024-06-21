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
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ott import utils
from ott.geometry import geometry
from ott.initializers.linear import initializers as init_lib
from ott.math import fixed_point_loop
from ott.math import unbalanced_functions as uf
from ott.math import utils as mu
from ott.problems.linear import linear_problem, potentials
from ott.solvers.linear import acceleration
from ott.solvers.linear import implicit_differentiation as implicit_lib

__all__ = ["Sinkhorn", "SinkhornOutput"]

ProgressCallbackFn_t = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "SinkhornState"]], None]


class SinkhornState(NamedTuple):
  """Holds the state variables used to solve OT with Sinkhorn."""

  potentials: Tuple[jnp.ndarray, ...]
  errors: Optional[jnp.ndarray] = None
  old_fus: Optional[jnp.ndarray] = None
  old_mapped_fus: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "SinkhornState":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def solution_error(
      self,
      ot_prob: linear_problem.LinearProblem,
      norm_error: Sequence[int],
      *,
      lse_mode: bool,
      parallel_dual_updates: bool,
      recenter: bool,
  ) -> jnp.ndarray:
    """State dependent function to return error."""
    fu, gv = self.fu, self.gv
    if recenter and lse_mode:
      fu, gv = self.recenter(fu, gv, ot_prob=ot_prob)

    return solution_error(
        fu,
        gv,
        ot_prob,
        norm_error=norm_error,
        lse_mode=lse_mode,
        parallel_dual_updates=parallel_dual_updates
    )

  def compute_kl_reg_cost(  # noqa: D102
      self, ot_prob: linear_problem.LinearProblem, lse_mode: bool
  ) -> float:
    return compute_kl_reg_cost(self.fu, self.gv, ot_prob, lse_mode)

  def recenter(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      ot_prob: linear_problem.LinearProblem,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Re-center dual potentials.

    If the ``ot_prob`` is balanced, the ``f`` potential is zero-centered.
    Otherwise, use Prop. 2 of :cite:`sejourne:22` re-center the potentials iff
    ``tau_a < 1`` and ``tau_b < 1``.

    Args:
      f: The first dual potential.
      g: The second dual potential.
      ot_prob: Linear OT problem.

    Returns:
      The centered potentials.
    """
    if ot_prob.is_balanced:
      # center the potentials for numerical stability
      is_finite = jnp.isfinite(f)
      shift = jnp.sum(jnp.where(is_finite, f, 0.0)) / jnp.sum(is_finite)
      return f - shift, g + shift

    if ot_prob.tau_a == 1.0 or ot_prob.tau_b == 1.0:
      # re-centering wasn't done during the lse-step, ignore
      return f, g

    rho_a = uf.rho(ot_prob.epsilon, ot_prob.tau_a)
    rho_b = uf.rho(ot_prob.epsilon, ot_prob.tau_b)
    tau = rho_a * rho_b / (rho_a + rho_b)

    shift = tau * (
        mu.logsumexp(-f / rho_a, b=ot_prob.a) -
        mu.logsumexp(-g / rho_b, b=ot_prob.b)
    )
    return f + shift, g - shift

  @property
  def fu(self) -> jnp.ndarray:
    """The first dual potential or scaling."""
    return self.potentials[0]

  @property
  def gv(self) -> jnp.ndarray:
    """The second dual potential or scaling."""
    return self.potentials[1]


def solution_error(
    f_u: jnp.ndarray,
    g_v: jnp.ndarray,
    ot_prob: linear_problem.LinearProblem,
    *,
    norm_error: Sequence[int],
    lse_mode: bool,
    parallel_dual_updates: bool,
) -> jnp.ndarray:
  """Given two potential/scaling solutions, computes deviation to optimality.

  When the ``ot_prob`` problem is balanced and the usual Sinkhorn updates are
  used, this is simply deviation of the coupling's marginal to ``ot_prob.b``.
  This is the case because the second (and last) update of the Sinkhorn
  algorithm equalizes the row marginal of the coupling to ``ot_prob.a``. To
  simplify the logic, this is parameterized by checking whether
  `parallel_dual_updates = False`.

  When that flag is `True`, or when the problem is unbalanced,
  additional quantities to qualify optimality must be taken into account.

  Args:
    f_u: jnp.ndarray, potential or scaling
    g_v: jnp.ndarray, potential or scaling
    ot_prob: linear OT problem
    norm_error: int, p-norm used to compute error.
    lse_mode: True if log-sum-exp operations, False if kernel vector products.
    parallel_dual_updates: Whether potentials/scalings were computed in
      parallel.

  Returns:
    a positive number quantifying how far from optimality current solution is.
  """
  if ot_prob.is_balanced and not parallel_dual_updates:
    return marginal_error(
        f_u, g_v, ot_prob.b, ot_prob.geom, 0, norm_error, lse_mode
    )

  # In the unbalanced case, we compute the norm of the gradient.
  # the gradient is equal to the marginal of the current plan minus
  # the gradient of < z, rho_z(exp^(-h/rho_z) -1> where z is either a or b
  # and h is either f or g. Note this is equal to z if rho_z → inf, which
  # is the case when tau_z → 1.0
  if lse_mode:
    grad_a = uf.grad_of_marginal_fit(
        ot_prob.a, f_u, ot_prob.tau_a, ot_prob.epsilon
    )
    grad_b = uf.grad_of_marginal_fit(
        ot_prob.b, g_v, ot_prob.tau_b, ot_prob.epsilon
    )
  else:
    u = ot_prob.geom.potential_from_scaling(f_u)
    v = ot_prob.geom.potential_from_scaling(g_v)
    grad_a = uf.grad_of_marginal_fit(
        ot_prob.a, u, ot_prob.tau_a, ot_prob.epsilon
    )
    grad_b = uf.grad_of_marginal_fit(
        ot_prob.b, v, ot_prob.tau_b, ot_prob.epsilon
    )
  err = marginal_error(f_u, g_v, grad_a, ot_prob.geom, 1, norm_error, lse_mode)
  err += marginal_error(f_u, g_v, grad_b, ot_prob.geom, 0, norm_error, lse_mode)
  return err


def marginal_error(
    f_u: jnp.ndarray,
    g_v: jnp.ndarray,
    target: jnp.ndarray,
    geom: geometry.Geometry,
    axis: int = 0,
    norm_error: Sequence[int] = (1,),
    lse_mode: bool = True
) -> jnp.asarray:
  """Output how far Sinkhorn solution is w.r.t target.

  Args:
    f_u: a vector of potentials or scalings for the first marginal.
    g_v: a vector of potentials or scalings for the second marginal.
    target: target marginal.
    geom: Geometry object.
    axis: axis (0 or 1) along which to compute marginal.
    norm_error: (tuple of int) p's to compute p-norm between marginal/target
    lse_mode: whether operating on scalings or potentials

  Returns:
    Array of floats, quantifying difference between target / marginal.
  """
  if lse_mode:
    marginal = geom.marginal_from_potentials(f_u, g_v, axis=axis)
  else:
    marginal = geom.marginal_from_scalings(f_u, g_v, axis=axis)
  norm_error = jnp.asarray(norm_error)
  return jnp.sum(
      jnp.abs(marginal - target) ** norm_error[:, jnp.newaxis], axis=1
  ) ** (1.0 / norm_error)


def compute_kl_reg_cost(
    f: jnp.ndarray, g: jnp.ndarray, ot_prob: linear_problem.LinearProblem,
    lse_mode: bool
) -> float:
  r"""Compute objective of Sinkhorn for OT problem given dual solutions.

  The objective is evaluated for dual solution ``f`` and ``g``, using
  information contained in  ``ot_prob``. The objective is the regularized
  optimal transport cost (i.e. the cost itself plus entropic and unbalanced
  terms). Situations where marginals ``a`` or ``b`` in ot_prob have zero
  coordinates are reflected in minus infinity entries in their corresponding
  dual potentials. To avoid NaN that may result when multiplying 0's by infinity
  values, ``jnp.where`` is used to cancel these contributions.

  Args:
    f: jnp.ndarray, potential
    g: jnp.ndarray, potential
    ot_prob: linear optimal transport problem.
    lse_mode: bool, whether to compute total mass in lse or kernel mode.

  Returns:
    The regularized transport cost.
  """
  supp_a = ot_prob.a > 0
  supp_b = ot_prob.b > 0
  fa = ot_prob.geom.potential_from_scaling(ot_prob.a)
  if ot_prob.tau_a == 1.0:
    div_a = jnp.sum(jnp.where(supp_a, ot_prob.a * (f - fa), 0.0))
  else:
    rho_a = uf.rho(ot_prob.epsilon, ot_prob.tau_a)
    div_a = -jnp.sum(
        jnp.where(supp_a, ot_prob.a * uf.phi_star(-(f - fa), rho_a), 0.0)
    )

  gb = ot_prob.geom.potential_from_scaling(ot_prob.b)
  if ot_prob.tau_b == 1.0:
    div_b = jnp.sum(jnp.where(supp_b, ot_prob.b * (g - gb), 0.0))
  else:
    rho_b = uf.rho(ot_prob.epsilon, ot_prob.tau_b)
    div_b = -jnp.sum(
        jnp.where(supp_b, ot_prob.b * uf.phi_star(-(g - gb), rho_b), 0.0)
    )

  # Using https://arxiv.org/pdf/1910.12958v2.pdf (24)
  if lse_mode:
    total_sum = jnp.sum(ot_prob.geom.marginal_from_potentials(f, g))
  else:
    u = ot_prob.geom.scaling_from_potential(f)
    v = ot_prob.geom.scaling_from_potential(g)
    total_sum = jnp.sum(ot_prob.geom.marginal_from_scalings(u, v))
  return div_a + div_b + ot_prob.epsilon * (
      jnp.sum(ot_prob.a) * jnp.sum(ot_prob.b) - total_sum
  )


class SinkhornOutput(NamedTuple):
  """Holds the output of a Sinkhorn solver applied to a problem.

  Objects of this class contain both solutions and problem definition of a
  regularized OT problem, along several methods that can be used to access its
  content, to, for instance, materialize an OT matrix or apply it to a vector
  (without having to materialize it when not needed).

  Args:
    potentials: list of optimal dual variables, two vector of size
      ``ot.prob.shape[0]`` and ``ot.prob.shape[1]`` returned by Sinkhorn
    errors: vector or errors, along iterations. This vector is of size
      ``max_iterations // inner_iterations`` where those were the parameters
      passed on to the :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.
      For each entry indexed at ``i``, ``errors[i]`` can be either a real
      non-negative value (meaning the algorithm recorded that error at the
      ``i * inner_iterations`` iteration), a ``jnp.inf`` value (meaning the
      algorithm computed that iteration but did not compute its error, because,
      for instance, ``i < min_iterations // inner_iterations``), or a ``-1``,
      meaning that execution was terminated before that iteration, because the
      criterion was found to be smaller than ``threshold``.
    reg_ot_cost: the regularized optimal transport cost. By default this is
      the linear contribution + KL term. See
      :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.ent_reg_cost`,
      :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.primal_cost` and
      :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.dual_cost` for other
      objective values.
    ot_prob: stores the definition of the OT problem, including geometry,
      marginals, unbalanced regularizers, etc.
    threshold: convergence threshold used to control the termination of the
      algorithm.
    converged: whether the output corresponds to a solution whose error is
      below the convergence threshold.
    inner_iterations: number of iterations that were run between two
      computations of errors.
  """

  potentials: Tuple[jnp.ndarray, ...]
  errors: Optional[jnp.ndarray] = None
  reg_ot_cost: Optional[float] = None
  ot_prob: Optional[linear_problem.LinearProblem] = None
  threshold: Optional[jnp.ndarray] = None
  converged: Optional[bool] = None
  inner_iterations: Optional[int] = None

  def set(self, **kwargs: Any) -> "SinkhornOutput":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def set_cost(  # noqa: D102
      self, ot_prob: linear_problem.LinearProblem, lse_mode: bool,
      use_danskin: bool
  ) -> "SinkhornOutput":
    f = jax.lax.stop_gradient(self.f) if use_danskin else self.f
    g = jax.lax.stop_gradient(self.g) if use_danskin else self.g
    return self.set(reg_ot_cost=compute_kl_reg_cost(f, g, ot_prob, lse_mode))

  @property
  def dual_cost(self) -> jnp.ndarray:
    """Return dual transport cost, without considering regularizer."""
    a, b = self.ot_prob.a, self.ot_prob.b
    dual_cost = jnp.sum(jnp.where(a > 0.0, a * self.f, 0))
    dual_cost += jnp.sum(jnp.where(b > 0.0, b * self.g, 0))
    return dual_cost

  @property
  def primal_cost(self) -> float:
    """Return transport cost of current transport solution at geometry."""
    return self.transport_cost_at_geom(other_geom=self.geom)

  @property
  def ent_reg_cost(self) -> float:
    r"""Entropy regularized cost.

    This outputs

    .. math::

      \langle P^{\star},C\rangle - \varepsilon H(P^{\star}) +
      \rho_a\text{KL}(P^{\star} 1|a) + \rho_b\text{KL}(1^T P^{\star}|b),

    where :math:`P^{\star}, a, b` is the coupling returned by the
    :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` and the two marginal weight
    vectors; :math:`\rho_a=\varepsilon \tau_a / (1-\tau_a)` and
    :math:`\rho_b=\varepsilon \tau_b / (1-\tau_b)` are obtained when the problem
    is unbalanced from parameters ``tau_a`` and ``tau_b``. Note that the last
    two terms vanish in the balanced case, when ``tau_a==tau_b==1``.
    """
    ent_a = jnp.sum(jsp.special.entr(self.ot_prob.a))
    ent_b = jnp.sum(jsp.special.entr(self.ot_prob.b))
    return self.reg_ot_cost - self.geom.epsilon * (ent_a + ent_b)

  @property
  def kl_reg_cost(self) -> float:
    r"""KL regularized OT transport cost.

    This outputs

    .. math::

      \langle P^{\star}, C \rangle + \varepsilon KL(P^{\star},ab^T) +
      \rho_a\text{KL}(P^{\star} 1|a) + \rho_b\text{KL}(1^T P^{\star}|b),

    where :math:`P^{\star}, a, b` are the coupling returned by the
    :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` algorithm and the two
    marginal weight vectors, respectively, and
    :math:`\rho_a=\varepsilon \tau_a / (1-\tau_a)` and
    :math:`\rho_b=\varepsilon \tau_b / (1-\tau_b)` are obtained when the problem
    is unbalanced from parameters ``tau_a`` and ``tau_b``. Note that the last
    two terms vanish in the balanced case, when ``tau_a==tau_b==1``. This
    quantity coincides with :attr:`reg_ot_cost`, which is computed using
    dual variables.
    """
    return self.reg_ot_cost

  def transport_cost_at_geom(
      self, other_geom: geometry.Geometry
  ) -> jnp.ndarray:
    r"""Return bare transport cost of current solution at any geometry.

    In order to compute cost, we check first if the geometry can be converted
    to a low-rank cost geometry in order to speed up computations, without
    having to materialize the full cost matrix. If this is not possible,
    we resort to instantiating both transport matrix and cost matrix.

    Args:
      other_geom: geometry whose cost matrix is used to evaluate the transport
        cost.

    Returns:
      the transportation cost at :math:`C`, i.e. :math:`\langle P, C \rangle`.
    """
    # TODO(cuturi): handle online mode for non Euclidean pointcloud geometries.
    # TODO(michalk8): handle SqEucl point cloud is not converted to LRCGeom
    if other_geom.can_LRC:
      geom = other_geom.to_LRCGeometry()
      return jnp.sum(self.apply(geom.cost_1.T) * geom.cost_2.T)
    return jnp.sum(self.matrix * other_geom.cost_matrix)

  @property
  def geom(self) -> geometry.Geometry:  # noqa: D102
    return self.ot_prob.geom

  @property
  def a(self) -> jnp.ndarray:  # noqa: D102
    return self.ot_prob.a

  @property
  def b(self) -> jnp.ndarray:  # noqa: D102
    return self.ot_prob.b

  @property
  def n_iters(self) -> int:  # noqa: D102
    """Returns the total number of iterations that were needed to terminate."""
    return jnp.sum(self.errors != -1) * self.inner_iterations

  @property
  def scalings(self) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
    u = self.ot_prob.geom.scaling_from_potential(self.f)
    v = self.ot_prob.geom.scaling_from_potential(self.g)
    return u, v

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix if it can be instantiated."""
    try:
      return self.ot_prob.geom.transport_from_potentials(self.f, self.g)
    except ValueError:
      return self.ot_prob.geom.transport_from_scalings(*self.scalings)

  @property
  def transport_mass(self) -> float:
    """Sum of transport matrix."""
    return self.marginal(0).sum()

  def apply(
      self,
      inputs: jnp.ndarray,
      axis: int = 0,
      lse_mode: bool = True
  ) -> jnp.ndarray:
    """Apply the transport to a ndarray; axis=1 for its transpose."""
    geom = self.ot_prob.geom
    if lse_mode:
      return geom.apply_transport_from_potentials(
          self.f, self.g, inputs, axis=axis
      )
    u = geom.scaling_from_potential(self.f)
    v = geom.scaling_from_potential(self.g)
    return geom.apply_transport_from_scalings(u, v, inputs, axis=axis)

  def marginal(self, axis: int) -> jnp.ndarray:  # noqa: D102
    return self.ot_prob.geom.marginal_from_potentials(self.f, self.g, axis=axis)

  def cost_at_geom(self, other_geom: geometry.Geometry) -> float:
    """Return reg-OT cost for matrix, evaluated at other cost matrix."""
    return (
        jnp.sum(self.matrix * other_geom.cost_matrix) -
        self.geom.epsilon * jnp.sum(jax.scipy.special.entr(self.matrix))
    )

  def to_dual_potentials(self) -> potentials.EntropicPotentials:
    """Return the entropic map estimator."""
    return potentials.EntropicPotentials(self.f, self.g, self.ot_prob)

  @property
  def f(self) -> jnp.ndarray:
    """The first dual potential."""
    return self.potentials[0]

  @property
  def g(self) -> jnp.ndarray:
    """The second dual potential."""
    return self.potentials[1]


@jax.tree_util.register_pytree_node_class
class Sinkhorn:
  r"""Sinkhorn solver.

  The Sinkhorn algorithm is a fixed point iteration that solves a regularized
  optimal transport (reg-OT) problem between two measures.
  The optimization variables are a pair of vectors (called potentials, or
  scalings when parameterized as exponential of the former). Calling this
  function returns therefore a pair of optimal vectors. In addition to these,
  it also returns the objective value achieved by these optimal vectors;
  a vector of size ``max_iterations/inner_iterations`` that records the vector
  of values recorded to monitor convergence, throughout the execution of the
  algorithm (padded with `-1` if convergence happens before), as well as a
  boolean to signify whether the algorithm has converged within the number of
  iterations specified by the user.

  The reg-OT problem is specified by two measures, of respective sizes ``n`` and
  ``m``. From the viewpoint of the ``sinkhorn`` function, these two measures are
  only seen through a triplet (``geom``, ``a``, ``b``), where ``geom`` is a
  ``Geometry`` object, and ``a`` and ``b`` are weight vectors of respective
  sizes ``n`` and ``m``. Starting from two initial values for those potentials
  or scalings (both can be defined by the user by passing value in
  ``init_dual_a`` or ``init_dual_b``), the Sinkhorn algorithm will use
  elementary operations that are carried out by the ``geom`` object.

  Math:
    Given a geometry ``geom``, which provides a cost matrix :math:`C` with its
    regularization parameter :math:`\varepsilon`, (or a kernel matrix :math:`K`)
    the reg-OT problem consists in finding two vectors `f`, `g` of size ``n``,
    ``m`` that maximize the following criterion.

    .. math::

      \arg\max_{f, g}{- \langle a, \phi_a^{*}(-f) \rangle -  \langle b,
      \phi_b^{*}(-g) \rangle - \varepsilon \langle e^{f/\varepsilon},
      e^{-C/\varepsilon} e^{g/\varepsilon}} \rangle

    where :math:`\phi_a(z) = \rho_a z(\log z - 1)` is a scaled entropy, and
    :math:`\phi_a^{*}(z) = \rho_a e^{z/\varepsilon}`, its Legendre transform
    :cite:`sejourne:19`.

    That problem can also be written, instead, using positive scaling vectors
    `u`, `v` of size ``n``, ``m``, handled with the kernel
    :math:`K := e^{-C/\varepsilon}`,

    .. math::

      \arg\max_{u, v >0} - \langle a,\phi_a^{*}(-\varepsilon\log u) \rangle +
      \langle b, \phi_b^{*}(-\varepsilon\log v) \rangle - \langle u, K v \rangle

    Both of these problems corresponds, in their *primal* formulation, to
    solving the unbalanced optimal transport problem with a variable matrix
    :math:`P` of size ``n`` x ``m``:

    .. math::

      \arg\min_{P>0} \langle P,C \rangle +\varepsilon \text{KL}(P | ab^T)
      + \rho_a \text{KL}(P\mathbf{1}_m | a) + \rho_b \text{KL}(P^T \mathbf{1}_n
      | b)

    where :math:`KL` is the generalized Kullback-Leibler divergence.

    The very same primal problem can also be written using a kernel :math:`K`
    instead of a cost :math:`C` as well:

    .. math::

      \arg\min_{P} \varepsilon \text{KL}(P|K)
      + \rho_a \text{KL}(P\mathbf{1}_m | a) +
      \rho_b \text{KL}(P^T \mathbf{1}_n | b)

    The *original* OT problem taught in linear programming courses is recovered
    by using the formulation above relying on the cost :math:`C`, and letting
    :math:`\varepsilon \rightarrow 0`, and :math:`\rho_a, \rho_b \rightarrow
    \infty`.
    In that case the entropy disappears, whereas the :math:`KL` regularization
    above become constraints on the marginals of :math:`P`: This results in a
    standard min cost flow problem. This problem is not handled for now in this
    toolbox, which focuses exclusively on the case :math:`\varepsilon > 0`.

    The *balanced* regularized OT problem is recovered for finite
    :math:`\varepsilon > 0` but letting :math:`\rho_a, \rho_b \rightarrow
    \infty`. This problem can be shown to be equivalent to a matrix scaling
    problem, which can be solved using the Sinkhorn fixed-point algorithm.
    To handle the case :math:`\rho_a, \rho_b \rightarrow \infty`, the
    ``sinkhorn`` function uses parameters ``tau_a`` and ``tau_b`` equal
    respectively to :math:`\rho_a /(\varepsilon + \rho_a)` and
    :math:`\rho_b / (\varepsilon + \rho_b)` instead. Setting either of these
    parameters to 1 corresponds to setting the corresponding
    :math:`\rho_a, \rho_b` to :math:`\infty`.

    The Sinkhorn algorithm solves the reg-OT problem by seeking optimal
    :math:`f`, :math:`g` potentials (or alternatively their parameterization
    as positive scaling vectors :math:`u`, :math:`v`), rather than solving the
    primal problem in :math:`P`. This is mostly for efficiency (potentials and
    scalings have a ``n + m`` memory footprint, rather than ``n m`` required
    to store `P`). This is also because both problems are, in fact, equivalent,
    since the optimal transport :math:`P^{\star}` can be recovered from
    optimal potentials :math:`f^{\star}`, :math:`g^{\star}` or scaling
    :math:`u^{\star}`, :math:`v^{\star}`, using the geometry's cost or kernel
    matrix respectively:

    .. math::

      P^{\star} = \exp\left(\frac{f^{\star}\mathbf{1}_m^T + \mathbf{1}_n g^{*T}-
      C}{\varepsilon}\right) \text{ or } P^{\star} = \text{diag}(u^{\star}) K
      \text{diag}(v^{\star})

    By default, the Sinkhorn algorithm solves this dual problem in :math:`f, g`
    or :math:`u, v` using block coordinate ascent, i.e. devising an update for
    each :math:`f` and :math:`g` (resp. :math:`u` and :math:`v`) that cancels
    their respective gradients, one at a time. These two iterations are repeated
    ``inner_iterations`` times, after which the norm of these gradients will be
    evaluated and compared with the ``threshold`` value. The iterations are then
    repeated as long as that error exceeds ``threshold``.

  Note on Sinkhorn updates:
    The boolean flag ``lse_mode`` sets whether the algorithm is run in either:

    - log-sum-exp mode (``lse_mode=True``), in which case it is directly
      defined in terms of updates to `f` and `g`, using log-sum-exp
      computations. This requires access to the cost matrix :math:`C`, as it is
      stored, or possibly computed on the fly by ``geom``.

    - kernel mode (``lse_mode=False``), in which case it will require access
      to a matrix vector multiplication operator :math:`z \rightarrow K z`,
      where :math:`K` is either instantiated from :math:`C` as
      :math:`\exp(-C/\varepsilon)`, or provided directly. In that case, rather
      than optimizing on :math:`f` and :math:`g`, it is more convenient to
      optimize on their so called scaling formulations,
      :math:`u := \exp(f / \varepsilon)` and :math:`v := \exp(g / \varepsilon)`.
      While faster (applying matrices is faster than applying ``lse`` repeatedly
      over lines), this mode is also less stable numerically, notably for
      smaller :math:`\varepsilon`.

    In the source code, the variables ``f_u`` or ``g_v`` can be either regarded
    as potentials (real) or scalings (positive) vectors, depending on the choice
    of ``lse_mode`` by the user. Once optimization is carried out, we only
    return dual variables in potential form, i.e. ``f`` and ``g``.

    In addition to standard Sinkhorn updates, the user can also use heavy-ball
    type updates using a ``momentum`` parameter in ]0,2[. We also implement a
    strategy that tries to set that parameter adaptively at
    ``chg_momentum_from`` iterations, as a function of progress in the error,
    as proposed in the literature.

    Another upgrade to the standard Sinkhorn updates provided to the users lies
    in using Anderson acceleration. This can be parameterized by setting the
    otherwise null ``anderson`` to a positive integer. When selected,the
    algorithm will recompute, every ``refresh_anderson_frequency`` (set by
    default to 1) an extrapolation of the most recently computed ``anderson``
    iterates. When using that option, notice that differentiation (if required)
    can only be carried out using implicit differentiation, and that all
    momentum related parameters are ignored.

    The ``parallel_dual_updates`` flag is set to ``False`` by default. In that
    setting, ``g_v`` is first updated using the latest values for ``f_u`` and
    ``g_v``, before proceeding to update ``f_u`` using that new value for
    ``g_v``. When the flag is set to ``True``, both ``f_u`` and ``g_v`` are
    updated simultaneously. Note that setting that choice to ``True`` requires
    using some form of averaging (e.g. ``momentum=0.5``). Without this, and on
    its own ``parallel_dual_updates`` won't work.

  Differentiation:
    The optimal solutions ``f`` and ``g`` and the optimal objective
    (``reg_ot_cost``) outputted by the Sinkhorn algorithm can be differentiated
    w.r.t. relevant inputs ``geom``, ``a`` and ``b``. In the default setting,
    implicit differentiation of the optimality conditions (``implicit_diff``
    not equal to ``None``), this has two consequences, treating ``f`` and ``g``
    differently from ``reg_ot_cost``.

    - The termination criterion used to stop Sinkhorn (cancellation of
      gradient of objective w.r.t. ``f_u`` and ``g_v``) is used to differentiate
      ``f`` and ``g``, given a change in the inputs. These changes are computed
      by solving a linear system. The arguments starting with
      ``implicit_solver_*`` allow to define the linear solver that is used, and
      to control for two types or regularization (we have observed that,
      depending on the architecture, linear solves may require higher ridge
      parameters to remain stable). The optimality conditions in Sinkhorn can be
      analyzed as satisfying a ``z=z'`` condition, which are then
      differentiated. It might be beneficial (e.g., as in :cite:`cuturi:20a`)
      to use a preconditioning function ``precondition_fun`` to differentiate
      instead ``h(z) = h(z')``.

    - The objective ``reg_ot_cost`` returned by Sinkhorn uses the so-called
      envelope (or Danskin's) theorem. In that case, because it is assumed that
      the gradients of the dual variables ``f_u`` and ``g_v`` w.r.t. dual
      objective are zero (reflecting the fact that they are optimal), small
      variations in ``f_u`` and ``g_v`` due to changes in inputs (such as
      ``geom``, ``a`` and ``b``) are considered negligible. As a result,
      ``stop_gradient`` is applied on dual variables ``f_u`` and ``g_v`` when
      evaluating the ``reg_ot_cost`` objective. Note that this approach is
      `invalid` when computing higher order derivatives. In that case the
      ``use_danskin`` flag must be set to ``False``.

    An alternative yet more costly way to differentiate the outputs of the
    Sinkhorn iterations is to use unrolling, i.e. reverse mode differentiation
    of the Sinkhorn loop. This is possible because Sinkhorn iterations are
    wrapped in a custom fixed point iteration loop, defined in
    ``fixed_point_loop``, rather than a standard while loop. This is to ensure
    the end result of this fixed point loop can also be differentiated, if
    needed, using standard JAX operations. To ensure differentiability,
    the ``fixed_point_loop.fixpoint_iter_backprop`` loop does checkpointing of
    state variables (here ``f_u`` and ``g_v``) every ``inner_iterations``, and
    backpropagates automatically, block by block, through blocks of
    ``inner_iterations`` at a time.

  Note:
    * The Sinkhorn algorithm may not converge within the maximum number of
      iterations for possibly several reasons:

      1. the regularizer (defined as ``epsilon`` in the geometry ``geom``
         object) is too small. Consider either switching to ``lse_mode=True``
         (at the price of a slower execution), increasing ``epsilon``, or,
         alternatively, if you are unable or unwilling to increase  ``epsilon``,
         either increase ``max_iterations`` or ``threshold``.
      2. the probability weights ``a`` and ``b`` do not have the same total
         mass, while using a balanced (``tau_a=tau_b=1.0``) setup.
         Consider either normalizing ``a`` and ``b``, or set either ``tau_a``
         and/or ``tau_b<1.0``.
      3. OOMs issues may arise when storing either cost or kernel matrices that
         are too large in ``geom``. In the case where, the ``geom`` geometry is
         a ``PointCloud``, some of these issues might be solved by setting the
         ``online`` flag to ``True``. This will trigger a re-computation on the
         fly of the cost/kernel matrix.

    * The weight vectors ``a`` and ``b`` can be passed on with coordinates that
      have zero weight. This is then handled by relying on simple arithmetic for
      ``inf`` values that will likely arise (due to :math:`\log 0` when
      ``lse_mode`` is ``True``, or divisions by zero when ``lse_mode`` is
      ``False``). Whenever that arithmetic is likely to produce ``NaN`` values
      (due to ``-inf * 0``, or ``-inf - -inf``) in the forward pass, we use
      ``jnp.where`` conditional statements to carry ``inf`` rather than ``NaN``
      values. In the reverse mode differentiation, the inputs corresponding to
      these 0 weights (a location `x`, or a row in the corresponding cost/kernel
      matrix), and the weight itself will have ``NaN`` gradient values. This is
      reflects that these gradients are undefined, since these points were not
      considered in the optimization and have therefore no impact on the output.

  Args:
    lse_mode: ``True`` for log-sum-exp computations, ``False`` for kernel
      multiplication.
    threshold: tolerance used to stop the Sinkhorn iterations. This is
      typically the deviation between a target marginal and the marginal of the
      current primal solution when either or both tau_a and tau_b are 1.0
      (balanced or semi-balanced problem), or the relative change between two
      successive solutions in the unbalanced case.
    norm_error: power used to define p-norm of error for marginal/target.
    inner_iterations: the Sinkhorn error is not recomputed at each
      iteration but every ``inner_iterations`` instead.
    min_iterations: the minimum number of Sinkhorn iterations carried
      out before the error is computed and monitored.
    max_iterations: the maximum number of Sinkhorn iterations. If
      ``max_iterations`` is equal to ``min_iterations``, Sinkhorn iterations are
      run by default using a :func:`jax.lax.scan` loop rather than a custom,
      unroll-able :func:`jax.lax.while_loop` that monitors convergence.
      In that case the error is not monitored and the ``converged``
      flag will return ``False`` as a consequence.
    momentum: Momentum instance.
    anderson: AndersonAcceleration instance.
    implicit_diff: instance used to solve implicit differentiation. Unrolls
      iterations if None.
    parallel_dual_updates: updates potentials or scalings in parallel if True,
      sequentially (in Gauss-Seidel fashion) if False.
    recenter_potentials: Whether to re-center the dual potentials.
      If the problem is balanced, the ``f`` potential is zero-centered for
      numerical stability. Otherwise, use the approach of :cite:`sejourne:22`
      to achieve faster convergence. Only used when ``lse_mode = True`` and
      ``tau_a < 1`` and ``tau_b < 1``.
    use_danskin: when ``True``, it is assumed the entropy regularized cost
      is evaluated using optimal potentials that are frozen, i.e. whose
      gradients have been stopped. This is useful when carrying out first order
      differentiation, and is only valid (as with ``implicit_differentiation``)
      when the algorithm has converged with a low tolerance.
    initializer: how to compute the initial potentials/scalings. This refers to
      a few possible classes implemented following the template in
      :class:`~ott.initializers.linear.SinkhornInitializer`.
    progress_fn: callback function which gets called during the Sinkhorn
      iterations, so the user can display the error at each iteration,
      e.g., using a progress bar. See :func:`~ott.utils.default_progress_fn`
      for a basic implementation.
    kwargs_init: keyword arguments when creating the initializer.
  """

  def __init__(
      self,
      lse_mode: bool = True,
      threshold: float = 1e-3,
      norm_error: int = 1,
      inner_iterations: int = 10,
      min_iterations: int = 0,
      max_iterations: int = 2000,
      momentum: Optional[acceleration.Momentum] = None,
      anderson: Optional[acceleration.AndersonAcceleration] = None,
      parallel_dual_updates: bool = False,
      recenter_potentials: bool = False,
      use_danskin: Optional[bool] = None,
      implicit_diff: Optional[implicit_lib.ImplicitDiff
                             ] = implicit_lib.ImplicitDiff(),  # noqa: B008
      initializer: Union[Literal["default", "gaussian", "sorting", "subsample"],
                         init_lib.SinkhornInitializer] = "default",
      progress_fn: Optional[ProgressCallbackFn_t] = None,
      kwargs_init: Optional[Mapping[str, Any]] = None,
  ):
    self.lse_mode = lse_mode
    self.threshold = threshold
    self.inner_iterations = inner_iterations
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self._norm_error = norm_error
    self.anderson = anderson
    self.implicit_diff = implicit_diff

    if momentum is not None:
      self.momentum = acceleration.Momentum(
          momentum.start, momentum.error_threshold, momentum.value,
          self.inner_iterations
      )
    else:
      # Use no momentum if using Anderson or unrolling.
      if self.anderson is not None or self.implicit_diff is None:
        self.momentum = acceleration.Momentum(
            inner_iterations=self.inner_iterations
        )
      else:
        # no momentum
        self.momentum = acceleration.Momentum()

    self.parallel_dual_updates = parallel_dual_updates
    self.recenter_potentials = recenter_potentials
    self.initializer = initializer
    self.progress_fn = progress_fn
    self.kwargs_init = {} if kwargs_init is None else kwargs_init

    # Force implicit_differentiation to True when using Anderson acceleration,
    # Reset all momentum parameters to default (i.e. no momentum)
    if anderson:
      self.implicit_diff = (
          implicit_lib.ImplicitDiff()
          if self.implicit_diff is None else self.implicit_diff
      )
      self.momentum = acceleration.Momentum(
          inner_iterations=self.inner_iterations
      )

    # By default, use Danskin theorem to differentiate
    # the objective when using implicit_lib.
    self.use_danskin = ((self.implicit_diff is not None)
                        if use_danskin is None else use_danskin)

  def __call__(
      self,
      ot_prob: linear_problem.LinearProblem,
      init: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]] = (None, None),
      rng: Optional[jax.Array] = None,
  ) -> SinkhornOutput:
    """Run Sinkhorn algorithm.

    Args:
      ot_prob: Linear OT problem.
      init: Initial dual potentials/scalings f_u and g_v, respectively.
        Any `None` values will be initialized using the initializer.
      rng: Random number generator key for stochastic initialization.

    Returns:
      The Sinkhorn output.
    """
    rng = utils.default_prng_key(rng)
    initializer = self.create_initializer()
    init_dual_a, init_dual_b = initializer(
        ot_prob, *init, lse_mode=self.lse_mode, rng=rng
    )
    return run(ot_prob, self, (init_dual_a, init_dual_b))

  def lse_step(
      self, ot_prob: linear_problem.LinearProblem, state: SinkhornState,
      iteration: int
  ) -> SinkhornState:
    """Sinkhorn LSE update."""

    def k(tau_i: float, tau_j: float) -> float:
      num = -tau_j * (tau_a - 1) * (tau_b - 1) * (tau_i - 1)
      denom = (tau_j - 1) * (tau_a * (tau_b - 1) + tau_b * (tau_a - 1))
      return num / denom

    def xi(tau_i: float, tau_j: float) -> float:
      k_ij = k(tau_i, tau_j)
      return k_ij / (1.0 - k_ij)

    def smin(
        potential: jnp.ndarray, marginal: jnp.ndarray, tau: float
    ) -> float:
      rho = uf.rho(ot_prob.epsilon, tau)
      return -rho * mu.logsumexp(-potential / rho, b=marginal)

    # only for an unbalanced problems with `tau_{a,b} < 1`
    recenter = (
        self.recenter_potentials and ot_prob.tau_a < 1.0 and ot_prob.tau_b < 1.0
    )
    w = self.momentum.weight(state, iteration)
    tau_a, tau_b = ot_prob.tau_a, ot_prob.tau_b
    old_fu, old_gv = state.fu, state.gv

    if recenter:
      k11, k22 = k(tau_a, tau_a), k(tau_b, tau_b)
      xi12, xi21 = xi(tau_a, tau_b), xi(tau_b, tau_a)

    # update g potential
    new_gv = tau_b * ot_prob.geom.update_potential(
        old_fu, old_gv, jnp.log(ot_prob.b), iteration, axis=0
    )
    if recenter:
      new_gv -= k22 * smin(old_fu, ot_prob.a, tau_a)
      new_gv += xi21 * smin(new_gv, ot_prob.b, tau_b)
    gv = self.momentum(w, old_gv, new_gv, self.lse_mode)

    if not self.parallel_dual_updates:
      old_gv = gv

    # update f potential
    new_fu = tau_a * ot_prob.geom.update_potential(
        old_fu, old_gv, jnp.log(ot_prob.a), iteration, axis=1
    )
    if recenter:
      new_fu -= k11 * smin(old_gv, ot_prob.b, tau_b)
      new_fu += xi12 * smin(new_fu, ot_prob.a, tau_a)
    fu = self.momentum(w, old_fu, new_fu, self.lse_mode)

    return state.set(potentials=(fu, gv))

  def kernel_step(
      self, ot_prob: linear_problem.LinearProblem, state: SinkhornState,
      iteration: int
  ) -> SinkhornState:
    """Sinkhorn multiplicative update."""
    w = self.momentum.weight(state, iteration)
    old_gv = state.gv
    new_gv = ot_prob.geom.update_scaling(
        state.fu, ot_prob.b, iteration, axis=0
    ) ** ot_prob.tau_b
    gv = self.momentum(w, state.gv, new_gv, self.lse_mode)
    new_fu = ot_prob.geom.update_scaling(
        old_gv if self.parallel_dual_updates else gv,
        ot_prob.a,
        iteration,
        axis=1
    ) ** ot_prob.tau_a
    fu = self.momentum(w, state.fu, new_fu, self.lse_mode)
    return state.set(potentials=(fu, gv))

  def one_iteration(
      self, ot_prob: linear_problem.LinearProblem, state: SinkhornState,
      iteration: int, compute_error: bool
  ) -> SinkhornState:
    """Carries out one Sinkhorn iteration.

    Depending on lse_mode, these iterations can be either in:

      - log-space for numerical stability.
      - scaling space, using standard kernel-vector multiply operations.

    Args:
      ot_prob: the transport problem definition
      state: SinkhornState named tuple.
      iteration: the current iteration of the Sinkhorn loop.
      compute_error: flag to indicate this iteration computes/stores an error

    Returns:
      The updated state.
    """
    # When running updates in parallel (Gauss-Seidel mode), old_g_v will be
    # used to update f_u, rather than the latest g_v computed in this loop.
    # Unused otherwise.
    if self.anderson:
      state = self.anderson.update(state, iteration, ot_prob, self.lse_mode)

    if self.lse_mode:  # In lse_mode, run additive updates.
      state = self.lse_step(ot_prob, state, iteration)
    else:
      state = self.kernel_step(ot_prob, state, iteration)

    if self.anderson:
      state = self.anderson.update_history(state, ot_prob, self.lse_mode)

    # re-computes error if compute_error is True, else set it to inf.
    err = jax.lax.cond(
        jnp.logical_or(
            iteration == self.max_iterations - 1,
            jnp.logical_and(compute_error, iteration >= self.min_iterations)
        ),
        lambda state, prob: state.solution_error(
            prob,
            self.norm_error,
            lse_mode=self.lse_mode,
            parallel_dual_updates=self.parallel_dual_updates,
            recenter=self.recenter_potentials
        )[0],
        lambda *_: jnp.inf,
        state,
        ot_prob,
    )
    errors = state.errors.at[iteration // self.inner_iterations, :].set(err)
    state = state.set(errors=errors)

    if self.progress_fn is not None:
      jax.debug.callback(
          self.progress_fn,
          (iteration, self.inner_iterations, self.max_iterations, state)
      )
    return state

  def _converged(self, state: SinkhornState, iteration: int) -> bool:
    err = state.errors[iteration // self.inner_iterations - 1, 0]
    return jnp.logical_and(iteration > 0, err < self.threshold)

  def _diverged(self, state: SinkhornState, iteration: int) -> bool:
    err = state.errors[iteration // self.inner_iterations - 1, 0]
    return jnp.logical_not(jnp.isfinite(err))

  def _continue(self, state: SinkhornState, iteration: int) -> bool:
    """Continue while not(converged) and not(diverged)."""
    return jnp.logical_and(
        jnp.logical_not(self._diverged(state, iteration)),
        jnp.logical_not(self._converged(state, iteration))
    )

  @property
  def outer_iterations(self) -> int:
    """Upper bound on number of times inner_iterations are carried out.

    This integer can be used to set constant array sizes to track the algorithm
    progress, notably errors.
    """
    return np.ceil(self.max_iterations / self.inner_iterations).astype(int)

  def init_state(
      self, ot_prob: linear_problem.LinearProblem, init: Tuple[jnp.ndarray,
                                                               jnp.ndarray]
  ) -> SinkhornState:
    """Return the initial state of the loop."""
    errors = -jnp.ones((self.outer_iterations, len(self.norm_error)))
    state = SinkhornState(init, errors=errors)
    return self.anderson.init_maps(ot_prob, state) if self.anderson else state

  def output_from_state(
      self, ot_prob: linear_problem.LinearProblem, state: SinkhornState
  ) -> SinkhornOutput:
    """Create an output from a loop state.

    Note:
      When differentiating the regularized OT cost, and assuming Sinkhorn has
      run to convergence, Danskin's (or the envelope)
      `theorem <https://en.wikipedia.org/wiki/Danskin%27s_theorem>`_
      :cite:`danskin:67,bertsekas:71`
      states that the resulting OT cost as a function of the inputs
      (``geometry``, ``a``, ``b``) behaves locally as if the dual optimal
      potentials were frozen and did not vary with those inputs.

      Notice this is only valid, as when using ``implicit_differentiation``
      mode, if the Sinkhorn algorithm outputs potentials that are near optimal.
      namely when the threshold value is set to a small tolerance.

      The flag ``use_danskin`` controls whether that assumption is made. By
      default, that flag is set to the value of ``implicit_differentiation`` if
      not specified. If you wish to compute derivatives of order 2 and above,
      set ``use_danskin`` to ``False``.

    Args:
      ot_prob: the transport problem.
      state: a SinkhornState.

    Returns:
      A SinkhornOutput.
    """
    geom = ot_prob.geom

    f = state.fu if self.lse_mode else geom.potential_from_scaling(state.fu)
    g = state.gv if self.lse_mode else geom.potential_from_scaling(state.gv)
    if self.recenter_potentials:
      f, g = state.recenter(f, g, ot_prob=ot_prob)

    # By convention, the algorithm is said to have converged if the algorithm
    # has not nan'ed during iterations (notice some errors might be infinite,
    # this convention is used when the error is not recomputed), and if the
    # last recorded error is lower than the threshold. Note that this will be
    # the case if either the algorithm terminated earlier (in which case the
    # last state.errors[-1] = -1 by convention) or if the algorithm carried out
    # the maximal number of iterations and its last recorded error (at -1
    # position) is lower than the threshold.

    converged = jnp.logical_and(
        jnp.logical_not(jnp.any(jnp.isnan(state.errors))), state.errors[-1]
        < self.threshold
    )[0]

    return SinkhornOutput((f, g),
                          errors=state.errors[:, 0],
                          threshold=jnp.array(self.threshold),
                          converged=converged,
                          inner_iterations=self.inner_iterations)

  @property
  def norm_error(self) -> Tuple[int, ...]:
    """Powers used to compute the p-norm between marginal/target."""
    # To change momentum adaptively, one needs errors in ||.||_1 norm.
    # In that case, we add this exponent to the list of errors to compute,
    # notably if that was not the error requested by the user.
    if self.momentum and self.momentum.start > 0 and self._norm_error != 1:
      return self._norm_error, 1
    return self._norm_error,

  # TODO(michalk8): in the future, enforce this (+ in GW) via abstract method
  def create_initializer(self) -> init_lib.SinkhornInitializer:  # noqa: D102
    if isinstance(self.initializer, init_lib.SinkhornInitializer):
      return self.initializer
    if self.initializer == "default":
      return init_lib.DefaultInitializer()
    if self.initializer == "gaussian":
      return init_lib.GaussianInitializer()
    if self.initializer == "sorting":
      return init_lib.SortingInitializer(**self.kwargs_init)
    if self.initializer == "subsample":
      return init_lib.SubsampleInitializer(**self.kwargs_init)
    raise NotImplementedError(
        f"Initializer `{self.initializer}` is not yet implemented."
    )

  def tree_flatten(self):  # noqa: D102
    aux = vars(self).copy()
    aux["norm_error"] = aux.pop("_norm_error")
    aux.pop("threshold")
    return [self.threshold], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(**aux_data, threshold=children[0])


def run(
    ot_prob: linear_problem.LinearProblem, solver: Sinkhorn,
    init: Tuple[jnp.ndarray, ...]
) -> SinkhornOutput:
  """Run loop of the solver, outputting a state upgraded to an output."""
  iter_fun = _iterations_implicit if solver.implicit_diff else iterations
  out = iter_fun(ot_prob, solver, init)
  # Be careful here, the geom and the cost are injected at the end, where it
  # does not interfere with the implicit differentiation.
  out = out.set_cost(ot_prob, solver.lse_mode, solver.use_danskin)
  return out.set(ot_prob=ot_prob)


def iterations(
    ot_prob: linear_problem.LinearProblem, solver: Sinkhorn,
    init: Tuple[jnp.ndarray, ...]
) -> SinkhornOutput:
  """Jittable Sinkhorn loop. args contain initialization variables."""

  def cond_fn(
      iteration: int, const: Tuple[linear_problem.LinearProblem, Sinkhorn],
      state: SinkhornState
  ) -> bool:
    _, solver = const
    return solver._continue(state, iteration)

  def body_fn(
      iteration: int, const: Tuple[linear_problem.LinearProblem, Sinkhorn],
      state: SinkhornState, compute_error: bool
  ) -> SinkhornState:
    ot_prob, solver = const
    return solver.one_iteration(ot_prob, state, iteration, compute_error)

  # Run the Sinkhorn loop. Choose either a standard fixpoint_iter loop if
  # differentiation is implicit, otherwise switch to the backprop friendly
  # version of that loop if unrolling to differentiate.
  if solver.implicit_diff:
    fix_point = fixed_point_loop.fixpoint_iter
  else:
    fix_point = fixed_point_loop.fixpoint_iter_backprop

  const = ot_prob, solver
  state = solver.init_state(ot_prob, init)
  state = fix_point(
      cond_fn, body_fn, solver.min_iterations, solver.max_iterations,
      solver.inner_iterations, const, state
  )
  return solver.output_from_state(ot_prob, state)


def _iterations_taped(
    ot_prob: linear_problem.LinearProblem, solver: Sinkhorn,
    init: Tuple[jnp.ndarray, ...]
) -> Tuple[SinkhornOutput, Tuple[jnp.ndarray, jnp.ndarray,
                                 linear_problem.LinearProblem, Sinkhorn]]:
  """Run forward pass of the Sinkhorn algorithm storing side information."""
  state = iterations(ot_prob, solver, init)
  return state, (state.f, state.g, ot_prob, solver)


def _iterations_implicit_bwd(res, gr: SinkhornOutput):
  """Run Sinkhorn in backward mode, using implicit differentiation.

  Args:
    res: residual data sent from fwd pass, used for computations below. In this
      case consists in the output itself, as well as inputs against which we
      wish to differentiate.
    gr: gradients w.r.t outputs of fwd pass, here w.r.t size f, g, errors. Note
      that differentiability w.r.t. errors is not handled, and only f, g is
      considered.

  Returns:
    a tuple of gradients: PyTree for geom, one jnp.ndarray for each of a and b.
  """
  f, g, ot_prob, solver = res
  out = solver.implicit_diff.gradient(
      ot_prob, f, g, solver.lse_mode, gr.potentials
  )
  return *out, None, None


# sets threshold, norm_errors, geom, a and b to be differentiable, as those are
# non-static. Only differentiability w.r.t. geom, a and b will be used.
_iterations_implicit = jax.custom_vjp(iterations)
_iterations_implicit.defvjp(_iterations_taped, _iterations_implicit_bwd)
