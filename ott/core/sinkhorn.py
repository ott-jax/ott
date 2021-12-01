# coding=utf-8
# Copyright 2021 Google LLC.
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
r"""A Jax version of Sinkhorn's algorithm.

Solves regularized OT problem using Sinkhorn iterations.

The Sinkhorn algorithm is a fixed point iteration that solves a regularized
optimal transport (reg-OT) problem between two measures.
The optimization variables are a pair of vectors (called potentials, or
scalings when parameterized as exponentials of the former). Calling this
function returns therefore a pair of optimal vectors. In addition to these,
`sinkhorn` also returns the objective value achieved by these optimal vectors;
a vector of size `max_iterations/inner_terations` that records the vector of
values recorded to monitor convergence, throughout the execution of the
algorithm (padded with ``-1`` if convergence happens before), as well as a
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

Some maths:
  Given a geometry ``geom``, which provides a cost matrix :math:`C` with its
  regularization parameter :math:`\epsilon`, (resp. a kernel matrix :math:`K`)
  the reg-OT problem consists in finding two vectors `f`, `g` of size ``n``,
  ``m`` that maximize the following criterion.

  :math:`\arg\max_{f, g}{- <a, \phi_a^{*}(-f)> - <b, \phi_b^{*}(-g)> - \epsilon
  <e^{f/\epsilon}, e^{-C/\epsilon} e^{-g/\epsilon}}>`

  where :math:`\phi_a(z) = \rho_a z(\log z - 1)` is a scaled entropy, and
  :math:`\phi_a^{*}(z) = \rho_a e^{z/\varepsilon}` its Legendre transform.

  That problem can also be written, instead, using positive scaling vectors
  `u`, `v` of size ``n``, ``m``, handled with the kernel :math:`K:=e^{-C/\epsilon}`,
  :math:`\arg\max_{u, v >0} - <a,\phi_a^{*}(-\epsilon\log u)> + <b, \phi_b^{*}(-\epsilon\log v)> -  <u, K
  v>`

  Both of these problems corresponds, in their *primal* formulation, to solving the
  unbalanced optimal transport problem with a variable matrix `P` of size ``n``
  x ``m``:

  :math:`\arg\min_{P>0} <P,C> -\epsilon \text{KL}(P | ab^T) + \rho_a \text{KL}(P1 | a) + \rho_b \text{KL}(P^T1 | b)`

  where :math:`KL` is the generalized Kullback-Leibler divergence.

  The very same primal problem can also be written using a kernel :math:`K`
  instead of a cost :math:`C` as well:

  :math:`\arg\min_{P} \epsilon KL(P|K) + \rho_a \text{KL}(P1 | a) + \rho_b \text{KL}(P^T1 | b)`

  The *original* OT problem taught in linear programming courses is recovered
  by using the formulation above relying on the cost :math:`C`, and letting
  :math:`\epsilon \rightarrow 0`, and :math:`\rho_a, \rho_b \rightarrow \infty`.
  In that case the entropy disappears, whereas the :math:`KL` regularizations
  above become constraints on the marginals of :math:`P`: This results in a
  standard min cost flow problem. This problem is not handled for now in this
  toolbox, which focuses exclusively on the case :math:`\epsilon > 0`.

  The *balanced* regularized OT problem is recovered for finite
  :math:`\epsilon > 0` but letting :math:`\rho_a, \rho_b \rightarrow \infty`.
  This problem can be shown to be equivalent to a matrix scaling problem,
  which can be solved using the Sinkhorn fixed-point algorithm. To handle the
  case :math:`\rho_a, \rho_b \rightarrow \infty`, the ``sinkhorn`` function
  uses parameters ``tau_a`` := :math:`\rho_a / (\epsilon + \rho_a)` and
  ``tau_b`` := :math:`\rho_b / (\epsilon + \rho_b)` instead.
  Setting either of these parameters to 1 corresponds to setting the
  corresponding :math:`\rho_a, \rho_b` to :math:`\infty`.

  The Sinkhorn algorithm solves the reg-OT problem by seeking optimal `f`, `g`
  potentials (or alternatively their parameterization as positive scalings `u`,
  `v`), rather than solving the primal problem in :math:`P`. This is mostly for
  efficiency (potentials and scalings have a ``n + m`` memory footprint, rather
  than ``n m`` required to store `P`). This is also because both problems are,
  in fact, equivalent, since the optimal transport :math:`P^*` can be recovered
  from optimal potentials :math:`f^*`, :math:`g^*` or scalings :math:`u^*`,
  :math:`v^*`, using the geometry's cost or kernel matrix respectively:

    :math:`P^* = \exp\left(\frac{f^*\mathbf{1}_m^T + \mathbf{1}_n g^{*T} - C}{\epsilon}\right) \text{ or } P^* =
    \text{diag}(u^*) K \text{diag}(v^*)`

  By default, the Sinkhorn algorithm solves this dual problem in `f, g` or
  `u, v` using block coordinate ascent, i.e. devising an update for each `f`
  and `g` (resp. `u` and `v`) that cancels their respective gradients, one at
  a time. These two iterations are repeated ``inner_iterations`` times, after
  which the norm of these gradients will be evaluated and compared with the
  ``threshold`` value. The iterations are then repeated as long as that error
  exceeds ``threshold``.

Note on Sinkhorn updates:
  The boolean flag ``lse_mode`` sets whether the algorithm is run in either:

    - log-sum-exp mode (``lse_mode=True``), in which case it is directly defined in terms of updates to `f` and `g`, using log-sum-exp computations. This requires access to the cost matrix :math:`C`, as it is stored, or possibly computed on the fly by ``geom``.

    - kernel mode (``lse_mode=False``), in which case it will require access to a matrix vector multiplication operator :math:`z \rightarrow K z`, where :math:`K` is either instantiated from :math:`C` as :math:`\exp(-C/\epsilon)`, or provided directly. In that case, rather than optimizing on :math:`f` and :math:`g`, it is more convenient to optimize on their so called scaling formulations, :math:`u := \exp(f / \epsilon)` and :math:`v := \exp(g / \epsilon)`. While faster (applying matrices is faster than applying ``lse`` repeatedly over lines), this mode is also less stable numerically, notably for smaller :math:`\epsilon`.

  In the source code, the variables ``f_u`` or ``g_v`` can be either regarded as potentials (real) or scalings (positive) vectors, depending on the choice of ``lse_mode`` by the user. Once optimization is carried out, we only return dual variables in potential form, i.e. ``f`` and ``g``.

  In addition to standard Sinkhorn updates, the user can also use heavy-ball type updates using a ``momentum`` parameter in ]0,2[. We also implement a strategy that tries to set that parameter adaptively ater ``chg_momentum_from`` iterations, as a function of progress in the error, as proposed in the literature.

  Another upgrade to the standard Sinkhorn updates provided to the users lies in using Anderson acceleration. This can be parameterized by setting the otherwise null ``anderson`` to a positive integer. When selected,the algorithm will recompute, every ``refresh_anderson_frequency`` (set by default to 1) an extrapolation of the most recently computed ``anderson`` iterates. When using that option, notice that differentiation (if required) can only be carried out using implicit differentiation, and that all momentum related parameters are ignored.

  The ``parallel_dual_updates`` flag is set to ``False`` by default. In that setting, ``g_v`` is first updated using the latest values for ``f_u`` and ``g_v``, before proceeding to update ``f_u`` using that new value for ``g_v``. When the flag is set to ``True``, both ``f_u`` and ``g_v`` are updated simultaneously. Note that setting that choice to ``True`` requires using some form of averaging (e.g. ``momentum=0.5``). Without this, and on its own ``parallel_dual_updates`` won't work.

Differentiation:
  The optimal solutions ``f`` and ``g`` and the optimal objective (``reg_ot_cost``) outputted by the Sinkhorn algorithm can be differentiated w.r.t. relevant inputs ``geom``, ``a`` and ``b`` using, by default, implicit differentiation of the optimality conditions (``implicit_differentiation`` set to ``True``). This choice has two consequences.

    - The termination criterion used to stop Sinkhorn (cancellation of gradient of objective w.r.t. ``f_u`` and ``g_v``) is used to differentiate ``f`` and ``g``, given a change in the inputs. These changes are computed by solving a linear system. The arguments starting with ``implicit_solver_*`` allow to define the linear solver that is used, and to control for two types or regularization (we have observed that, depending on the architecture, linear solves may require higher ridge parameters to remain stable). The optimality conditions in Sinkhorn can be analyzed as satisfying a ``z=z'`` condition, which are then differentiated. It might be beneficial (e.g. as in https://arxiv.org/abs/2002.03229) to use a preconditionning function ``precondition_fun`` to differentiate instead ``h(z)=h(z')``.

    - The objective ``reg_ot_cost`` returned by Sinkhon uses the so-called enveloppe (or Danskin's) theorem. In that case, because it is assumed that the gradients of the dual variables ``f_u`` and ``g_v`` w.r.t. dual objective are zero (reflecting the fact that they are optimal), small variations in ``f_u`` and ``g_v`` due to changes in inputs (such as ``geom``, ``a`` and ``b``) are considered negligible. As a result, ``stop_gradient`` is applied on dual variables ``f_u`` and ``g_v`` when evaluating the ``reg_ot_cost`` objective. Note that this approach is `invalid` when computing higher order derivatives. In that case the ``use_danskin`` flag must be set to ``False``.

  An alternative yet more costly way to differentiate the outputs of the Sinkhorn iterations is to use unrolling, i.e. reverse mode differentiation of the Sinkhorn loop. This is possible because Sinkhorn iterations are wrapped in a custom fixed point iteration loop, defined in ``fixed_point_loop``, rather than a standard while loop. This is to ensure the end result of this fixed point loop can also be differentiated, if needed, using standard JAX operations. To ensure backprop differentiability, the ``fixed_point_loop.fixpoint_iter_backprop`` loop does checkpointing of state variables (here ``f_u`` and ``g_v``) every ``inner_iterations``, and backpropagates automatically, block by block, through blocks of ``inner_iterations`` at a time.

Note:
  * The Sinkhorn algorithm may not converge within the maximum number of iterations for possibly several reasons:

    1. the regularizer (defined as ``epsilon`` in the geometry ``geom`` object) is too small. Consider either switching to ``lse_mode=True`` (at the price of a slower execution), increasing ``epsilon``, or, alternatively, if you are unable or unwilling to increase  ``epsilon``, either increase ``max_iterations`` or ``threshold``.
    2. the probability weights ``a`` and ``b`` do not have the same total mass, while using a balanced (``tau_a=tau_b=1.0``) setup. Consider either normalizing ``a`` and ``b``, or set either ``tau_a`` and/or ``tau_b<1.0``.
    3. OOMs issues may arise when storing either cost or kernel matrices that are too large in ``geom``. In the case where, the ``geom`` geometry is a ``PointCloud``, some of these issues might be solved by setting the ``online`` flag to ``True``. This will trigger a recomputation on the fly of the cost/kernel matrix.

  * The weight vectors ``a`` and ``b`` can be passed on with coordinates that have zero weight. This is then handled by relying on simple arithmetic for ``inf`` values that will likely arise (due to :math:`log(0)` when ``lse_mode`` is ``True``, or divisions by zero when ``lse_mode`` is ``False``). Whenever that arithmetic is likely to produce ``NaN`` values (due to ``-inf * 0``, or ``-inf - -inf``) in the forward pass, we use ``jnp.where`` conditional statements to carry ``inf`` rather than ``NaN`` values. In the reverse mode differentiation, the inputs corresponding to these 0 weights (a location `x`, or a row in the corresponding cost/kernel matrix), and the weight itself will have ``NaN`` gradient values. This is reflects that these gradients are undefined, since these points were not considered in the optimization and have therefore no impact on the output.
"""
import collections
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
from ott.core import anderson as anderson_lib
from ott.core import fixed_point_loop
from ott.core import implicit_differentiation as implicit_lib
from ott.core import momentum as momentum_lib
from ott.core import problems
from ott.geometry import geometry


SinkhornOutput = collections.namedtuple(
    'SinkhornOutput', ['f', 'g', 'reg_ot_cost', 'errors', 'converged'])


@jax.tree_util.register_pytree_node_class
class Sinkhorn:
  """A pytree describing a Sinkhorn solver for linear reg-OT problem.

  Atributes:
    threshold: tolerance used to stop the Sinkhorn iterations. This is
     typically the deviation between a target marginal and the marginal of the
     current primal solution when either or both tau_a and tau_b are 1.0
     (balanced or semi-balanced problem), or the relative change between two
     successive solutions in the unbalanced case.
    norm_error: power used to define p-norm of error for marginal/target.
    inner_iterations: the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead.
    min_iterations: the minimum number of Sinkhorn iterations carried
     out before the error is computed and monitored.
    max_iterations: the maximum number of Sinkhorn iterations. If
      ``max_iterations`` is equal to ``min_iterations``, sinkhorn iterations are
      run by default using a ``jax.lax.scan`` loop rather than a custom,
      unroll-able ``jax.lax.while_loop`` that monitors convergence. In that case
      the error is not monitored and the ``converged`` flag will return
      ``False`` as a consequence.
    lse_mode: ``True`` for log-sum-exp computations, ``False`` for kernel
      multiplication.
    momentum: a Momentum instance. See ott.core.momentum
    anderson: an AndersonAcceleration instance. See ott.core.anderson.
    implicit_diff: instance used to solve implicit differentiation. Unrolls
      iterations if None.
    parallel_dual_updates: updates potentials or scalings in parallel if True,
      sequentially (in Gauss-Seidel fashion) if False.
    use_danskin: when ``True``, it is assumed the entropy regularized cost is
      is evaluated using optimal potentials that are freezed, i.e. whose
      gradients have been stopped. This is useful when carrying out first order
      differentiation, and is only valid (as with ``implicit_differentiation``)
      when the algorithm has converged with a low tolerance.
    jit: if True, automatically jits the function upon first call.
      Should be set to False when used in a function that is jitted by the user,
      or when computing gradients (in which case the gradient function
      should be jitted by the user)

  Returns:
    a ``SinkhornOutput``. The tuple contains two optimal potential
    vectors ``f`` and ``g``, the objective ``reg_ot_cost`` evaluated at those
    solutions, an array of ``errors`` to monitor convergence every
    ``inner_iterations`` and a flag ``converged`` that is ``True`` if the
    algorithm has converged within the number of iterations that was predefined
    by the user.
  """

  def __init__(self,
               lse_mode: bool = True,
               threshold: float = 1e-3,
               norm_error: int = 1,
               inner_iterations: int = 10,
               min_iterations: int = 0,
               max_iterations: int = 2000,
               momentum=None,
               anderson=None,
               parallel_dual_updates: bool = False,
               use_danskin: bool = None,
               implicit_diff=implicit_lib.ImplicitDiff(),
               jit: bool = True):
    self.lse_mode = lse_mode
    self.threshold = threshold
    self.inner_iterations = inner_iterations
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self._norm_error = norm_error
    self.momentum = (
        momentum if momentum is not None else momentum_lib.Momentum())
    self.anderson = anderson
    self.implicit_diff = implicit_diff
    self.parallel_dual_updates = parallel_dual_updates
    self.jit = jit

    # Force implicit_differentiation to True when using Anderson acceleration,
    # Reset all momentum parameters.
    if anderson:
      self.implicit_diff = (
          implicit_lib.ImplicitDiff() if self.implicit_diff is None
          else self.implicit_diff)
      self.momentum = momentum_lib.Momentum(start=0, value=1.0)

    # By default, use Danskin theorem to differentiate
    # the objective when using implicit_lib.
    self.use_danskin = ((self.implicit_diff is not None) if use_danskin is None
                        else use_danskin)

  @property
  def norm_error(self):
    # To change momentum adaptively, one needs errors in ||.||_1 norm.
    # In that case, we add this exponent to the list of errors to compute,
    # notably if that was not the error requested by the user.
    if self.momentum and self.momentum.start > 0 and self._norm_error != 1:
      return (self._norm_error, 1)
    else:
      return (self._norm_error,)

  def __call__(self,
               ot_prob: problems.LinearProblem,
               init_dual_a=None,
               init_dual_b=None) -> SinkhornOutput:
    """Main interface to run sinkhorn."""
    a, b = ot_prob.a, ot_prob.b
    if init_dual_a is None:
      init_dual_a = jnp.zeros_like(a) if self.lse_mode else jnp.ones_like(a)
    if init_dual_b is None:
      init_dual_b = jnp.zeros_like(b) if self.lse_mode else jnp.ones_like(b)
    # Cancel dual variables for zero weights.
    init_dual_a = jnp.where(
        a > 0, init_dual_a, -jnp.inf if self.lse_mode else 0.0)
    init_dual_b = jnp.where(
        b > 0, init_dual_b, -jnp.inf if self.lse_mode else 0.0)

    run_fn = run if not self.jit else jax.jit(run)
    return run_fn(ot_prob, self, init_dual_a, init_dual_b)

  def tree_flatten(self):
    aux = vars(self).copy()
    aux['norm_error'] = aux.pop('_norm_error')
    aux.pop('threshold')
    return [self.threshold], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data, threshold=children[0])

  def one_iteration(self, iteration, ot_prob, state, compute_error):
    """Carries out sinkhorn iteration.

    Depending on lse_mode, these iterations can be either in:
      - log-space for numerical stability.
      - scaling space, using standard kernel-vector multiply operations.

    Args:
      iteration: iteration number
      ot_prob: the transport problem definition
      state: error log, potential/scaling variables updated in the loop, and
        history for Anderson acceleration if selected.
      compute_error: flag to indicate this iteration computes/stores an error

    Returns:
      state variables, i.e. errors and updated f_u, g_v potentials + history.
    """
    errors, f_u, g_v, old_f_u_s, old_mapped_f_u_s = state
    # Compute momentum term if needed, using previously seen errors.
    w = jax.lax.stop_gradient(
        self.momentum.weight(iteration, errors, self.inner_iterations))
    # When running updates in parallel (Gauss-Seidel mode), old_g_v will be
    # used to update f_u, rather than the latest g_v computed in this loop.
    # Unused otherwise.
    old_g_v = g_v
    # When using Anderson acceleration, first update the dual variable f_u
    # with previous updates (if iteration count sufficiently large), then
    # record new iterations in array.
    if self.anderson:
      # TODO(oliviert): turn the state into an object.
      f_u, old_f_u_s = self.anderson.update(
          ot_prob, self.lse_mode, iteration, f_u, old_f_u_s, old_mapped_f_u_s)

    if self.lse_mode:  # In lse_mode, run additive updates.
      new_g_v = ot_prob.tau_b * ot_prob.geom.update_potential(
          f_u, g_v, jnp.log(ot_prob.b), iteration, axis=0)
      g_v = (1.0 - w) * jnp.where(jnp.isfinite(g_v), g_v, 0.0) + w * new_g_v
      new_f_u = ot_prob.tau_a * ot_prob.geom.update_potential(
          f_u,
          old_g_v if self.parallel_dual_updates else g_v,
          jnp.log(ot_prob.a),
          iteration,
          axis=1)
      f_u = (1.0 - w) * jnp.where(jnp.isfinite(f_u), f_u, 0.0) + w * new_f_u
    else:  # In kernel mode, run multiplicative updates.
      new_g_v = ot_prob.geom.update_scaling(
          f_u, ot_prob.b, iteration, axis=0) ** ot_prob.tau_b
      g_v = jnp.where(g_v > 0, g_v, 1)**(1.0 - w) * new_g_v ** w
      new_f_u = ot_prob.geom.update_scaling(
          old_g_v if self.parallel_dual_updates else g_v, ot_prob.a, iteration,
          axis=1) ** ot_prob.tau_a
      f_u = jnp.where(f_u > 0, f_u, 1) ** (1.0 - w) * new_f_u ** w

    # When using Anderson acceleration, refresh latest update.
    if self.anderson:
      f = f_u if self.lse_mode else ot_prob.geom.potential_from_scaling(f_u)
      old_mapped_f_u_s = jnp.concatenate(
          (old_mapped_f_u_s[:, 1:], f[:, None]), axis=1)

    # re-computes error if compute_error is True, else set it to inf.
    err = jnp.where(
        jnp.logical_and(compute_error, iteration >= self.min_iterations),
        ot_prob.marginal_error(f_u, g_v, self.norm_error, self.lse_mode),
        jnp.inf)
    errors = errors.at[iteration // self.inner_iterations, :].set(err)
    return errors, f_u, g_v, old_f_u_s, old_mapped_f_u_s

  def not_converged(self, iteration, errors):
    err = errors[iteration // self.inner_iterations - 1, 0]
    return jnp.logical_or(
        iteration == 0,
        jnp.logical_and(jnp.isfinite(err), err > self.threshold))


def run(ot_prob, solver, init_dual_a, init_dual_b) -> SinkhornOutput:
  """A jittable sinkhorn.

  Note:
  When differentiating the regularized OT cost, and assuming Sinkhorn has
  run to convergence, Danskin's (or the enveloppe) theorem
  https://en.wikipedia.org/wiki/Danskin%27s_theorem
  states that the resulting OT cost as a function of any of the inputs
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
    solver: Sinkhorn parameters.
    init_dual_a: the initial value for ``f``.
    init_dual_b: the initial value for ``g``.

  Returns:
    A SinkhornOutput.
  """
  iter_fun = _iterations_implicit if solver.implicit_diff else iterations
  f, g, errors = iter_fun(ot_prob, solver, init_dual_a, init_dual_b)
  reg_ot_cost = ot_prob.ent_reg_cost(
      jax.lax.stop_gradient(f) if solver.use_danskin else f,
      jax.lax.stop_gradient(g) if solver.use_danskin else g,
      solver.lse_mode)
  converged = jnp.logical_and(
      jnp.sum(errors == -1) > 0,
      jnp.sum(jnp.isnan(errors)) == 0)
  return SinkhornOutput(f, g, reg_ot_cost, errors, converged)


def iterations(ot_prob, solver, init_dual_a, init_dual_b
               ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """A jit'able Sinkhorn loop."""

  def cond_fn(iteration, const, state):
    _, solver = const
    errors = state[0]
    return solver.not_converged(iteration, errors)

  def body_fn(iteration, const, state, compute_error):
    ot_prob, solver = const
    return solver.one_iteration(iteration, ot_prob, state, compute_error)

  # Run the Sinkhorn loop. Choose either a standard fixpoint_iter loop if
  # differentiation is implicit, otherwise switch to the backprop friendly
  # version of that loop if unrolling to differentiate.
  if solver.implicit_diff:
    fix_point = fixed_point_loop.fixpoint_iter
  else:
    fix_point = fixed_point_loop.fixpoint_iter_backprop

  const = ot_prob, solver

  f_u, g_v = init_dual_a, init_dual_b  # Initializing solutions
  outer_iterations = np.ceil(solver.max_iterations / solver.inner_iterations)
  errors = -jnp.ones((outer_iterations.astype(int), len(solver.norm_error)))
  # Initialize log matrix used in Anderson acceleration with nan values.
  # these values will be replaced by actual iteration values.
  old_f_u_s, old_mapped_f_u_s = (
      solver.anderson.init_maps(ot_prob) if solver.anderson
      else (None, None))
  state_0 = (errors, f_u, g_v, old_f_u_s, old_mapped_f_u_s)
  errors, f_u, g_v, _, _ = fix_point(
      cond_fn, body_fn,
      solver.min_iterations, solver.max_iterations, solver.inner_iterations,
      const, state_0)

  f = f_u if solver.lse_mode else ot_prob.geom.potential_from_scaling(f_u)
  g = g_v if solver.lse_mode else ot_prob.geom.potential_from_scaling(g_v)
  return f, g, errors[:, 0]


def _iterations_taped(ot_prob: problems.LinearProblem,
                      solver: Sinkhorn,
                      init_dual_a: jnp.ndarray,
                      init_dual_b: jnp.ndarray):
  """Runs forward pass of the Sinkhorn algorithm storing side information."""
  f, g, errors = iterations(ot_prob, solver, init_dual_a, init_dual_b)
  return (f, g, errors), (f, g, ot_prob, solver)


def _iterations_implicit_bwd(res, gr):
  """Runs Sinkhorn in backward mode, using implicit differentiation.

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
  return (
      *solver.implicit_diff.gradient(ot_prob, f, g, solver.lse_mode, gr),
      None, None, None)


# Sets threshold, norm_errors, geom, a and b to be differentiable, as those are
# non static. Only differentiability w.r.t. geom, a and b will be used.
_iterations_implicit = jax.custom_vjp(iterations)
_iterations_implicit.defvjp(_iterations_taped, _iterations_implicit_bwd)


def sinkhorn(
    geom: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    threshold: float = 1e-3,
    norm_error: int = 1,
    inner_iterations: int = 10,
    min_iterations: int = 0,
    max_iterations: int = 2000,
    momentum: float = 1.0,
    chg_momentum_from: int = 0,
    anderson_acceleration: int = 0,
    refresh_anderson_frequency: int = 1,
    lse_mode: bool = True,
    implicit_differentiation: bool = True,
    implicit_solver_fun=jax.scipy.sparse.linalg.cg,
    implicit_solver_ridge_kernel: float = 0.0,
    implicit_solver_ridge_identity: float = 0.0,
    implicit_solver_symmetric: bool = False,
    precondition_fun: Optional[Callable[[float], float]] = None,
    parallel_dual_updates: bool = False,
    use_danskin: bool = None,
    init_dual_a: Optional[jnp.ndarray] = None,
    init_dual_b: Optional[jnp.ndarray] = None,
    jit: bool = False) -> SinkhornOutput:
  """For backward compatibility."""
  if not implicit_differentiation:
    implicit_diff = None
  else:
    implicit_diff = implicit_lib.ImplicitDiff(
        solver_fun=implicit_solver_fun,
        ridge_kernel=implicit_solver_ridge_kernel,
        ridge_identity=implicit_solver_ridge_identity,
        symmetric=implicit_solver_symmetric,
        precondition_fun=precondition_fun)

  if anderson_acceleration > 0:
    anderson = anderson_lib.AndersonAcceleration(
        memory=anderson_acceleration,
        refresh_every=refresh_anderson_frequency)
  else:
    anderson = None

  sink = Sinkhorn(
      lse_mode=lse_mode,
      threshold=threshold,
      norm_error=norm_error,
      inner_iterations=inner_iterations,
      min_iterations=min_iterations,
      max_iterations=max_iterations,
      momentum=momentum_lib.Momentum(
          start=chg_momentum_from, value=momentum),
      anderson=anderson,
      implicit_diff=implicit_diff,
      parallel_dual_updates=parallel_dual_updates,
      use_danskin=use_danskin,
      jit=jit)
  ot_prob = problems.LinearProblem(geom, a, b, tau_a, tau_b)
  return sink(ot_prob, init_dual_a, init_dual_b)
