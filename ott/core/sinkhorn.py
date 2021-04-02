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
"""A Jax version of Sinkhorn's algorithm."""

import collections
import functools
import numbers
from typing import Any
from typing import Optional, Sequence, Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ott.core import fixed_point_loop
from ott.geometry import geometry


SinkhornOutput = collections.namedtuple(
    'SinkhornOutput', ['f', 'g', 'reg_ot_cost', 'errors', 'converged'])


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
    momentum_strategy: Optional[Union[float, str]] = None,
    lse_mode: bool = True,
    implicit_differentiation: bool = True,
    jit: bool = False,
) -> SinkhornOutput:
  r"""Solves regularized OT problems using Sinkhorn iterations.

  The Sinkhorn algorithm is a fixed point algorithm that seeks a pair of
  variables that optimize a regularized optimal transport (reg-OT) problem. This
  function outputs this pair of optimal solutions, in addition to the objective
  that is reached, a vector of errors computed during iterations and a flag.

  The reg-OT problem is specified by two measures, of respective sizes n and m.
  From the viewpoint of `sinkhorn` function, these two measures are only seen
  through a geometry object `geom` (a cost or kernel structure between their
  respective points) and weight vectors `a` and `b`. The Sinkhorn algorithm
  direct the `geom` object to carry out the heaviest computations.

  Given a geometry, which provides either a cost matrix C with its
  regularization parameter :math:`\epsilon`, (resp. a cost matrix K) the reg-OT
  problem solves for two vectors f, g of size n, m

  :math:`arg\max_{f, g}{- <a, \phi_a^{*}(-f)> + <b, \phi_b^{*}(-g)> - \epsilon
  <e^{f/\epsilon}, e^{-C/\epsilon} e^{-g/\epsilon}}>`

  (respectively, written the space of positive scaling vectors u, v of size n, m
  :math:`arg\max_{u, v} - <a,\phi_a*(-\log u)> + <b, \phi_b*(-\log v)> -  <u, K
  v>` )

  where :math:`\phi_a(z) = \rho_a z(\log z - 1)` is a scaled entropy. This
  problem corresponds, in a so-called primal representation, to solving the
  unbalanced optimal transport problem with a variable matrix P of size n x m:

  :math:`\arg\min_{P} <P,C> -\epsilon H(P) + \rho_a KL(P1 | a) + \rho_b KL(P^T1
  | b)`

  (resp. :math:`\arg\min_{P} KL(P|K) + \rho_a KL(P1 | a) + \rho_b KL(P^T1 | b)`
  )

  The *balanced* regularized OT problem is recovered when :math:`\rho_a, \rho_b
  \rightarrow \infty`.

  The *original* (not regularized) OT problem is recovered
  when :math:`\epsilon \rightarrow 0` using the cost formulation. This problem
  is not handled for now in this toolbox, which focuses exclusively
  on :math:`\epsilon > 0`.

  To allow for the option :math:`\rho_a, \rho_b \rightarrow \infty`,
  the sinkhorn function uses parameters
  tau_a := :math:`\rho_a / (\epsilon + \rho_a)` and tau_b := :math:`\rho_b /
  (\epsilon + \rho_b)`
  instead. Setting these parameters to 1 corresponds to setting ⍴ to ∞ above.

  The Sinkhorn algorithm solves the reg-OT problem by seeking optimal f, g
  potentials (or alternatively their parameterization as positive scalings u, v)
  rather than solving it directly for a matrix P. This is mostly for efficiency
  (potentials and scalings have a n + m memory footprint, rather than n x m
  required to store P) and also because both problems are in fact equivalent,
  since the optimal transport :math:`P^*` can be recovered from optimal
  potentials :math:`f^*`, :math:`g^*` or scalings :math:`u^*`, :math:`v^*`,
  using the geometry's cost or kernel matrices respectively:

    :math:`P^* = \text{jnp.exp}(( f^* + g^* - C )/\epsilon) \text{ or } P^* =
    \text{diag}(u^*) K \text{diag}(v^*)`

  The Sinkhorn algorithm solves this dual problem in f,g or u,v using block
  coordinate ascent, i.e. devising an update for each f and g (resp. u and v)
  that cancels their respective gradients, one at a time. These two iterations
  are repeated `inner_iterations` times, after which the norm of these gradients
  will be evaluated and compared with the `threshold` value. The iterations are
  then repeated as long as that errors does not go below `threshold`.

  The boolean flag `lse_mode` sets whether the algorithm is run in either:

    - log-sum-exp mode (`lse_mode=True`), in which case it is directly defined
  in terms of updates to f and g, using log-sum-exp computations. This requires
  access to the cost matrix C, as stored or computed on the fly by the geometry.

    - kernel mode (`lse_mode=False`), in which case it will require access to a
  matrix vector multiplication operator z → K z, where K is either instantiated
  from C as :math:`\exp(-C/\epsilon)`, or provided directly. In that case,
  rather than optimizing on f and g directly, it is more convenient to optimize
  on their so called scaling formulations, :math:`u := \exp(f / \epsilon)`
  & :math:`v := \exp(g / \epsilon)`. While faster (applying matrices is faster
  than applying lse repeatedly over lines), this mode is also less stable
  numerically, notably for smaller :math:`\epsilon`.

  In the code below, the variables f_u or g_v can be either regarded as
  potentials (real) or scalings (positive) vectors, depending on the choice
  of lse_mode by the end user.

  In addition to standard Sinkhorn updates, the user can also change them with
  a `momentum_strategy` parameter in ]0,2[. We also implement a strategy that
  tries to set that parameter adaptively, as a function of progress in the
  error, as proposed in the literature.

  Differentiation through the Sinkhorn algorithm is carried out by default
  using implicit differentiation of the optimality conditions, as reflected by
  the default setting of `implicit_differentiation` to`True`. In that case the
  termination criterion used to stop Sinkhorn (cancellation of gradient of
  objective w.r.t. `f_u` and `g_v`) is used to differentiate inputs given a
  desired change in the outputs.

  Alternatively, the Sinkhorn iterations have been wrapped in a fixed point
  iteration loop, defined in `fixed_point_loop`, rather than a standard while
  loop. This is to ensure the end result of this fixed point loop can also be
  differentiated, if needed, using standard JAX operations. To ensure
  backprop differentiability, the `fixed_point_loop.fixpoint_iter_backprop` loop
  does checkpointing of state variables (here `f_u` and `g_v`) every
  `inner_iterations`, and backpropagates automatically, block by block,
  through blocks of `inner_iterations` at a time.

  Note:
    * The Sinkhorn algorithm may not converge within the maximum number of
    iterations for possibly several reasons:
      1. the regularizer (defined as `epsilon` in the geometry `geom` object) is
      too small. Consider switching to `lse_mode = True` (at the price of a
      slower execution), increasing `epsilon`, or, alternatively, if you are
      sure that value `epsilon` is correct, or your cannot modify it, either
      increase `max_iterations` or `threshold`.
      2. the probability weights `a` and `b` do not have the same total mass,
      while using a balanced (`tau_a = tau_b = 1.0`) setup. Consider either
      normalizing `a` and `b`, or set either `tau_a` and/or `tau_b < 1.0`.
      3. OOMs issues may arise when storing either cost or kernel matrices that
      are too large in `geom`. In that case, in the case where, the `geom`
      geometry is a `PointCloud`, set the `online` flag to `True`.

    * The weight vectors `a` and `b` are assumed to be positive by default, but
    zero weights are currently handled by relying on simple arithmetic for inf
    values that will likely arise (starting with log(0) when `lse_mode` is
    `True`, or divisions by zero when `lse_mode` is `False`). Whenever that
    arithmetic is likely to produce `NaN`s (`-inf * 0`, or `-inf - -inf`) in the
    forward pass, we use jnp.where conditional statements. In the backward pass,
    the inputs corresponding to these 0 weights (typically a location `x`
    associated with that weight), and the weight itself will have `NaN` gradient
    values.

  Args:
    geom: a Geometry object.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
     marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to first
     marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    threshold: (float) tolerance used to stop the Sinkhorn iterations. This is
     typically the deviation between a target marginal and the marginal of the
     current primal solution when either or both tau_a and tau_b are 1.0
     (balanced or semi-balanced problem), or the relative change between two
     successive solutions in the unbalanced case.
    norm_error: int, power used to define p-norm of error for marginal/target.
    inner_iterations: (int32) the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead.
    min_iterations: (int32) the minimum number of Sinkhorn iterations carried
     out before the error is computed and monitored.
    max_iterations: (int32) the maximum number of Sinkhorn iterations.
    momentum_strategy: either a float between ]0,2[ or a string.
    lse_mode: True for log-sum-exp computations, False for kernel
      multiplication.
    implicit_differentiation: True if using implicit diff, False if backprop.
    jit: bool, if True, jits the function.
      Should be set to False when used in a function that is jitted by the user,
      or when computing gradients (in which case the gradient function
      should be jitted by the user)

  Returns:
    a SinkhornOutput named tuple.

  Raises:
    ValueError: If momentum parameter is not set correctly, or to a wrong value.
  """
  if jit:
    call_to_sinkhorn = functools.partial(
        jax.jit, static_argnums=(3, 4, 6, 7, 8, 9, 10, 11, 12))(
            _sinkhorn)
  else:
    call_to_sinkhorn = _sinkhorn
  return call_to_sinkhorn(geom, a, b, tau_a, tau_b, threshold, norm_error,
                          inner_iterations, min_iterations, max_iterations,
                          momentum_strategy, lse_mode, implicit_differentiation)


def _sinkhorn(
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
    momentum_strategy: Optional[Union[float, str]] = None,
    lse_mode: bool = True,
    implicit_differentiation: bool = True) -> SinkhornOutput:
  """Checks inputs and forks between implicit/backprop exec of Sinkhorn."""

  num_a, num_b = geom.shape
  a = jnp.ones((num_a,)) / num_a if a is None else a
  b = jnp.ones((num_b,)) / num_b if b is None else b

  if momentum_strategy is None:
    momentum_strategy = 1.0

  if (isinstance(momentum_strategy, str) and
      momentum_strategy.lower() == 'lehmann'):
    # check the unbalanced formulation is not selected.
    if tau_a != 1 and tau_b != 1:
      raise ValueError('The Lehmann momentum strategy cannot be selected for '
                       'unbalanced transport problems (namely when either '
                       'tau_a or tau_b < 1).')
    # The Lehmann strategy needs to keep track of errors in ||.||_1 norm.
    # In that case, we add this exponent to the list of errors to compute,
    # if that was not the error requested by the user.
    norm_error = (norm_error,) if norm_error == 1 else (norm_error, 1)
    momentum_default = 1.0
    chg_momentum_from = np.maximum(
        (min_iterations + 100) // inner_iterations, 2)
  elif isinstance(momentum_strategy, numbers.Number):
    if not 0 < momentum_strategy < 2:
      raise ValueError('Momentum parameter must be strictly between 0 and 2.')
    momentum_default, chg_momentum_from = momentum_strategy, max_iterations + 1
    norm_error = (norm_error,)
  else:
    raise ValueError('Momentum parameter must be either a float in ]0,2[ (when'
                     ' set to 1 one recovers the usual Sinkhorn updates) or '
                     'a valid string.')
  if implicit_differentiation:
    iteration_fun = _sinkhorn_iterations_implicit
  else:
    iteration_fun = _sinkhorn_iterations
  f, g, errors = iteration_fun(tau_a, tau_b, inner_iterations, min_iterations,
                               max_iterations, momentum_default,
                               chg_momentum_from, lse_mode,
                               implicit_differentiation, threshold, norm_error,
                               geom, a, b)

  # When differentiating the regularized OT cost, it is not necessary to compute
  # the Jacobian of the optimal dual potentials f, g w.r.t. inputs `geom`, `a`
  # and `b` because that value is precisely what is minimized by those optimal
  # dual potentials. In that case the gradients of reg_ot_cost w.r.t. `geom`,
  # `a` and `b` can be directly computed from dual potentials, using the
  # enveloppe theorem. This is impacted below by cutting gradients at the level
  # of f, g.
  #
  # Notice this is only valid, much like the implicit_differentiation mode, if
  # the Sinkhorn algorithm outputs potentials that are approximately optimal,
  # namely when the threshold value is set to a small tolerance.
  #
  # TODO(cuturi) raise error message when tolerance high, or switch to backprop
  # when threshold is too large, and therefore do not stop_gradients below.

  reg_ot_cost = ent_reg_cost(geom, a, b, tau_a, tau_b,
                             jax.lax.stop_gradient(f), jax.lax.stop_gradient(g))
  converged = jnp.logical_and(
      jnp.sum(errors == -1) > 0,
      jnp.sum(jnp.isnan(errors)) == 0)
  return SinkhornOutput(f, g, reg_ot_cost, errors, converged)


def _sinkhorn_iterations(
    tau_a: float,
    tau_b: float,
    inner_iterations: int,
    min_iterations: int,
    max_iterations: int,
    momentum_default: float,
    chg_momentum_from: int,
    lse_mode: bool,
    implicit_differentiation: bool,
    threshold: float,
    norm_error: Sequence[int],
    geom: geometry.Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """The jittable Sinkhorn loop, that uses a custom backward or not.

  Args:
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    inner_iterations: (int32) the Sinkhorn error is not recomputed at each
      iteration but every inner_num_iter instead.
    min_iterations: (int32) the minimum number of Sinkhorn iterations.
    max_iterations: (int32) the maximum number of Sinkhorn iterations.
    momentum_default: float, a float between ]0,2[
    chg_momentum_from: int, # of iterations after which momentum is computed
    lse_mode: True for log-sum-exp computations, False for kernel
      multiplication.
    implicit_differentiation: if True, do not backprop through the Sinkhorn
      loop, but use the implicit function theorem on the fixed point optimality
      conditions.
    threshold: (float) the relative threshold on the Sinkhorn error to stop the
      Sinkhorn iterations.
    norm_error: t-uple of int, p-norms of marginal / target errors to track
    geom: a Geometry object.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.

  Returns:
    f: potential
    g: potential
    errors: ndarray of errors
  """

  # Defining the Sinkhorn loop, by setting initializations, body/cond.
  num_a, num_b = geom.shape
  if lse_mode:
    f_u, g_v = jnp.zeros_like(a), jnp.zeros_like(b)
  else:
    f_u, g_v = jnp.ones_like(a) / num_a, jnp.ones_like(b) / num_b

  errors = -jnp.ones((np.ceil(max_iterations / inner_iterations).astype(int),
                      len(norm_error)))
  const = (geom, a, b, threshold)

  def cond_fn(iteration, const, state):
    threshold = const[-1]
    errors = state[0]
    err = errors[iteration // inner_iterations-1, 0]

    return jnp.logical_or(iteration == 0,
                          jnp.logical_and(jnp.isfinite(err), err > threshold))

  def get_momentum(errors, idx):
    """momentum formula, https://arxiv.org/pdf/2012.12562v1.pdf, p.7 and (5)."""
    error_ratio = jnp.minimum(errors[idx - 1, -1] / errors[idx - 2, -1], .99)
    power = 1.0 / inner_iterations
    return 2.0 / (1.0 + jnp.sqrt(1.0 - error_ratio ** power))

  def body_fn(iteration, const, state, compute_error):
    """Carries out sinkhorn iteration.

    Depending on lse_mode, these iterations can be either in:
      - log-space for numerical stability.
      - scaling space, using standard kernel-vector multiply operations.

    Args:
      iteration: iteration number
      const: tuple of constant parameters that do not change throughout the
        loop, here the geometry and the marginals a, b.
      state: potential/scaling variables updated in the loop & error log.
      compute_error: flag to indicate this iteration computes/stores an error

    Returns:
      state variables, i.e. errors and updated f_u, g_v potentials.
    """
    geom, a, b, _ = const
    errors, f_u, g_v = state

    # compute momentum term if needed, using previously seen errors.
    w = jax.lax.stop_gradient(jnp.where(iteration >= (
        inner_iterations * chg_momentum_from + min_iterations),
                                        get_momentum(errors, chg_momentum_from),
                                        momentum_default))

    # Sinkhorn updates using momentum, in either scaling or potential form.
    if lse_mode:
      new_g_v = tau_b * geom.update_potential(f_u, g_v, jnp.log(b),
                                              iteration, axis=0)
      g_v = (1.0 - w) * jnp.where(jnp.isfinite(g_v), g_v, 0.0) + w * new_g_v

      new_f_u = tau_a * geom.update_potential(f_u, g_v, jnp.log(a),
                                              iteration, axis=1)
      f_u = (1.0 - w) * jnp.where(jnp.isfinite(f_u), f_u, 0.0) + w * new_f_u
    else:
      new_g_v = geom.update_scaling(f_u, b, iteration, axis=0) ** tau_b
      g_v = jnp.where(g_v > 0, g_v, 1) ** (1.0 - w) * new_g_v ** w

      new_f_u = geom.update_scaling(g_v, a, iteration, axis=1) ** tau_a
      f_u = jnp.where(f_u > 0, f_u, 1) ** (1.0 - w) * new_f_u ** w

    # re-computes error if compute_error is True, else set it to inf.
    err = jnp.where(
        jnp.logical_and(compute_error, iteration >= min_iterations),
        marginal_error(geom, a, b, tau_a, tau_b, f_u, g_v, norm_error,
                       lse_mode),
        jnp.inf)

    errors = jax.ops.index_update(
        errors, jax.ops.index[iteration // inner_iterations, :], err)
    return errors, f_u, g_v

  # Run the Sinkhorn loop. choose either a standard fixpoint_iter loop if
  # differentiation is implicit, otherwise switch to the backprop friendly
  # version of that loop if using backprop to differentiate.

  if implicit_differentiation:
    fix_point = fixed_point_loop.fixpoint_iter
  else:
    fix_point = fixed_point_loop.fixpoint_iter_backprop

  errors, f_u, g_v = fix_point(
      cond_fn, body_fn, min_iterations, max_iterations, inner_iterations, const,
      (errors, f_u, g_v))

  f = f_u if lse_mode else geom.potential_from_scaling(f_u)
  g = g_v if lse_mode else geom.potential_from_scaling(g_v)

  return f, g, errors[:, 0]


def _sinkhorn_iterations_taped(
    tau_a: float,
    tau_b: float,
    inner_iterations: int,
    min_iterations: int,
    max_iterations: int,
    momentum_default: float,
    chg_momentum_from: int,
    lse_mode: bool,
    implicit_differentiation: bool,
    threshold: float,
    norm_error: Sequence[int],
    geom: geometry.Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray):
  """Runs forward pass of the Sinkhorn algorithm storing side information."""
  f, g, errors = _sinkhorn_iterations(tau_a, tau_b, inner_iterations,
                                      min_iterations, max_iterations,
                                      momentum_default, chg_momentum_from,
                                      lse_mode, implicit_differentiation,
                                      threshold, norm_error, geom, a, b)
  return (f, g, errors), (f, g, geom, a, b)


def _sinkhorn_iterations_implicit_bwd(
    tau_a, tau_b, inner_iterations, min_iterations, max_iterations,
    momentum_default, chg_momentum_from, lse_mode, implicit_differentiation,
    res, gr) -> Tuple[Any, Any, geometry.Geometry, jnp.ndarray, jnp.ndarray]:
  """Runs Sinkhorn in backward mode, using implicit differentiation.

  Args:
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    inner_iterations: (int32) the Sinkhorn error is not recomputed at each
      iteration but every inner_num_iter instead.
    min_iterations: (int32) the minimum number of Sinkhorn iterations.
    max_iterations: (int32) the maximum number of Sinkhorn iterations.
    momentum_default: float, a float between ]0,2[
    chg_momentum_from: int, # of iterations after which momentum is computed
    lse_mode: True for log-sum-exp computations, False for kernel
      multiplication.
    implicit_differentiation: if True, do not backprop through the Sinkhorn
      loop, but use the implicit function theorem on the fixed point optimality
      conditions.
    res: residual data sent from fwd pass, used for computations below. In this
      case consists in the output itself, as well as inputs against which we
      wish to differentiate.
    gr: gradients w.r.t outputs of fwd pass, here w.r.t size f, g, errors. Note
      that differentiability w.r.t. errors is not handled, and only f, g is
      considered.

  Returns:
    a tuple of gradients: PyTree for geom, one jnp.ndarray for each of a and b.
  """
  del inner_iterations, min_iterations, max_iterations, momentum_default
  del chg_momentum_from, implicit_differentiation
  f, g, geom, a, b = res
  # Ignores gradients info with respect to 'errors' output.
  gr = gr[0], gr[1]


  # Applies first part of vjp to gr: inverse part of implicit function theorem.
  vjp_gr = apply_inv_hessian(gr, geom, a, b, f, g, tau_a, tau_b, lse_mode)
  # Instantiates vjp of first order conditions of the objective, as a
  # function of geom, a, b parameters (against which we seek to differentiate)
  foc_geom_a_b = lambda geom, a, b: first_order_conditions(
      geom, a, b, f, g, tau_a, tau_b, lse_mode)
  # Carries pullback onto original inputs, here geom, a and b.
  _, pull_geom_a_b = jax.vjp(foc_geom_a_b, geom, a, b)
  g_geom, g_a, g_b = pull_geom_a_b(vjp_gr)

  # First gradients are for threshold and norm_errors: we set them to None
  return None, None, g_geom, g_a, g_b


# We set threshold, norm_errors, geom, a and b to be differentiable
# as those are non static.
_sinkhorn_iterations_implicit = functools.partial(
    jax.custom_vjp, nondiff_argnums=range(9))(_sinkhorn_iterations)
_sinkhorn_iterations_implicit.defvjp(_sinkhorn_iterations_taped,
                                     _sinkhorn_iterations_implicit_bwd)


def marginal_error(geom: geometry.Geometry, a: jnp.ndarray, b: jnp.ndarray,
                   tau_a: float, tau_b: float, f_u: jnp.ndarray,
                   g_v: jnp.ndarray, norm_error: int, lse_mode) -> jnp.ndarray:
  """Conputes marginal error, the stopping criterion used to terminate Sinkhorn.

  Args:
    geom: a Geometry object.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    f_u: jnp.ndarray, potential or scaling
    g_v: jnp.ndarray, potential or scaling
    norm_error: int, p-norm used to compute error.
    lse_mode: True if log-sum-exp operations, False if kernel vector producs.

  Returns:
    a positive number quantifying how far from convergence the algorithm stands.

  """
  if tau_b == 1.0 and tau_b == 1.0:
    err = geom.error(f_u, g_v, b, 0, norm_error, lse_mode)
  else:
    # In the unbalanced case, we compute the norm of the gradient.
    # the gradient is equal to the marginal of the current plan minus
    # the gradient of < z, rho_z(exp^(-h/rho_z) -1> where z is either a or b
    # and h is either f or g. Note this is equal to z if rho_z → inf, which
    # is the case when tau_z → 1.0
    if lse_mode:
      grad_a = grad_of_marginal_fit(a, f_u, tau_a, geom.epsilon)
      grad_b = grad_of_marginal_fit(b, g_v, tau_b, geom.epsilon)
    else:
      grad_a = grad_of_marginal_fit(a, geom.potential_from_scaling(f_u),
                                    tau_a, geom.epsilon)
      grad_b = grad_of_marginal_fit(b, geom.potential_from_scaling(g_v),
                                    tau_b, geom.epsilon)
    err = geom.error(f_u, g_v, grad_a, 1, norm_error, lse_mode)
    err += geom.error(f_u, g_v, grad_b, 0, norm_error, lse_mode)
  return err


def ent_reg_cost(geom: geometry.Geometry,
                 a: jnp.ndarray,
                 b: jnp.ndarray,
                 tau_a: float,
                 tau_b: float,
                 f: jnp.ndarray,
                 g: jnp.ndarray) -> jnp.ndarray:
  """Computes objective of regularized OT given dual solutions f,g.

  In all sums below, jnp.where handle situations in which some coordinates of
  a and b are zero. For those coordinates, their potential is -inf.
  This leads to -inf - -inf or -inf x 0 operations which result in NaN.
  These contributions are discarded when computing the objective.

  Args:
    geom: a Geometry object.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to first
      marginal and itself + epsilon regularizer used in the unbalanced
      formulation.
    f: jnp.ndarray, potential
    g: jnp.ndarray, potential

  Returns:
    a float, the regularized transport cost.
  """

  if tau_a == 1.0:
    div_a = jnp.sum(
        jnp.where(a > 0, (f - geom.potential_from_scaling(a)) * a, 0.0))
  else:
    rho_a = geom.epsilon * (tau_a / (1 - tau_a))
    div_a = - jnp.sum(jnp.where(
        a > 0,
        a * phi_star(-(f - geom.potential_from_scaling(a)), rho_a),
        0.0))

  if tau_b == 1.0:
    div_b = jnp.sum(
        jnp.where(b > 0, (g - geom.potential_from_scaling(b)) * b, 0.0))
  else:
    rho_b = geom.epsilon * (tau_b / (1 - tau_b))
    div_b = - jnp.sum(jnp.where(
        b > 0,
        b * phi_star(-(g - geom.potential_from_scaling(b)), rho_b),
        0.0))

  # Using https://arxiv.org/pdf/1910.12958.pdf (24)
  # The total mass of the coupling is computed in scaling space. This avoids
  # differentiation issues linked with the automatic differention of
  # jnp.exp(jnp.logsumexp(...)) when some of those logs appear as -inf.
  # Because we are computing total mass it is irrelevant to have underflow since
  # this would simply result in near 0 contributions, which, unlike Sinkhorn
  # iterations, do not appear next in a numerator.
  total_sum = jnp.sum(geom.marginal_from_scalings(
      geom.scaling_from_potential(f), geom.scaling_from_potential(g)))
  return div_a + div_b + geom.epsilon * (jnp.sum(a) * jnp.sum(b) - total_sum)


def grad_of_marginal_fit(c, h, tau, epsilon):
  """Computes grad of terms linked to marginals in objective.

  Computes gradient w.r.t. f ( or g) of terms in
  https://arxiv.org/pdf/1910.12958.pdf, left-hand-side of Eq. 15
  (terms involving phi_star)

  Args:
    c: jnp.ndarray, first target marginal (either a or b in practice)
    h: jnp.ndarray, potential (either f or g in practice)
    tau: float, strength (in ]0,1]) of regularizer w.r.t. marginal
    epsilon: regularization
  Returns:
    a vector of the same size as c or h
  """
  if tau == 1.0:
    return c
  else:
    rho = epsilon * tau / (1 - tau)
    return jnp.where(c > 0, c * derivative_phi_star(-h, rho), 0.0)


def phi_star(h: jnp.ndarray, rho: float) -> jnp.ndarray:
  """Legendre transform of KL, https://arxiv.org/pdf/1910.12958.pdf p.9."""
  return rho * (jnp.exp(h / rho) - 1)


def derivative_phi_star(f: jnp.ndarray, rho: float) -> jnp.ndarray:
  """Derivative of Legendre transform of KL, see phi_star."""
  return jnp.exp(f / rho)


def second_derivative_phi_star(f: jnp.ndarray, rho: float) -> jnp.ndarray:
  """Second Derivative of Legendre transform of KL, see phi_star."""
  return jnp.exp(f / rho) / rho


def diag_jacobian_of_marginal_fit(c, h, tau, epsilon):
  """Computes grad of terms linked to marginals in objective.

  Computes second derivative w.r.t. f ( or g) of terms in
  https://arxiv.org/pdf/1910.12958.pdf, left-hand-side of Eq. 15
  (terms involving phi_star)

  Args:
    c: jnp.ndarray, first target marginal (either a or b in practice)
    h: jnp.ndarray, potential (either f or g in practice)
    tau: float, strength (in ]0,1]) of regularizer w.r.t. marginal
    epsilon: regularization
  Returns:
    a vector of the same size as c or h
  """
  if tau == 1.0:
    return 0
  else:
    rho = epsilon * tau / (1 - tau)
    # here no minus sign because we are taking derivative w.r.t -h
    return jnp.where(c > 0, c * second_derivative_phi_star(-h, rho), 0.0)


def get_transport_functions(geom, lse_mode):
  """Instantiates useful functions from geometry depending on lse_mode."""
  if lse_mode:
    marginal_a = lambda f, g: geom.marginal_from_potentials(f, g, 1)
    marginal_b = lambda f, g: geom.marginal_from_potentials(f, g, 0)
    app_transport = geom.apply_transport_from_potentials
  else:
    marginal_a = lambda f, g: geom.marginal_from_scalings(
        geom.scaling_from_potential(f), geom.scaling_from_potential(g), 1)
    marginal_b = lambda f, g: geom.marginal_from_scalings(
        geom.scaling_from_potential(f), geom.scaling_from_potential(g), 0)
    app_transport = lambda f, g, z, axis: geom.apply_transport_from_scalings(
        geom.scaling_from_potential(f), geom.scaling_from_potential(g), z, axis)
  return marginal_a, marginal_b, app_transport


def apply_inv_hessian(gr: Tuple[np.ndarray],
                      geom: geometry.Geometry,
                      a: np.ndarray,
                      b: np.ndarray,
                      f: np.ndarray,
                      g: np.ndarray,
                      tau_a: float,
                      tau_b: float,
                      lse_mode: bool,
                      ridge=1e-6):
  """Applies - inverse of (hessian of reg_ot_cost w.r.t potentials (f,g)).

  If the Hessian were to be instantiated as a matrix, it would be symmetric
  and of size (n+m) x (n+m), written [A, B; C, D].

  The implicit function theorem requires solving a linear system w.r.t that
  Hessian. A and D are diagonal matrices, equal to the row and column marginals
  respectively, corrected (if handling the unbalanced case) by the second
  derivative of the part of the objective that ties potentials to the
  marginals (terms in phi_star). B and C are equal respectively to the OT
  matrix and its transpose, i.e. a n x m and m x n matrices. Note that we
  never instantiate those transport matrices, but instead resort to calling
  the app_transport method from the `Geometry` object (which will either use
  potentials or scalings, depending on `lse_mode`.

  The Hessian is symmetric definite. Rather than solve the linear system
  directly we exploit the block diagonal property to use Schur complements.
  Depending on the sizes involved, it is better to instantiate the Schur
  complement of the first or of the second diagonal block. Because either Schur
  complement is rank deficient (1 is a vector with 0 eigenvalue), we use a
  ridge factor, adding 11' to these complements to promote solutions
  orthogonal to 1, i.e. with zero sum.

  Args:
    gr: 2-uple, (vector of size n, vector of size m).
    geom: Geometry object
    a: marginal
    b: marginal
    f: potential, w.r.t marginal a
    g: potential, w.r.t marginal b
    tau_a: float, ratio lam/(lam+eps), ratio of regularizers, first marginal
    tau_b: float, ratio lam/(lam+eps), ratio of regularizers, second marginal
    lse_mode: bool
    ridge: ridge added to promote solutions with 0 sum.

  Returns:
    A tuple of two vectors of the same size as gr.
  """
  marginal_a, marginal_b, app_transport = get_transport_functions(geom,
                                                                  lse_mode)

  vjp_fg = lambda z: app_transport(f, g, z, axis=1) / geom.epsilon
  vjp_gf = lambda z: app_transport(f, g, z, axis=0) / geom.epsilon

  diag_hess_a = (marginal_a(f, g) / geom.epsilon +
                 diag_jacobian_of_marginal_fit(a, f, tau_a, geom.epsilon))
  diag_hess_b = (marginal_b(f, g) / geom.epsilon +
                 diag_jacobian_of_marginal_fit(b, g, tau_b, geom.epsilon))

  # fork on either Schur complement of A or D, depending on size.
  # since the Schur complement has a 0 eigenvalue for vector of 1, we use
  # https://mathoverflow.net/questions/35643/conjugate-gradient-for-a-slightly-singular-system
  # wrapping solver because of https://github.com/google/jax/issues/4322
  my_cg = lambda f, b: jax.scipy.sparse.linalg.cg(f, b)[0]

  if geom.shape[0] > geom.shape[1]:
    inv_vjp_ff = lambda z: z / diag_hess_a
    vjp_gg = lambda z: z * diag_hess_b
    schur = lambda z: (
        vjp_gg(z) - vjp_gf(inv_vjp_ff(vjp_fg(z))) + ridge * jnp.sum(z))
    out = jax.lax.custom_linear_solve(
        schur, jnp.stack((vjp_gf(inv_vjp_ff(gr[0])), gr[1])), my_cg)
    sch_f, sch_g = out[0,:], out[1,:]
    vjp_gr_f = inv_vjp_ff(gr[0] + vjp_fg(sch_f) - vjp_fg(sch_g))
    vjp_gr_g = -sch_f + sch_g
  else:
    vjp_ff = lambda z: z * diag_hess_a
    inv_vjp_gg = lambda z: z / diag_hess_b
    schur = lambda z: (
        vjp_ff(z) - vjp_fg(inv_vjp_gg(vjp_gf(z))) + ridge * jnp.sum(z))
    out = jax.lax.custom_linear_solve(
        schur, jnp.stack((vjp_fg(inv_vjp_gg(gr[1])), gr[0])), my_cg)
    sch_g, sch_f = out[0,:], out[1,:]
    vjp_gr_g = inv_vjp_gg(gr[1] + vjp_gf(sch_g) - vjp_gf(sch_f))
    vjp_gr_f = -sch_g + sch_f

  return jnp.concatenate((-vjp_gr_f, -vjp_gr_g))


def first_order_conditions(geom: geometry.Geometry,
                           a: jnp.ndarray,
                           b: jnp.ndarray,
                           f: jnp.ndarray,
                           g: jnp.ndarray,
                           tau_a: float,
                           tau_b: float,
                           lse_mode):
  """Computes vector of first order conditions for the reg-OT problem.

  The output of this vector should be close to zero at optimality.
  Upon completion of the Sinkhorn forward pass, its norm (as computed using
  the norm_error setting) should be below the threshold parameter.

  This error will be itself assumed to be close to zero when using implicit
  differentiation.

  Args:
    geom: a geometry object
    a: jnp.ndarray, first marginal
    b: jnp.ndarray, second marginal
    f: jnp.ndarray, first potential
    g: jnp.ndarray, second potential
    tau_a: float, ratio lam/(lam+eps), ratio of regularizers, first marginal
    tau_b: float, ratio lam/(lam+eps), ratio of regularizers, second marginal
    lse_mode: bool

  Returns:
    a jnp.ndarray of size (size of f + size of g) quantifying deviation to
    optimality.
  """
  marginal_a, marginal_b, _ = get_transport_functions(geom, lse_mode)

  grad_a = grad_of_marginal_fit(a, f, tau_a, geom.epsilon)
  grad_b = grad_of_marginal_fit(b, g, tau_b, geom.epsilon)
  return jnp.concatenate((
      jnp.where(a > 0, marginal_a(f, g) - grad_a, 0.0),
      jnp.where(b > 0, marginal_b(f, g) - grad_b, 0.0)))

