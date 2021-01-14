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
from typing import Optional, Union, Sequence

import jax
import jax.numpy as np
import numpy as onp
from ott.core import fixed_point_loop
from ott.core.ground_geometry import geometry


SinkhornOutput = collections.namedtuple(
    'SinkhornOutput', ['f', 'g', 'reg_ot_cost', 'errors', 'converged'])


def sinkhorn(geom: geometry.Geometry,
             a: Optional[np.ndarray] = None,
             b: Optional[np.ndarray] = None,
             tau_a: float = 1.0,
             tau_b: float = 1.0,
             threshold: float = 1e-2,
             norm_error: int = 1,
             inner_iterations: int = 10,
             max_iterations: int = 2000,
             momentum_strategy: Optional[Union[float, str]] = None,
             lse_mode: bool = True) -> SinkhornOutput:
  """Runs Sinkhorn iterations, using convergence parameters and momentum.

  The Sinkhorn algorithm may not converge within the maximum number
  of iterations for possibly two reasons:
    1. the regularizer you are using (defined as epsilon in the geometry
  geom object) is likely too small. Consider increasing it, or,
  alternatively, if you are sure that value is correct, or your cannot
  modify it, either increase max_iterations or threshold;
    2. the probability weights a and b you use do not have the same total
  mass, while you are using a balanced (tau_a=tau_b=1.0) setup. You
  should either normalize data or set either tau_a and/or tau_b <1.0

  Args:
    geom: a GroundGeometry object.
    a: np.ndarray<float>[num_a,] or np.ndarray<float>[batch,num_a] weights.
    b: np.ndarray<float>[num_b,] or np.ndarray<float>[batch,num_b] weights.
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
   norm_error: int, power used to define p-norm of error from marginal to target
   inner_iterations: (int32) the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead.
   max_iterations: (int32) the maximum number of Sinkhorn iterations.
   momentum_strategy: either a float between ]0,2[ or a string.
   lse_mode: True for log-sum-exp computations, False for kernel multiplication.

  Returns:
    a SinkhornOutput named tuple.

  Raises:
    ValueError: If momentum parameter is not set correctly, or to a wrong value.
  """
  num_a, num_b = geom.shape
  a = np.ones((num_a,)) / num_a if a is None else a
  b = np.ones((num_b,)) / num_b if b is None else b

  # Set momentum strategy if the problem is balanced or semi-balanced.
  if (tau_a == 1 or tau_b == 1) and momentum_strategy is None:
    momentum_strategy = 'lehmann'
  elif momentum_strategy is None:
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
    chg_momentum_from = onp.maximum(20 // inner_iterations, 2)
  elif isinstance(momentum_strategy, numbers.Number):
    if not 0 < momentum_strategy < 2:
      raise ValueError('Momentum parameter must be strictly between 0 and 2.')
    momentum_default, chg_momentum_from = momentum_strategy, max_iterations + 1
    norm_error = (norm_error,)
  else:
    raise ValueError('Momentum parameter must be either a float in ]0,2[ (when'
                     ' set to 1 one recovers the usual Sinkhorn updates) or '
                     'a valid string.')
  return _sinkhorn_iterations(
      geom, a, b, threshold, norm_error, tau_a, tau_b,
      inner_iterations, max_iterations, momentum_default,
      chg_momentum_from, lse_mode)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def _sinkhorn_iterations(geom: geometry.Geometry,
                         a: np.ndarray,
                         b: np.ndarray,
                         threshold: float,
                         norm_error: Sequence[int],
                         tau_a: float,
                         tau_b: float,
                         inner_iterations,
                         max_iterations,
                         momentum_default,
                         chg_momentum_from,
                         lse_mode) -> SinkhornOutput:
  """Backprop friendly Jit'ed version of the Sinkhorn algorithm.

  Args:
    geom: a GroundGeometry object.
    a: np.ndarray<float>[num_a,] or np.ndarray<float>[batch,num_a] weights.
    b: np.ndarray<float>[num_b,] or np.ndarray<float>[batch,num_b] weights.
    threshold: (float) the relative threshold on the Sinkhorn error to stop the
      Sinkhorn iterations.
    norm_error: t-uple of int, p-norms of marginal / target errors to track
    tau_a: float, ratio lam/(lam+eps) between KL divergence regularizer to first
     marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    tau_b: float, ratio lam/(lam+eps) between KL divergence regularizer to first
     marginal and itself + epsilon regularizer used in the unbalanced
     formulation.
    inner_iterations: (int32) the Sinkhorn error is not recomputed at each
       iteration but every inner_num_iter instead.
    max_iterations: (int32) the maximum number of Sinkhorn iterations.
    momentum_default: float, a float between ]0,2[
    chg_momentum_from: int, # of iterations after which momentum is computed
    lse_mode: True for log-sum-exp computations, False for kernel
      multiplication.

  Returns:
    a SinkhornOutput named tuple.
  """
  num_a, num_b = geom.shape
  if lse_mode:
    f_u, g_v = np.zeros_like(a), np.zeros_like(b)
  else:
    f_u, g_v = np.ones_like(a) / num_a, np.ones_like(b) / num_b

  errors = -np.ones((onp.ceil(max_iterations / inner_iterations).astype(int),
                     len(norm_error)))
  const = (geom, a, b, threshold)

  def compute_cost(geom: geometry.Geometry,
                   a: np.ndarray,
                   b: np.ndarray,
                   tau_a: float,
                   tau_b: float,
                   f: np.ndarray,
                   g: np.ndarray) -> np.ndarray:
    """Computes objective of regularized OT given dual solutions f,g."""
    if tau_a == 1.0:
      contrib_a = np.sum(f * a)
    else:
      rho_a = geom.epsilon * (tau_a / (1 - tau_a))
      contrib_a = - np.sum(a * phi_star(-f, rho_a))
    if tau_b == 1.0:
      contrib_b = np.sum(g * b)
    else:
      rho_b = geom.epsilon * (tau_b / (1 - tau_b))
      contrib_b = -np.sum(b * phi_star(-g, rho_b))

    if tau_a == 1.0 and tau_b == 1.0:
      # Regularized transport cost as in dual formula
      # 4.30 in https://arxiv.org/pdf/1803.00567.pdf. Notice that the double sum
      # < e^f/eps, K e^g/eps> = 1 when using the Sinkhorn algorithm, due to the
      # way we perform updates. The last term is therefore constant and equal to
      # the epsilon regularization strength.
      regularized_transport_cost = contrib_a + contrib_b - geom.epsilon
    else:
      # When unbalanced, several correction terms, including mass difference,
      # included as in  https://arxiv.org/pdf/1910.12958.pdf p.4,
      # dual expression obtained for KL, p.4, with dual in Eq. 15
      regularized_transport_cost = (
          contrib_a + contrib_b - geom.epsilon * (
              np.sum(a * geom.apply_transport_from_potentials(f, g, b, axis=1))
              - np.sum(a) * np.sum(b))
          + 0.5 * geom.epsilon * (np.sum(a) - np.sum(b))**2
          )
    return regularized_transport_cost

  def cond_fn(iteration, const, state):  # pylint: disable=unused-argument
    threshold = const[-1]
    errors = state[0]
    err = errors[iteration // inner_iterations-1, 0]

    return np.logical_or(iteration == 0,
                         np.logical_and(np.isfinite(err), err > threshold))

  def get_momentum(errors, idx):
    """momentum formula, https://arxiv.org/pdf/2012.12562v1.pdf, p.7 and (5)."""
    error_ratio = errors[idx - 1, -1] / errors[idx - 2, -1]
    power = 1.0 / inner_iterations
    return 2.0 / (1.0 + np.sqrt(1.0 - error_ratio ** power))

  def body_fn(iteration, const, state, last):
    """Carries out sinkhorn iteration.

    Depending on lse_mode, these iterations can be either in:
      0. scaling space, this is the usual kernel multiply iteration
      1. log-space for numerical stability.

    Args:
      iteration: iteration number
      const: tuple of constant parameters that do not change throughout the
        loop.
      state: state variables currently updated in the loop.
      last: flag to indicate this is the last iteration in the inner loop

    Returns:
      state variables.
    """
    geom, a, b, _ = const
    errors, f_u, g_v = state
    # if purely unbalanced, monitor error using two successive iterations.
    if tau_a != 1.0 and tau_b != 1.0 and last:
      f_u_copy = f_u

    w = jax.lax.stop_gradient(
        np.where(iteration >= (inner_iterations * chg_momentum_from),
                 get_momentum(errors, chg_momentum_from),
                 momentum_default))

    if lse_mode:
      g_v = (1.0 - w) * g_v + w * tau_b * geom.update_potential(
          f_u, g_v, np.log(b), iteration,
          axis=0)
      f_u = (1.0 - w) * f_u + w * tau_a * geom.update_potential(
          f_u, g_v, np.log(a), iteration,
          axis=1)
    else:
      g_v = g_v ** (1.0 - w) * (geom.update_scaling(
          f_u, b, iteration, axis=0)**tau_b) ** w
      f_u = f_u ** (1.0 - w) * (geom.update_scaling(
          g_v, a, iteration, axis=1)**tau_a) ** w

    if last:
      if tau_b == 1:
        err = geom.error(f_u, g_v, b, 0, norm_error, lse_mode)
      elif tau_a == 1:
        if lse_mode:
          g_v = tau_b * geom.update_potential(f_u, g_v, np.log(b), iteration,
                                              axis=0)
        else:
          g_v = geom.update_scaling(f_u, b, iteration, axis=0) ** tau_b

        err = geom.error(f_u, g_v, a, 1, norm_error, lse_mode)
      else:
        # in the unbalanced case, we just keep track of differences in iterates.
        # these differences are computed on dual potentials.
        if lse_mode:
          err = np.sum(np.abs(f_u-f_u_copy) ** norm_error[0]
                       ) ** (1.0 / norm_error[0])
        else:
          err = np.sum(
              np.abs(geom.potential_from_scaling(f_u) -
                     geom.potential_from_scaling(f_u_copy)) ** norm_error[0]
              ) ** (1.0 / norm_error[0])
      errors = jax.ops.index_update(
          errors, jax.ops.index[iteration // inner_iterations, :], err)
    return errors, f_u, g_v

  errors, f_u, g_v = fixed_point_loop.fixpoint_iter(
      cond_fn, body_fn, max_iterations, inner_iterations, const,
      (errors, f_u, g_v))

  f = f_u if lse_mode else geom.potential_from_scaling(f_u)
  g = g_v if lse_mode else geom.potential_from_scaling(g_v)

  regularized_transport_cost = compute_cost(geom, a, b, tau_a, tau_b, f, g)
  # test if the Sinkhorn algorithm has converged.
  converged = np.logical_and(
      np.sum(errors == -1) > 0,
      np.sum(np.logical_not(np.isfinite(errors))) == 0)

  return SinkhornOutput(f, g, regularized_transport_cost, errors[:, 0],
                        converged)


def phi_star(f: np.ndarray, rho: float) -> np.ndarray:
  """Legendre transform of KL, https://arxiv.org/pdf/1910.12958.pdf p.9."""
  return rho * (np.exp(f / rho) - 1)


