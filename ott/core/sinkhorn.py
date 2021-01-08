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
from typing import Optional, Union

import jax
import jax.numpy as np
from ott.core import fixed_point_loop
from ott.core.ground_geometry import geometry


SinkhornOutput = collections.namedtuple(
    'SinkhornOutput', ['f', 'g', 'reg_ot_cost', 'errors'])


def sinkhorn(geom: geometry.Geometry,
             a: Optional[np.ndarray] = None,
             b: Optional[np.ndarray] = None,
             tau_a: float = 1.0,
             tau_b: float = 1.0,
             threshold: float = 1e-2,
             norm_error: int = 1,
             inner_iterations: int = 10,
             max_iterations: int = 2000,
             momentum_strategy: Optional[Union[float,str]] = 'Lehmann',
             lse_mode: bool = True) -> SinkhornOutput:
  """Runs Sinkhorn iterations, using convergence parameters and momentum.

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
   threshold: (float) the relative threshold on the Sinkhorn error to stop the
     Sinkhorn iterations.
   norm_error: int, power used to define p-norm of error from marginal to target
   inner_iterations: (int32) the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead.
   max_iterations: (int32) the maximum number of Sinkhorn iterations.
   momentum_strategy: either a float between ]0,2[ or a string.
   lse_mode: True for log-sum-exp computations, False for kernel multiplication.
  Returns:
    tuples of sinkhorn_iterations outputs.
  """
  num_a, num_b = geom.shape
  a = np.ones((num_a,)) / num_a if a is None else a
  b = np.ones((num_b,)) / num_b if b is None else b

  if isinstance(momentum_strategy, str) and momentum_strategy == 'Lehmann':
    if norm_error != 1:
      raise ValueError('To use Lehmann momentum, norm_error should be 1.')
    momentum_default, chg_momentum_from = 1.0, 20 // inner_iterations + 1
  elif isinstance(momentum_strategy, numbers.Number):
    if momentum_strategy > 2 - 1e-10 or momentum_strategy < 0 + 1e-10:
      raise ValueError('Momentum parameter must be strictly between 0 and 2.')
    momentum_default, chg_momentum_from = momentum_strategy, max_iterations + 1

  return _sinkhorn_iterations(
      geom, a, b, threshold, norm_error, tau_a, tau_b,
      inner_iterations, max_iterations, momentum_default,
      chg_momentum_from, lse_mode)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def _sinkhorn_iterations(geom: geometry.Geometry,
                         a,
                         b,
                         threshold: float,
                         norm_error: int,
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
    norm_error: int, p-norm of error between marginal / target
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
    tuples of sinkhorn_iterations outputs.
  """
  num_a, num_b = geom.shape
  if lse_mode:
    f_u, g_v = (np.zeros_like(a), np.zeros_like(b))
  else:
    f_u, g_v = (np.ones_like(a) / num_a,
                np.ones_like(b) / num_b)

  errors = np.ones((max_iterations // inner_iterations + 1,)) * np.inf
  const = (geom, a, b, threshold)

  def cond_fn(iteration, const, state):  # pylint: disable=unused-argument
    threshold = const[-1]
    errors = state[0]
    return np.where(iteration > 0,
                    errors[iteration // inner_iterations-1] > threshold,
                    True)

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
    geom, a, b, threshold = const
    errors, f_u, g_v = state

    w = jax.lax.stop_gradient(
        np.where((iteration > inner_iterations * chg_momentum_from),
                 2.0 / (1.0 + np.sqrt(1.0 - (errors[chg_momentum_from - 1] /
                                             errors[chg_momentum_from - 2])**
                                      (1.0 / inner_iterations))),
                 momentum_default))

    if lse_mode:
      g_v = (1 - w) * g_v + w * tau_b * geom.update_potential(
          f_u, g_v, np.log(b), iteration,
          axis=0)
      f_u = (1 - w) * f_u + w * tau_a * geom.update_potential(
          f_u, g_v, np.log(a), iteration,
          axis=1)
    else:
      g_v = g_v ** (1 - w) * (geom.update_scaling(
          f_u, b, iteration, axis=0)**tau_b) ** w
      f_u = f_u ** (1 - w) * (geom.update_scaling(
          g_v, a, iteration, axis=1)**tau_a) ** w

    if last:
      if tau_b == 1:
        err = geom.error(f_u, g_v, b, iteration, 0, threshold + 1,
                         norm_error, lse_mode)
      elif tau_a == 1:
        if lse_mode:
          g_v = tau_b * geom.update_potential(f_u, g_v, np.log(b), iteration,
                                              axis=0)
        else:
          g_v = geom.update_scaling(f_u, b, iteration, axis=0) ** tau_b

        err = geom.error(f_u, g_v, a, iteration, 1, threshold + 1, norm_error,
                         lse_mode)
      else:
        # TODO(cuturi,lpapaxanthos): implement convergence for unbalanced case.
        err = threshold
      errors = jax.ops.index_update(errors,
                                    jax.ops.index[iteration// inner_iterations],
                                    err)
    return errors, f_u, g_v

  errors, f_u, g_v = fixed_point_loop.fixpoint_iter(
      cond_fn, body_fn, max_iterations, inner_iterations, const,
      (errors, f_u, g_v))

  f = f_u if lse_mode else geom.potential_from_scaling(f_u)
  g = g_v if lse_mode else geom.potential_from_scaling(g_v)

  # Regularized transport cost as in dual formula
  # 4.30 in https://arxiv.org/pdf/1803.00567.pdf. Notice that the double sum
  # < e^f/eps, K e^g/eps> = 1 when using the Sinkhorn algorithm, due to the
  # way we perform updates. The last term is therefore constant and equal to
  # the epsilon regularization strength.
  # TODO(cuturi): formula below is not correct in unbalanced case.

  regularized_transport_cost = (np.sum(f * a) + np.sum(g * b) - geom.epsilon)
  return SinkhornOutput(f, g, regularized_transport_cost, errors)
