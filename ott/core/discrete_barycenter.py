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

# Lint as: python3
"""Implementation of Janati+(2020) Wasserstein barycenter algorithm."""

import collections
import functools
from typing import Sequence

import jax
import jax.numpy as jnp

from ott.core import fixed_point_loop
from ott.core import sinkhorn
from ott.geometry import geometry


SinkhornBarycenterOutput = collections.namedtuple(
    'Barycenter', ['f', 'g', 'histogram', 'errors'])


def discrete_barycenter(geom: geometry.Geometry,
                        a: jnp.ndarray,
                        weights: jnp.ndarray = None,
                        dual_initialization: jnp.ndarray = None,
                        threshold: float = 1e-2,
                        norm_error: int = 1,
                        inner_iterations: float = 10,
                        min_iterations: int = 0,
                        max_iterations: int = 2000,
                        lse_mode: bool = True,
                        debiased: bool = False) -> SinkhornBarycenterOutput:
  """Compute discrete barycenter using https://arxiv.org/abs/2006.02575.

  Args:
    geom: a Cost object able to apply kernels with a certain epsilon.
    a: jnp.ndarray<float>[batch, geom.num_a]: batch of histograms.
    weights: jnp.ndarray of weights in the probability simplex
    dual_initialization: jnp.ndarray, size [batch, num_b] initialization for g_v
    threshold: (float) tolerance to monitor convergence.
    norm_error: int, power used to define p-norm of error for marginal/target.
    inner_iterations: (int32) the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead to avoid computational overhead.
    min_iterations: (int32) the minimum number of Sinkhorn iterations carried
     out before the error is computed and monitored.
    max_iterations: (int32) the maximum number of Sinkhorn iterations.
    lse_mode: True for log-sum-exp computations, False for kernel multiply.
    debiased: whether to run the debiased version of the Sinkhorn divergence.

  Returns:
    A ``SinkhornBarycenterOutput``, which contains two arrays of potentials,
    each of size ``batch`` times ``geom.num_a``, summarizing the OT between each
    histogram in the database onto the barycenter, described in ``histogram``,
    as well as a sequence of errors that monitors convergence.
  """
  batch_size, num_a = a.shape
  _, num_b = geom.shape

  if weights is None:
    weights = jnp.ones((batch_size,)) / batch_size
  if not jnp.alltrue(weights > 0) or weights.shape[0] != batch_size:
    raise ValueError(f'weights must have positive values and size {batch_size}')

  if dual_initialization is None:
    # initialization strategy from https://arxiv.org/pdf/1503.02533.pdf, (3.6)
    dual_initialization = geom.apply_cost(a.T, axis=0).T
    dual_initialization -= jnp.average(dual_initialization,
                                       weights=weights,
                                       axis=0)[jnp.newaxis, :]

  if debiased and not geom.is_symmetric:
    raise ValueError('Geometry must be symmetric to use debiased option.')
  norm_error = (norm_error,)
  return _discrete_barycenter(geom, a, weights, dual_initialization, threshold,
                              norm_error, inner_iterations, min_iterations,
                              max_iterations, lse_mode, debiased, num_a, num_b)


@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def _discrete_barycenter(geom: geometry.Geometry,
                         a: jnp.ndarray,
                         weights: jnp.ndarray,
                         dual_initialization: jnp.ndarray,
                         threshold: float,
                         norm_error: Sequence[int],
                         inner_iterations: int,
                         min_iterations: int,
                         max_iterations: int,
                         lse_mode: bool,
                         debiased: bool,
                         num_a: int,
                         num_b: int) -> SinkhornBarycenterOutput:
  """Jit'able function to compute discrete barycenters."""
  if lse_mode:
    f_u = jnp.zeros_like(a)
    g_v = dual_initialization
  else:
    f_u = jnp.ones_like(a)
    g_v = geom.scaling_from_potential(dual_initialization)
  # d below is as described in https://arxiv.org/abs/2006.02575. Note that
  # d should be considered to be equal to eps log(d) with those notations
  # if running in log-sum-exp mode.
  d = jnp.zeros((num_b,)) if lse_mode else jnp.ones((num_b,))

  if lse_mode:
    parallel_update = jax.vmap(
        lambda f, g, marginal, iter: geom.update_potential(
            f, g, jnp.log(marginal), axis=1),
        in_axes=[0, 0, 0, None])
    parallel_apply = jax.vmap(
        lambda f_, g_, eps_: geom.apply_lse_kernel(
            f_, g_, eps_, vec=None, axis=0)[0],
        in_axes=[0, 0, None])
  else:
    parallel_update = jax.vmap(
        lambda f, g, marginal, iter: geom.update_scaling(g, marginal, axis=1),
        in_axes=[0, 0, 0, None])
    parallel_apply = jax.vmap(
        lambda f_, g_, eps_: geom.apply_kernel(f_, eps_, axis=0),
        in_axes=[0, 0, None])

  errors_fn = jax.vmap(
      functools.partial(
          sinkhorn.marginal_error,
          geom=geom,
          axis=1,
          norm_error=norm_error,
          lse_mode=lse_mode),
      in_axes=[0, 0, 0])

  errors = - jnp.ones(
      (max_iterations // inner_iterations + 1, len(norm_error)))

  const = (geom, a, weights)
  def cond_fn(iteration, const, state):  # pylint: disable=unused-argument
    errors = state[0]
    return jnp.logical_or(
        iteration == 0,
        errors[iteration // inner_iterations - 1, 0] > threshold)

  def body_fn(iteration, const, state, compute_error):
    geom, a, weights = const
    errors, d, f_u, g_v = state

    eps = geom._epsilon.at(iteration)  # pylint: disable=protected-access
    f_u = parallel_update(f_u, g_v, a, iteration)
    # kernel_f_u stands for K times potential u if running in scaling mode,
    # eps log K exp f / eps in lse mode.
    kernel_f_u = parallel_apply(f_u, g_v, eps)
    # b below is the running estimate for the barycenter if running in scaling
    # mode, eps log b if running in lse mode.
    if lse_mode:
      b = jnp.average(kernel_f_u, weights=weights, axis=0)
    else:
      b = jnp.prod(kernel_f_u ** weights[:, jnp.newaxis], axis=0)

    if debiased:
      if lse_mode:
        b += d
        d = 0.5 * (
            d +
            geom.update_potential(jnp.zeros((num_a,)), d,
                                  b / eps,
                                  iteration=iteration, axis=0))
      else:
        b *= d
        d = jnp.sqrt(
            d *
            geom.update_scaling(d, b,
                                iteration=iteration, axis=0))
    if lse_mode:
      g_v = b[jnp.newaxis, :] - kernel_f_u
    else:
      g_v = b[jnp.newaxis, :] / kernel_f_u

    # re-compute error if compute_error is True, else set to inf.
    err = jnp.where(
        jnp.logical_and(compute_error, iteration >= min_iterations),
        jnp.mean(errors_fn(f_u, g_v, a)),
        jnp.inf)

    errors = errors.at[iteration // inner_iterations, :].set(err)
    return errors, d, f_u, g_v

  state = (errors, d, f_u, g_v)

  state = fixed_point_loop.fixpoint_iter_backprop(cond_fn, body_fn,
                                                  min_iterations,
                                                  max_iterations,
                                                  inner_iterations, const,
                                                  state)

  errors, d, f_u, g_v = state
  kernel_f_u = parallel_apply(f_u, g_v, geom.epsilon)
  if lse_mode:
    b = jnp.average(kernel_f_u, weights=weights, axis=0)
  else:
    b = jnp.prod(kernel_f_u ** weights[:, jnp.newaxis], axis=0)

  if debiased:
    if lse_mode:
      b += d
    else:
      b *= d
  if lse_mode:
    b = jnp.exp(b / geom.epsilon)
  return SinkhornBarycenterOutput(f_u, g_v, b, errors)
