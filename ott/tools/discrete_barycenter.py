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
"""Implementation of Janati+(2020) barycenter algorithm."""

import collections
import functools
from typing import Sequence

import jax
import jax.numpy as np

from ott.core import fixed_point_loop
from ott.core.ground_geometry import geometry


SinkhornBarycenterOutput = collections.namedtuple(
    'Barycenter', ['f', 'g', 'histogram'])


def discrete_barycenter(geom: geometry.Geometry,
                        a: np.ndarray,
                        weights: np.ndarray = None,
                        threshold: float = 1e-2,
                        norm_error: int = 1,
                        inner_iterations: float = 10,
                        max_iterations: int = 2000,
                        lse_mode: bool = True,
                        debiased: bool = False) -> SinkhornBarycenterOutput:
  """Compute discrete barycenter using https://arxiv.org/abs/2006.02575.

  Args:
   geom: a Cost object able to apply kernels with a certain epsilon.
   a: np.ndarray<float>[batch, geom.num_a]: batch of histograms.
   weights: np.ndarray of weights in the probability simplex
   threshold: (float) the relative threshold on the Sinkhorn error to stop the
     Sinkhorn iterations.
   norm_error: int, power used to define p-norm of error from marginal to target
   inner_iterations: (int32) the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead to avoid computational overhead.
   max_iterations: (int32) the maximum number of Sinkhorn iterations.
   lse_mode: True for log-sum-exp computations, False for kernel multiplication.
   debiased: whether to run the debiased version of the Sinkhorn divergence.
  Returns:
   A histogram a of size geom.num_b, the Wasserstein barycenter of vectors a.
  """
  batch_size, num_a = a.shape
  _, num_b = geom.shape

  if weights is None:
    weights = np.ones((batch_size,)) / batch_size
  if not np.alltrue(weights > 0) or weights.shape[0] != batch_size:
    raise ValueError(f'weights must have positive values and size {batch_size}')

  if debiased and not geom.is_symmetric:
    raise ValueError('Geometry must be symmetric to use debiased option.')

  return _discrete_barycenter(geom, a, weights, threshold, (norm_error,),
                              inner_iterations, max_iterations, lse_mode,
                              debiased, batch_size,
                              num_a, num_b)


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 11))
def _discrete_barycenter(geom: geometry.Geometry,
                         a: np.ndarray,
                         weights: np.ndarray,
                         threshold: float,
                         norm_error: Sequence[int],
                         inner_iterations: int,
                         max_iterations: int,
                         lse_mode: bool,
                         debiased: bool,
                         batch_size: int,
                         num_a: int,
                         num_b: int) -> SinkhornBarycenterOutput:
  """Jit'ed function to compute discrete barycenters."""
  f_u = np.zeros_like(a) if lse_mode else np.ones_like(a) / num_a
  g_v = np.zeros((batch_size, num_b)) if lse_mode else np.ones(
      (batch_size, num_b)) / num_b
  # d below is as described in https://arxiv.org/abs/2006.02575. Note that
  # d should be considered to be equal to eps log(d) with those notations
  # if running in log-sum-exp mode.
  d = np.zeros((num_b,)) if lse_mode else np.ones((num_b,))

  if lse_mode:
    parallel_update = jax.vmap(
        lambda f, g, marginal, iter: geom.update_potential(
            f, g, np.log(marginal), axis=1),
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
      functools.partial(geom.error, axis=1, norm_error=norm_error,
                        lse_mode=lse_mode),
      in_axes=[0, 0, 0, None])

  err = threshold + 1.0

  const = (geom, a, weights, threshold)
  def cond_fn(iteration, const, state):  # pylint: disable=unused-argument
    threshold = const[-1]
    err = state[0]
    return err >= threshold
  def body_fn(iteration, const, state, last):
    geom, a, weights, _ = const
    err, d, f_u, g_v = state

    eps = geom._epsilon.at(iteration)  # pylint: disable=protected-access
    f_u = parallel_update(f_u, g_v, a, iteration)
    # kernel_f_u stands for K times potential u if running in scaling mode,
    # eps log K exp f / eps in lse mode.
    kernel_f_u = parallel_apply(f_u, g_v, eps)
    # b below is the running estimate for the barycenter if running in scaling
    # mode, eps log b if running in lse mode.
    if lse_mode:
      b = np.average(kernel_f_u, weights=weights, axis=0)
    else:
      b = np.prod(
          kernel_f_u ** weights[:, np.newaxis], axis=0)
    if debiased:
      if lse_mode:
        b += d
        d = 0.5 * (
            d +
            geom.update_potential(np.zeros((num_a,)), d,
                                  b / eps,
                                  iteration=iteration, axis=0))
      else:
        b *= d
        d = np.sqrt(
            d *
            geom.update_scaling(d, b,
                                iteration=iteration, axis=0))
    if lse_mode:
      g_v = b[np.newaxis, :] - kernel_f_u
    else:
      g_v = b[np.newaxis, :] / kernel_f_u

    if last:
      err = np.max(errors_fn(f_u, g_v, a, iteration))
    return err, d, f_u, g_v
  state = (err, d, f_u, g_v)

  state = fixed_point_loop.fixpoint_iter(cond_fn, body_fn, max_iterations,
                                         inner_iterations, const, state)

  _, d, f_u, g_v = state
  kernel_f_u = parallel_apply(f_u, g_v, geom.epsilon)
  if lse_mode:
    b = np.average(
        kernel_f_u, weights=weights, axis=0)
  else:
    b = np.prod(
        kernel_f_u**weights[:, np.newaxis], axis=0)

  if debiased:
    if lse_mode:
      b += d
    else:
      b *= d
  if lse_mode:
    b = np.exp(b / geom.epsilon)
  return SinkhornBarycenterOutput(f_u, g_v, b)
