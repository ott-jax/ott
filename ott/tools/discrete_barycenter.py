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
import jax
import jax.numpy as np

from google3.experimental.brain.ott.core import fixed_point_loop
from google3.experimental.brain.ott.core.ground_geometry import geometry


SinkhornBarycenterOutput = collections.namedtuple(
    'Barycenter', ['f', 'g', 'histogram'])


def discrete_barycenter(geom: geometry.Geometry,
                        a: np.ndarray,
                        weights: np.ndarray = None,
                        threshold: float = 1e-2,
                        inner_iterations: float = 10,
                        max_iterations: int = 2000,
                        debiased: bool = True) -> SinkhornBarycenterOutput:
  """Compute discrete barycenter using https://arxiv.org/abs/2006.02575.

  Args:
   geom: a Cost object able to apply kernels with a certain epsilon.
   a: np.ndarray<float>[batch, geom.num_a]: batch of histograms.
   weights: np.ndarray of weights in the probability simplex
   threshold: (float) the relative threshold on the Sinkhorn error to stop the
     Sinkhorn iterations.
   inner_iterations: (int32) the Sinkhorn error is not recomputed at each
     iteration but every inner_num_iter instead to avoid computational overhead.
   max_iterations: (int32) the maximum number of Sinkhorn iterations.
   debiased: whether to run the debiased version of the Sinkhorn divergence.
  Returns:
   A histogram a of size geom.num_b, the Wasserstein barycenter of vectors a.
  """
  batch_size = a.shape[0]

  if weights is None:
    weights = np.ones((batch_size,)) / batch_size
  if not np.alltrue(weights > 0) or weights.shape[0] != batch_size:
    raise ValueError(f'weights must have positive values and size {batch_size}')
  return _discrete_barycenter(geom, a, weights, threshold, inner_iterations,
                              max_iterations, debiased, a.shape[0],
                              geom.shape[0], geom.shape[1])


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def _discrete_barycenter(geom: geometry.Geometry, a: np.ndarray,
                         weights: np.ndarray, threshold: float,
                         inner_iterations: int, max_iterations: int,
                         debiased: bool, batch_size,
                         num_a, num_b) -> SinkhornBarycenterOutput:
  """Jit'ed function to compute discrete barycenters."""
  log_a = np.log(a)
  f = np.zeros_like(a)
  g = np.zeros((batch_size, num_b))
  d = np.zeros((num_b,))

  err = threshold + 1.0

  parallel_update_potentials = jax.vmap(
      functools.partial(geom.update_potential, axis=1), in_axes=[0, 0, 0, None])

  parallel_apply_lse = jax.vmap(
      lambda f_, g_, eps_: geom.apply_lse_kernel(
          f_, g_, eps_, vec=None, axis=0)[0],
      in_axes=[0, 0, None])
  errors_fn = jax.vmap(
      functools.partial(geom.error, axis=1), in_axes=[0, 0, 0, None])

  def cond_fn(iteration, const, state):  # pylint: disable=unused-argument
    threshold = const[-1]
    err = state[0]
    return err >= threshold

  def body_fn(iteration, const, state, last):
    geom, a, weights, _ = const
    err, d, f, g = state

    eps = geom._epsilon.at(iteration)  # pylint: disable=protected-access
    f = parallel_update_potentials(f, g, log_a, iteration)
    log_kernel_f = parallel_apply_lse(f, g, eps)
    eps_log_b = np.average(log_kernel_f, weights=weights, axis=0)
    if debiased:
      eps_log_b += d
      d = 0.5 * (d + geom.update_potential(
          np.zeros((num_a,)), d, eps_log_b / eps, iteration, axis=0))

    g = eps_log_b[np.newaxis, :] - log_kernel_f
    if last:
      err = np.max(errors_fn(f, g, a, iteration))
    return err, d, f, g
  err, d, f, g = fixed_point_loop.fixpoint_iter(
      cond_fn, body_fn, max_iterations, inner_iterations,
      (geom, a, weights, threshold),
      (err, d, f, g))

  log_kernel_f = parallel_apply_lse(f, g, geom.epsilon)
  eps_log_b = np.average(log_kernel_f, weights=weights, axis=0)
  return SinkhornBarycenterOutput(f, g, np.exp(eps_log_b / geom.epsilon))
