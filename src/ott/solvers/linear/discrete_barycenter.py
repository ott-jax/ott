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
import functools
from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp

from ott.geometry import geometry
from ott.math import fixed_point_loop
from ott.problems.linear import barycenter_problem
from ott.solvers.linear import sinkhorn

__all__ = ["SinkhornBarycenterOutput", "FixedBarycenter"]


class SinkhornBarycenterOutput(NamedTuple):  # noqa: D101
  f: jnp.ndarray
  g: jnp.ndarray
  histogram: jnp.ndarray
  errors: jnp.ndarray


@jax.tree_util.register_pytree_node_class
class FixedBarycenter:
  """A Wasserstein barycenter solver for histograms on a common geometry.

  This solver uses a variant of the
  :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` algorithm proposed in
  :cite:`janati:20a` to compute the barycenter of various measures supported on
  the same (common to all) geometry. The geometry is assumed to be either
  symmetric, or to describe costs between a set of points and another. In that
  case all reference measures have support on the first measure, whereas the
  barycenter is supported on the second.

  Args:
    threshold: convergence threshold. The algorithm stops when the marginal
      violations of all transport plans computed for that barycenter go below
      that threshold.
    norm_error: norm used to compute marginal deviation.
    inner_iterations: number of iterations run before recomputing errors.
    min_iterations: number of iterations run without checking whether
      termination criterion is true.
    max_iterations: maximal number of iterations.
    lse_mode: sets computations in kernel (``False``) or log-sum-exp mode.
    debiased: uses debiasing correction to avoid blur due to entropic
      regularization.
  """

  def __init__(
      self,
      threshold: float = 1e-2,
      norm_error: int = 1,
      inner_iterations: float = 10,
      min_iterations: int = 0,
      max_iterations: int = 2000,
      lse_mode: bool = True,
      debiased: bool = False
  ):
    self.threshold = threshold
    self.norm_error = norm_error
    self.inner_iterations = inner_iterations
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.lse_mode = lse_mode
    self.debiased = debiased

  def __call__(
      self,
      fixed_bp: barycenter_problem.FixedBarycenterProblem,
      dual_initialization: Optional[jnp.ndarray] = None,
  ) -> SinkhornBarycenterOutput:
    """Solve barycenter problem, possibly using clever initialization.

    Args:
      fixed_bp: Fixed barycenter problem.
      dual_initialization: Initial value for the g_v potential/scalings,
        one for each of the histograms described in ``fixed_bp``. If ``None``,
        use initialization from :cite:`cuturi:15`, eq. 3.6.

    Returns:
      The barycenter.
    """
    geom = fixed_bp.geom
    a = fixed_bp.a
    num_a, num_b = geom.shape

    weights = fixed_bp.weights

    if dual_initialization is None:
      # initialization strategy from :cite:`cuturi:15`, (3.6).
      dual_initialization = geom.apply_cost(a.T, axis=0).T
      dual_initialization -= jnp.average(
          dual_initialization, weights=weights, axis=0
      )[jnp.newaxis, :]

    if self.debiased and not geom.is_symmetric:
      raise ValueError("Geometry must be symmetric to use debiased option.")
    norm_error = (self.norm_error,)
    return _discrete_barycenter(
        geom, a, weights, dual_initialization, self.threshold, norm_error,
        self.inner_iterations, self.min_iterations, self.max_iterations,
        self.lse_mode, self.debiased, num_a, num_b
    )

  def tree_flatten(self):  # noqa: D102
    aux = vars(self).copy()
    aux.pop("threshold")
    return [
        self.threshold,
    ], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(**aux_data, threshold=children[0])


@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def _discrete_barycenter(
    geom: geometry.Geometry, a: jnp.ndarray, weights: jnp.ndarray,
    dual_initialization: jnp.ndarray, threshold: float,
    norm_error: Sequence[int], inner_iterations: int, min_iterations: int,
    max_iterations: int, lse_mode: bool, debiased: bool, num_a: int, num_b: int
) -> SinkhornBarycenterOutput:
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
        lambda f, g, marginal, iter: geom.
        update_potential(f, g, jnp.log(marginal), axis=1),
        in_axes=[0, 0, 0, None]
    )
    parallel_apply = jax.vmap(
        lambda f_, g_, eps_: geom.
        apply_lse_kernel(f_, g_, eps_, vec=None, axis=0)[0],
        in_axes=[0, 0, None]
    )
  else:
    parallel_update = jax.vmap(
        lambda f, g, marginal, iter: geom.update_scaling(g, marginal, axis=1),
        in_axes=[0, 0, 0, None]
    )
    parallel_apply = jax.vmap(
        lambda f_, g_, eps_: geom.apply_kernel(f_, eps_, axis=0),
        in_axes=[0, 0, None]
    )

  errors_fn = jax.vmap(
      functools.partial(
          sinkhorn.marginal_error,
          geom=geom,
          axis=1,
          norm_error=norm_error,
          lse_mode=lse_mode
      ),
      in_axes=[0, 0, 0]
  )
  errors = -jnp.ones((max_iterations // inner_iterations + 1, len(norm_error)))

  const = (geom, a, weights)

  def cond_fn(iteration, const, state):  # pylint: disable=unused-argument
    errors = state[0]
    return jnp.logical_or(
        iteration == 0, errors[iteration // inner_iterations - 1, 0] > threshold
    )

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
            d + geom.update_potential(
                jnp.zeros((num_a,)), d, b / eps, iteration=iteration, axis=0
            )
        )
      else:
        b *= d
        d = jnp.sqrt(d * geom.update_scaling(d, b, iteration=iteration, axis=0))
    if lse_mode:
      g_v = b[jnp.newaxis, :] - kernel_f_u
    else:
      g_v = b[jnp.newaxis, :] / kernel_f_u

    # re-compute error if compute_error is True, else set to inf.
    err = jnp.where(
        jnp.logical_and(compute_error, iteration >= min_iterations),
        jnp.mean(errors_fn(f_u, g_v, a)), jnp.inf
    )

    errors = errors.at[iteration // inner_iterations, :].set(err)
    return errors, d, f_u, g_v

  state = (errors, d, f_u, g_v)

  state = fixed_point_loop.fixpoint_iter_backprop(
      cond_fn, body_fn, min_iterations, max_iterations, inner_iterations, const,
      state
  )

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
