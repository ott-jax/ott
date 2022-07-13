# Copyright 2022 The OTT Authors
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
"""Sinkhorn initializers."""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from ott.core import linear_problems
from ott.geometry import pointcloud


def _default_dual_a(
    ot_problem: linear_problems.LinearProblem, lse_mode: bool
) -> jnp.ndarray:
  """Return dual potential vector, f.

  Args:
    ot_problem:
    lse_mode: Return potentials if true, scaling if false.

  Returns:
    potential f, array of size n
  """
  a = ot_problem.a
  init_dual_a = jnp.zeros_like(a) if lse_mode else jnp.ones_like(a)
  return init_dual_a


def _default_dual_b(
    ot_problem: linear_problems.LinearProblem, lse_mode: bool
) -> jnp.ndarray:
  """Return dual potential vector, g.

  Args:
    ot_problem:
    lse_mode: Return potentials if true, scaling if false.

  Returns:
    potential g, array of size m
  """
  b = ot_problem.b
  init_dual_b = jnp.zeros_like(b) if lse_mode else jnp.ones_like(b)
  return init_dual_b


def _remove_single_weight_potential(
    weights: jnp.ndarray, init_dual: jnp.ndarray, lse_mode: bool
) -> Tuple[jnp.ndarray]:
  """Cancel dual variables for zero weights.

  Args:
    weights: array of probability masses
    init_dual: dual potential array
    lse_mode: Return potentials if true, scaling if false.
  Returns:
    potential
  """
  return jnp.where(weights > 0, init_dual, -jnp.inf if lse_mode else 0.0)


def remove_weight_potentials(
    weights_a: jnp.ndarray, weights_b: jnp.ndarray, init_dual_a: jnp.ndarray,
    init_dual_b: jnp.ndarray, lse_mode: bool
) -> Tuple[jnp.ndarray]:
  """Cancel dual variables for zero weights.

  Args:
    weights_a: array of probability masses, array of size n
    weights_b: array of probability masses, array of size m
    init_dual_a: potential f, array of size n
    init_dual_b: potential g, array of size m
    lse_mode: Return potentials if true, scaling if false.

  Returns:
    potentials (f,g)
  """
  init_dual_a = _remove_single_weight_potential(
      weights_a, init_dual_a, lse_mode
  )
  init_dual_b = _remove_single_weight_potential(
      weights_b, init_dual_b, lse_mode
  )
  return init_dual_a, init_dual_b


class SinkhornInitializer:
  """Initialization of Sinkhorn dual potentials.

  Args:
    ot_problem: OT problem between discrete distributions of size n and m.
    lse_mode: Return potential if true, scaling if false.

  Returns:
    dual potential, array of size n
  """

  def init_dual_a(
      self, ot_problem: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialzation for Sinkhorn potential f.

    Args:
      ot_problem: OT problem between discrete distributions of size n and m.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      dual potential, array of size n
    """
    return _default_dual_a(ot_problem=ot_problem, lse_mode=lse_mode)

  def init_dual_b(
      self, ot_problem: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialzation for Sinkhorn potential g.

    Args:
      ot_problem: OT problem between discrete distributions of size n and m.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      dual potential, array of size m
    """
    return _default_dual_b(ot_problem=ot_problem, lse_mode=lse_mode)


class GaussianInitializer(SinkhornInitializer):
  """GaussianInitializer.

  From https://arxiv.org/abs/2206.07630.
  Compute Gaussian approximations of each pointcloud, then compute closed from
  Kantorovic potential betwen Gaussian approximations using Brenier's theorem
  (adapt convex/ Brenier potential to Kantoroic). Use this Gaussian potential to
  initialize Sinkhorn potentials.

  Args:
    stop_gradient: Defaults to True.
  """

  def __init__(self, stop_gradient: bool = True) -> None:

    super().__init__()

    self.stop_gradient = stop_gradient

  def init_dual_a(
      self,
      ot_problem: linear_problems.LinearProblem,
      lse_mode: bool,
  ) -> jnp.ndarray:
    """Gaussian init function.

    Args:
      ot_problem: OT problem description with geometry and weights.
      init_f: Pre dual sort initialization, when none sets entries as 0.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      potential f, array of size n.
    """
    # import Gaussian here due to circular imports
    from ott.tools.gaussian_mixture import gaussian

    if not isinstance(ot_problem.geom, pointcloud.PointCloud):
      # warning that init not applied
      return _default_dual_a(ot_problem, lse_mode)
    else:

      x, y = ot_problem.geom.x, ot_problem.geom.y
      a, b = ot_problem.a, ot_problem.b
      if self.stop_gradient:
        x, y = jax.lax.stop_gradient(x), jax.lax.stop_gradient(y)
        a, b = jax.lax.stop_gradient(a), jax.lax.stop_gradient(b)

      gaussian_a = gaussian.Gaussian.from_samples(x, weights=a)
      gaussian_b = gaussian.Gaussian.from_samples(y, weights=b)
      # Brenier potential for cost ||x-y||^2/2, multiply by two for ||x-y||^2
      f_potential = 2 * gaussian_a.f_potential(dest=gaussian_b, points=x)
      f_potential = f_potential - jnp.mean(f_potential)
      f_potential = f_potential if lse_mode else ot_problem.scaling_from_potential(
          f_potential
      )
      return f_potential


class SortingInit(SinkhornInitializer):
  """Sorting Init class.

  DualSort algorithm from https://arxiv.org/abs/2206.07630, solve
  non-regularized OT problem via sorting, then compute potential through
  iterated minimum on C-transform and use this potentials to initialize
  regularized potential

  Args:
    vector_min: Use vectorized inner loop if true. Defaults to True.
    tol: DualSort convergence threshold. Defaults to 1e-2.
    max_iter: Max DualSort steps. Defaults to 100.
    stop_gradient: Do not trace gradient. Defaults to True.
  """

  def __init__(
      self,
      vector_min: bool = True,
      tol: float = 1e-2,
      max_iter: int = 100,
      stop_gradient: bool = True
  ) -> None:

    super().__init__()

    self.tolerance = tol
    self.stop_gradient = stop_gradient
    self.max_iter = max_iter
    self.update_fn = lambda f, mod_cost: jax.lax.cond(
        vector_min, self.vectorized_update, self.coordinate_update, f, mod_cost
    )

  def vectorized_update(
      self, f: jnp.ndarray, modified_cost: jnp.ndarray
  ) -> jnp.ndarray:
    """Inner loop DualSort Update.

    Args:
      f : potential f, array of size n.
      modified_cost: cost matrix minus diagonal column-wise.

    Returns:
      updated potential vector, f.
    """
    f = jnp.min(modified_cost + f[None, :], axis=1)
    return f

  def coordinate_update(
      self, f: jnp.ndarray, modified_cost: jnp.ndarray
  ) -> jnp.ndarray:
    """Coordinate-wise updates within inner loop.

    Args:
      f: potential f, array of size n.
      modified_cost: cost matrix minus diagonal column-wise.

    Returns:
      updated potential vector, f.
    """

    def body_fn(i, f):
      new_f = jnp.min(modified_cost[i, :] + f)
      f = f.at[i].set(new_f)
      return f

    return jax.lax.fori_loop(0, len(f), body_fn, f)

  def init_sorting_dual(
      self, modified_cost: jnp.ndarray, f_potential: jnp.ndarray
  ) -> jnp.ndarray:
    """Run DualSort algorithm.

    Args:
      modified_cost:  cost matrix minus diagonal column-wise.
      f_potential: potential f, array of size n.

    Returns:
      potential f, array of size n.
    """

    def body_fn(state):
      prev_f, _, it = state
      f_potential = self.update_fn(prev_f, modified_cost)
      diff = jnp.sum((f_potential - prev_f) ** 2)
      it += 1
      return f_potential, diff, it

    def cond_fn(state):
      _, diff, it = state
      return jnp.logical_and(diff > self.tolerance, it < self.max_iter)

    it = 0
    diff = self.tolerance + 1.0
    state = (f_potential, diff, it)

    f_potential, _, it = jax.lax.while_loop(
        cond_fun=cond_fn, body_fun=body_fn, init_val=state
    )

    return f_potential

  def init_dual_a(
      self,
      ot_problem: linear_problems.LinearProblem,
      lse_mode: bool,
      init_f: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Apply DualSort algo.

    Args:
      ot_problem: OT problem.
      lse_mode: Return potential if true, scaling if false.
      init_f: potential f, array of size n.

    Returns:
      potential f, array of size n.
    """
    cost_matrix = ot_problem.geom.cost_matrix
    if self.stop_gradient:
      cost_matrix = jax.lax.stop_gradient(cost_matrix)

    modified_cost = cost_matrix - jnp.diag(cost_matrix)[None, :]

    n = cost_matrix.shape[0]
    f_potential = jnp.zeros(n) if init_f is None else init_f

    f_potential = self.init_sorting_dual(modified_cost, f_potential)
    f_potential = f_potential - jnp.mean(f_potential)

    f_potential = f_potential if lse_mode else ot_problem.scaling_from_potential(
        f_potential
    )

    return f_potential
