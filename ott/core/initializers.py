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
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

from ott.core import linear_problems
from ott.geometry import pointcloud


class SinkhornInitializer(ABC):

  @abstractmethod
  def init_dual_a(
      self, ot_problem: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialization for Sinkhorn potential/ scaling f_u."""

  @abstractmethod
  def init_dual_b(
      self, ot_problem: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialization for Sinkhorn potential/ scaling g_v."""


class DefaultInitializer(SinkhornInitializer):
  """Default Initialization of Sinkhorn dual potentials/ primal scalings."""

  def init_dual_a(
      self, ot_problem: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialization for Sinkhorn potential/ scaling f_u.

    Args:
      ot_problem: OT problem between discrete distributions of size n and m.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      potential/ scaling, array of size n
    """
    a = ot_problem.a
    init_dual_a = jnp.zeros_like(a) if lse_mode else jnp.ones_like(a)
    return init_dual_a

  def init_dual_b(
      self, ot_problem: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialization for Sinkhorn potential/ scaling g_v.

    Args:
      ot_problem: OT problem between discrete distributions of size n and m.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      potential/ scaling, array of size m
    """
    b = ot_problem.b
    init_dual_b = jnp.zeros_like(b) if lse_mode else jnp.ones_like(b)
    return init_dual_b


class GaussianInitializer(DefaultInitializer):
  """GaussianInitializer.

  From :cite:`thornton2022rethinking:22`.
  Compute Gaussian approximations of each pointcloud, then compute closed from
  Kantorovich potential betwen Gaussian approximations using Brenier's theorem
  (adapt convex/ Brenier potential to Kantorovich). Use this Gaussian potential to
  initialize Sinkhorn potentials/ scalings.

  """

  def __init__(self):
    super().__init__()

  def init_dual_a(
      self,
      ot_problem: linear_problems.LinearProblem,
      lse_mode: bool,
  ) -> jnp.ndarray:
    """Gaussian init function.

    Args:
      ot_problem: OT problem description with geometry and weights.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      potential/ scaling f_u, array of size n.
    """
    # import Gaussian here due to circular imports
    from ott.tools.gaussian_mixture import gaussian

    assert isinstance(
        ot_problem.geom, pointcloud.PointCloud
    ), "Gaussian initializer valid only for PointCloud geom"

    x, y = ot_problem.geom.x, ot_problem.geom.y
    a, b = ot_problem.a, ot_problem.b

    gaussian_a = gaussian.Gaussian.from_samples(x, weights=a)
    gaussian_b = gaussian.Gaussian.from_samples(y, weights=b)
    # Brenier potential for cost ||x-y||^2/2, multiply by two for ||x-y||^2
    f_potential = 2 * gaussian_a.f_potential(dest=gaussian_b, points=x)
    f_potential = f_potential - jnp.mean(f_potential)
    f_u = f_potential if lse_mode else ot_problem.scaling_from_potential(
        f_potential
    )
    return f_u


class SortingInitializer(DefaultInitializer):
  """Sorting Init class.

  DualSort algorithm from :cite:`thornton2022rethinking:22`, solve
  non-regularized OT problem via sorting, then compute potential through
  iterated minimum on C-transform and use this potential to initialize
  regularized potential

  Args:
    vectorized_update: Use vectorized inner loop if true.
    tolerance: DualSort convergence threshold.
    max_iter: Max DualSort steps.
  """

  def __init__(
      self,
      vectorized_update: bool = True,
      tolerance: float = 1e-2,
      max_iter: int = 100
  ):
    super().__init__()
    self.tolerance = tolerance
    self.max_iter = max_iter
    self.update_fn = lambda f, mod_cost: jax.lax.cond(
        vectorized_update, _vectorized_update, _coordinate_update, f, mod_cost
    )

  def init_sorting_dual(
      self, modified_cost: jnp.ndarray, init_f: jnp.ndarray
  ) -> jnp.ndarray:
    """Run DualSort algorithm.

    Args:
      modified_cost:  cost matrix minus diagonal column-wise.
      init_f: potential f, array of size n. This is the starting potential,
      which is then updated to make the init potential, so an init of an init.

    Returns:
      potential f, array of size n.
    """

    def body_fn(state):
      prev_f, _, it = state
      new_f = self.update_fn(prev_f, modified_cost)
      diff = jnp.sum((new_f - prev_f) ** 2)
      it += 1
      return new_f, diff, it

    def cond_fn(state):
      _, diff, it = state
      return jnp.logical_and(diff > self.tolerance, it < self.max_iter)

    it = 0
    diff = self.tolerance + 1.0
    state = (init_f, diff, it)

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
      init_f: potential f, array of size n. This is the starting potential,
      which is then updated to make the init potential, so an init of an init.

    Returns:
      potential/ scaling f_u, array of size n.
    """
    assert not ot_problem.geom.is_online, "Sorting initializer does not work for online geom"
    # check for sorted x, y requires pointcloud and could slow initializer
    cost_matrix = ot_problem.geom.cost_matrix

    assert cost_matrix.shape[0] == cost_matrix.shape[
        1], "Requires square cost matrix"

    modified_cost = cost_matrix - jnp.diag(cost_matrix)[None, :]

    n = cost_matrix.shape[0]
    init_f = jnp.zeros(n) if init_f is None else init_f

    f_potential = self.init_sorting_dual(modified_cost, init_f)
    f_potential = f_potential - jnp.mean(f_potential)

    f_u = f_potential if lse_mode else ot_problem.scaling_from_potential(
        f_potential
    )

    return f_u


def _vectorized_update(
    f: jnp.ndarray, modified_cost: jnp.ndarray
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


def _coordinate_update(
    f: jnp.ndarray, modified_cost: jnp.ndarray
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
