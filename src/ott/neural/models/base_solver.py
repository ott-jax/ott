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
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import tree_util

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr

ScaleCost_t = Union[int, float, Literal["mean", "max_cost", "median"]]
ScaleCostQuad_t = Union[ScaleCost_t, Dict[str, ScaleCost_t]]

__all__ = [
    "BaseOTMatcher",
    "OTMatcherLinear",
    "OTMatcherQuad",
]


def _get_sinkhorn_match_fn(
    ot_solver: Any,
    epsilon: float = 1e-2,
    cost_fn: Optional[costs.CostFn] = None,
    scale_cost: ScaleCost_t = 1.0,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
) -> Callable:

  @jax.jit
  def match_pairs(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    geom = pointcloud.PointCloud(
        x, y, epsilon=epsilon, scale_cost=scale_cost, cost_fn=cost_fn
    )
    return ot_solver(
        linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    )

  return match_pairs


def _get_gromov_match_fn(
    ot_solver: Any,
    cost_fn: Union[Any, Mapping[str, Any]],
    scale_cost: ScaleCostQuad_t,
    tau_a: float,
    tau_b: float,
    fused_penalty: float,
) -> Callable:
  if isinstance(cost_fn, Mapping):
    assert "cost_fn_xx" in cost_fn
    assert "cost_fn_yy" in cost_fn
    cost_fn_xx = cost_fn["cost_fn_xx"]
    cost_fn_yy = cost_fn["cost_fn_yy"]
    if fused_penalty > 0:
      assert "cost_fn_xy" in cost_fn_xx
      cost_fn_xy = cost_fn["cost_fn_xy"]
  else:
    cost_fn_xx = cost_fn_yy = cost_fn_xy = cost_fn

  if isinstance(scale_cost, Mapping):
    assert "scale_cost_xx" in scale_cost
    assert "scale_cost_yy" in scale_cost
    scale_cost_xx = scale_cost["scale_cost_xx"]
    scale_cost_yy = scale_cost["scale_cost_yy"]
    if fused_penalty > 0:
      assert "scale_cost_xy" in scale_cost
      scale_cost_xy = cost_fn["scale_cost_xy"]
  else:
    scale_cost_xx = scale_cost_yy = scale_cost_xy = scale_cost

  @jax.jit
  def match_pairs(
      x_quad: jnp.ndarray,
      y_quad: jnp.ndarray,
      x_lin: Optional[jnp.ndarray],
      y_lin: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    geom_xx = pointcloud.PointCloud(
        x=x_quad, y=x_quad, cost_fn=cost_fn_xx, scale_cost=scale_cost_xx
    )
    geom_yy = pointcloud.PointCloud(
        x=y_quad, y=y_quad, cost_fn=cost_fn_yy, scale_cost=scale_cost_yy
    )
    if fused_penalty > 0:
      geom_xy = pointcloud.PointCloud(
          x=x_lin, y=y_lin, cost_fn=cost_fn_xy, scale_cost=scale_cost_xy
      )
    else:
      geom_xy = None
    prob = quadratic_problem.QuadraticProblem(
        geom_xx,
        geom_yy,
        geom_xy,
        fused_penalty=fused_penalty,
        tau_a=tau_a,
        tau_b=tau_b
    )
    return ot_solver(prob)

  return match_pairs


class BaseOTMatcher:
  """Base class for mini-batch neural OT matching classes."""

  def sample_joint(
      self,
      rng: jax.Array,
      joint_dist: jnp.ndarray,
      source_arrays: Tuple[Optional[jnp.ndarray], ...],
      target_arrays: Tuple[Optional[jnp.ndarray], ...],
  ) -> Tuple[jnp.ndarray, ...]:
    """Resample from arrays according to discrete joint distribution.

    Args:
      rng: Random number generator.
      joint_dist: Joint distribution between source and target to sample from.
      source_arrays: Arrays corresponding to source distriubution to sample
        from.
      target_arrays: Arrays corresponding to target arrays to sample from.

    Returns:
      Resampled source and target arrays.
    """
    _, n_tgt = joint_dist.shape
    tmat_flattened = joint_dist.flatten()
    indices = jax.random.choice(
        rng, len(tmat_flattened), p=tmat_flattened, shape=[joint_dist.shape[0]]
    )
    indices_source = indices // n_tgt
    indices_target = indices % n_tgt
    return tree_util.tree_map(lambda b: b[indices_source],
                              source_arrays), tree_util.tree_map(
                                  lambda b: b[indices_target], target_arrays
                              )

  def sample_conditional_indices_from_tmap(
      self,
      rng: jax.Array,
      conditional_distributions: jnp.ndarray,
      *,
      k_samples_per_x: int,
      source_arrays: Tuple[Optional[jnp.ndarray], ...],
      target_arrays: Tuple[Optional[jnp.ndarray], ...],
      source_is_balanced: bool,
  ) -> Tuple[jnp.ndarray, ...]:
    """Sample from arrays according to discrete conditional distributions.

    Args:
      rng: Random number generator.
      conditional_distributions: Conditional distributions to sample from.
      k_samples_per_x: Expectation of number of samples to draw from each
        conditional distribution.
      source_arrays: Arrays corresponding to source distriubution to sample
        from.
      target_arrays: Arrays corresponding to target arrays to sample from.
      source_is_balanced: Whether the source distribution is balanced.
        If :obj:`False`, the number of samples drawn from each conditional
        distribution `k_samples_per_x` is proportional to the left marginals.

    Returns:
      Resampled source and target arrays.
    """
    n_src, n_tgt = conditional_distributions.shape
    left_marginals = conditional_distributions.sum(axis=1)
    if not source_is_balanced:
      rng, rng_2 = jax.random.split(rng, 2)
      indices = jax.random.choice(
          key=rng_2,
          a=jnp.arange(len(left_marginals)),
          p=left_marginals,
          shape=(len(left_marginals),)
      )
    else:
      indices = jnp.arange(n_src)
    tmat_adapted = conditional_distributions[indices]
    indices_per_row = jax.vmap(
        lambda row: jax.random.
        choice(key=rng, a=n_tgt, p=row, shape=(k_samples_per_x,)),
        in_axes=0,
        out_axes=0,
    )(
        tmat_adapted
    )

    indices_source = jnp.repeat(indices, k_samples_per_x)
    indices_target = jnp.reshape(
        indices_per_row % n_tgt, (n_src * k_samples_per_x,)
    )
    return tree_util.tree_map(
        lambda b: jnp.
        reshape(b[indices_source],
                (k_samples_per_x, n_src, *b.shape[1:])), source_arrays
    ), tree_util.tree_map(
        lambda b: jnp.
        reshape(b[indices_target],
                (k_samples_per_x, n_src, *b.shape[1:])), target_arrays
    )


class OTMatcherLinear(BaseOTMatcher):
  """Class for mini-batch OT in neural optimal transport solvers.

  Args:
    ot_solver: OT solver to match samples from the source and the target
      distribution as proposed in :cite:`tong:23`, :cite:`pooladian:23`.
      If :obj:`None`, no matching will be performed as proposed in
      :cite:`lipman:22`.
  """

  def __init__(
      self,
      ot_solver: sinkhorn.Sinkhorn,
      epsilon: float = 1e-2,
      cost_fn: Optional[costs.CostFn] = None,
      scale_cost: ScaleCost_t = 1.0,
      tau_a: float = 1.0,
      tau_b: float = 1.0,
  ) -> None:

    if isinstance(
        ot_solver, gromov_wasserstein.GromovWasserstein
    ) and epsilon is not None:
      raise ValueError(
          "If `ot_solver` is `GromovWasserstein`, `epsilon` must be `None`. " +
          "This check is performed to ensure that in the (fused) Gromov case " +
          "the `epsilon` parameter is passed via the `ot_solver`."
      )
    self.ot_solver = ot_solver
    self.epsilon = epsilon
    self.cost_fn = cost_fn
    self.scale_cost = scale_cost
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.match_fn = None if ot_solver is None else self._get_sinkhorn_match_fn(
        self.ot_solver, self.epsilon, self.cost_fn, self.scale_cost, self.tau_a,
        self.tau_b
    )

  def _get_sinkhorn_match_fn(self, *args, **kwargs) -> jnp.ndarray:
    fn = _get_sinkhorn_match_fn(*args, **kwargs)

    @jax.jit
    def match_pairs(*args, **kwargs):
      return fn(*args, **kwargs).matrix

    return match_pairs


class OTMatcherQuad(BaseOTMatcher):
  """Class for mini-batch OT in neural optimal transport solvers.

  Args:
    ot_solver: OT solver to match samples from the source and the target
      distribution as proposed in :cite:`tong:23`, :cite:`pooladian:23`.
      If :obj:`None`, no matching will be performed as proposed in
      :cite:`lipman:22`.
  """

  def __init__(
      self,
      ot_solver: Union[gromov_wasserstein.GromovWasserstein,
                       gromov_wasserstein_lr.LRGromovWasserstein],
      cost_fn: Optional[costs.CostFn] = None,
      scale_cost: ScaleCostQuad_t = 1.0,
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      fused_penalty: float = 0.0,
  ) -> None:
    self.ot_solver = ot_solver
    self.cost_fn = cost_fn
    self.scale_cost = scale_cost
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.fused_penalty = fused_penalty
    self.match_fn = self._get_gromov_match_fn(
        self.ot_solver,
        self.cost_fn,
        self.scale_cost,
        self.tau_a,
        self.tau_b,
        fused_penalty=self.fused_penalty
    )

  def _get_gromov_match_fn(self, *args, **kwargs) -> jnp.ndarray:
    fn = _get_gromov_match_fn(*args, **kwargs)

    @jax.jit
    def match_pairs(*args, **kwargs):
      return fn(*args, **kwargs).matrix

    return match_pairs
