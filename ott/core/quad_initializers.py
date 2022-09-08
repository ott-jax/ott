from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import jax

from ott.core import linear_problems
from ott.geometry import geometry

if TYPE_CHECKING:
  from ott.core import initializers_lr, quad_problems, sinkhorn_lr

__all__ = ["QuadraticInitializer", "LRQuadraticInitializer"]


@jax.tree_util.register_pytree_node_class
class BaseQuadraticInitializer(ABC):
  """TODO."""

  def __init__(self, **kwargs: Any):
    self._kwargs = kwargs

  def __call__(
      self, quad_prob: 'quad_problems.QuadraticProblem', **kwargs: Any
  ) -> linear_problems.LinearProblem:
    geom = self._create_geometry(quad_prob, **kwargs)
    return linear_problems.LinearProblem(
        geom,
        a=quad_prob.a,
        b=quad_prob.b,
        tau_a=quad_prob.tau_a,
        tau_b=quad_prob.tau_b
    )

  @abstractmethod
  def _create_geometry(
      self, quad_prob: 'quad_problems.QuadraticProblem', **kwargs: Any
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_problem: Quadratic problem.
      kwargs: Additional keyword arguments.

    Returns:
      Geometry used to initialize
      :class:`~ott.core.linear_problem.LinearProblem`.
    """

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [], self._kwargs

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "BaseQuadraticInitializer":
    return cls(*children, **aux_data)


class QuadraticInitializer(BaseQuadraticInitializer):
  """TODO(michalk8): move docstring.

  the equation of the cost follows eq. 6, p. 1 of :cite:`peyre:16`.
  """

  def _create_geometry(
      self, quad_prob: 'quad_problems.QuadraticProblem', *, epsilon: float,
      **kwargs: Any
  ) -> linear_problems.LinearProblem:
    # TODO(michalk8): update
    from ott.core.quad_problems import apply_cost, update_epsilon_unbalanced

    unbalanced_correction = 0.0
    tmp = quad_prob.init_transport()
    marginal_1 = tmp.sum(1)
    marginal_2 = tmp.sum(0)

    # Initialises cost.
    marginal_cost = quad_prob.marginal_dependent_cost(marginal_1, marginal_2)

    if not quad_prob.is_balanced:
      unbalanced_correction = quad_prob.cost_unbalanced_correction(
          tmp, marginal_1, marginal_2, epsilon, 1.0
      )

    h1, h2 = quad_prob.quad_loss
    tmp = apply_cost(quad_prob.geom_xx, tmp, axis=1, fn=h1)
    tmp = apply_cost(quad_prob.geom_yy, tmp.T, axis=1, fn=h2).T
    cost_matrix = (marginal_cost.cost_matrix - tmp + unbalanced_correction)

    # Initialises epsilon for Unbalanced GW according to Sejourne et al (2021).
    if not quad_prob.is_balanced:
      transport_mass = marginal_1.sum()
      epsilon = update_epsilon_unbalanced(epsilon, transport_mass)

    cost_matrix += quad_prob.fused_penalty * quad_prob._fused_cost_matrix

    return geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon)


class LRQuadraticInitializer(BaseQuadraticInitializer):
  """TODO."""

  def __init__(self, linear_lr_initializer: 'initializers_lr.LRInitializer'):
    super().__init__()
    self._linear_lr_initializer = linear_lr_initializer

  def _create_geometry(
      self, quad_prob: 'quad_problems.QuadraticProblem', **kwargs: Any
  ) -> geometry.Geometry:
    q, r, g = self._linear_lr_initializer(quad_prob, **kwargs)
    tmp_out = sinkhorn_lr.LRSinkhornOutput(
        q=q, r=r, g=g, costs=None, criterions=None, ot_prob=None
    )

    return quad_prob.update_lr_geom(tmp_out)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_unflatten()
    return children + [self._linear_lr_initializer], aux_data
