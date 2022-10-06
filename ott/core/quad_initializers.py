from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import jax

from ott.core import linear_problems, sinkhorn_lr
from ott.geometry import geometry

if TYPE_CHECKING:
  from ott.core import initializers_lr, quad_problems

__all__ = ["QuadraticInitializer", "LRQuadraticInitializer"]


@jax.tree_util.register_pytree_node_class
class BaseQuadraticInitializer(ABC):
  """Base class for quadratic initializers.

  Args:
    kwargs: Keyword arguments.
  """

  def __init__(self, **kwargs: Any):
    self._kwargs = kwargs

  def __call__(
      self, quad_prob: 'quad_problems.QuadraticProblem', **kwargs: Any
  ) -> linear_problems.LinearProblem:
    """Compute the initial linearization of a quadratic problem.

    Args:
      quad_prob: Quadratic problem to linearize.
      kwargs: Additional keyword arguments.

    Returns:
      Linear problem.
    """
    n, m = quad_prob.geom_xx.shape[0], quad_prob.geom_yy.shape[0]
    geom = self._create_geometry(quad_prob, **kwargs)
    assert geom.shape == (n, m), f"Expected geometry of shape `{n, m}`, " \
                                 f"found `{geom.shape}`."
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
      The initial geometry used to initialize a linear problem.
    """

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [], self._kwargs

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "BaseQuadraticInitializer":
    return cls(*children, **aux_data)


class QuadraticInitializer(BaseQuadraticInitializer):
  """Initialize a linear problem locally around a naive initializer ab'.

  If the problem is balanced (``tau_a = 1`` and ``tau_b = 1``),
  the equation of the cost follows eq. 6, p. 1 of :cite:`peyre:16`.

  If the problem is unbalanced (``tau_a < 1`` or ``tau_b < 1``), there are two
  possible cases. A first possibility is to introduce a quadratic KL
  divergence on the marginals in the objective as done in :cite:`sejourne:21`
  (``gw_unbalanced_correction = True``), which in turns modifies the
  local cost matrix.

  Alternatively, it could be possible to leave the formulation of the
  local cost unchanged, i.e. follow eq. 6, p. 1 of :cite:`peyre:16`
  (``gw_unbalanced_correction = False``) and include the unbalanced terms
  at the level of the linear problem only.

  Let :math:`P` [num_a, num_b] be the transport matrix, `cost_xx` is the
  cost matrix of `geom_xx` and `cost_yy` is the cost matrix of `geom_yy`.
  `left_x` and `right_y` depend on the loss chosen for GW.
  `gw_unbalanced_correction` is an boolean indicating whether or not the
  unbalanced correction applies.
  The equation of the local cost can be written as:

  `cost_matrix` = `marginal_dep_term`
              + `left_x`(`cost_xx`) :math:`P` `right_y`(`cost_yy`):math:`^T`
              + `unbalanced_correction` * `gw_unbalanced_correction`

  When working with the fused problem, a linear term is added to the cost
  matrix: `cost_matrix` += `fused_penalty` * `geom_xy.cost_matrix`
  """

  def _create_geometry(
      self, quad_prob: 'quad_problems.QuadraticProblem', *, epsilon: float,
      **kwargs: Any
  ) -> linear_problems.LinearProblem:
    """Compute initial geometry for linearization.

    Args:
      quad_prob: Quadratic OT problem.
      epsilon: Epsilon regularization.
      kwargs: Additional keyword arguments, unused.

    Returns:
      The initial geometry used to initialize a linear problem.
    """
    from ott.core.quad_problems import apply_cost, update_epsilon_unbalanced
    del kwargs

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
  """Wrapper that wraps low-rank Sinkhorn initializers.

  Args:
    lr_linear_initializer: Low-rank linear initializer.
  """

  def __init__(self, lr_linear_initializer: 'initializers_lr.LRInitializer'):
    super().__init__()
    self._linear_lr_initializer = lr_linear_initializer

  def _create_geometry(
      self, quad_prob: 'quad_problems.QuadraticProblem', **kwargs: Any
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_prob: Quadratic OT problem.
      kwargs: Keyword arguments for
        :meth:`ott.core.initializers_lr.LRInitializer.__call__`.

    Returns:
      The initial geometry used to initialize a linear problem.
    """
    q, r, g = self._linear_lr_initializer(quad_prob, **kwargs)
    tmp_out = sinkhorn_lr.LRSinkhornOutput(
        q=q, r=r, g=g, costs=None, errors=None, ot_prob=None
    )

    return quad_prob.update_lr_geom(tmp_out)

  @property
  def rank(self) -> int:
    """Rank of the transport matrix factorization."""
    return self._linear_lr_initializer.rank

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    return children + [self._linear_lr_initializer], aux_data
