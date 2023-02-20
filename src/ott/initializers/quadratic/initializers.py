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
import abc
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import geometry

if TYPE_CHECKING:
  from ott.initializers.linear import initializers_lr
  from ott.problems.linear import linear_problem
  from ott.problems.quadratic import quadratic_problem

__all__ = ["QuadraticInitializer", "LRQuadraticInitializer"]


@jax.tree_util.register_pytree_node_class
class BaseQuadraticInitializer(abc.ABC):
  """Base class for quadratic initializers.

  Args:
    kwargs: Keyword arguments.
  """

  def __init__(self, **kwargs: Any):
    self._kwargs = kwargs

  def __call__(
      self, quad_prob: 'quadratic_problem.QuadraticProblem', **kwargs: Any
  ) -> 'linear_problem.LinearProblem':
    """Compute the initial linearization of a quadratic problem.

    Args:
      quad_prob: Quadratic problem to linearize.
      kwargs: Additional keyword arguments.

    Returns:
      Linear problem.
    """
    from ott.problems.linear import linear_problem

    n, m = quad_prob.geom_xx.shape[0], quad_prob.geom_yy.shape[0]
    geom = self._create_geometry(quad_prob, **kwargs)
    assert geom.shape == (n, m), f"Expected geometry of shape `{n, m}`, " \
                                 f"found `{geom.shape}`."
    return linear_problem.LinearProblem(
        geom,
        a=quad_prob.a,
        b=quad_prob.b,
        tau_a=quad_prob.tau_a,
        tau_b=quad_prob.tau_b
    )

  @abc.abstractmethod
  def _create_geometry(
      self, quad_prob: 'quadratic_problem.QuadraticProblem', **kwargs: Any
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_problem: Quadratic problem.
      kwargs: Additional keyword arguments.

    Returns:
      Geometry used to initialize the linearized problem.
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
      self, quad_prob: 'quadratic_problem.QuadraticProblem', *, epsilon: float,
      **kwargs: Any
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_prob: Quadratic OT problem.
      epsilon: Epsilon regularization.
      kwargs: Additional keyword arguments, unused.

    Returns:
      The initial geometry used to initialize the linearized problem.
    """
    from ott.problems.quadratic import quadratic_problem
    del kwargs

    marginal_cost = quad_prob.marginal_dependent_cost(quad_prob.a, quad_prob.b)
    geom_xx, geom_yy = quad_prob.geom_xx, quad_prob.geom_yy

    h1, h2 = quad_prob.quad_loss
    tmp1 = quadratic_problem.apply_cost(geom_xx, quad_prob.a, axis=1, fn=h1)
    tmp2 = quadratic_problem.apply_cost(geom_yy, quad_prob.b, axis=1, fn=h2)
    tmp = jnp.outer(tmp1, tmp2)

    if quad_prob.is_balanced:
      cost_matrix = marginal_cost.cost_matrix - tmp
    else:
      # initialize epsilon for Unbalanced GW according to Sejourne et. al (2021)
      init_transport = jnp.outer(quad_prob.a, quad_prob.b)
      marginal_1, marginal_2 = init_transport.sum(1), init_transport.sum(0)

      epsilon = quadratic_problem.update_epsilon_unbalanced(
          epsilon=epsilon, transport_mass=marginal_1.sum()
      )
      unbalanced_correction = quad_prob.cost_unbalanced_correction(
          init_transport, marginal_1, marginal_2, epsilon=epsilon
      )
      cost_matrix = marginal_cost.cost_matrix - tmp + unbalanced_correction

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
      self, quad_prob: 'quadratic_problem.QuadraticProblem', **kwargs: Any
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_prob: Quadratic OT problem.
      kwargs: Keyword arguments for
        :meth:`~ott.initializers.linear.initializers_lr.LRInitializer.__call__`.

    Returns:
      The initial geometry used to initialize a linear problem.
    """
    from ott.solvers.linear import sinkhorn_lr

    q, r, g = self._linear_lr_initializer(quad_prob, **kwargs)
    tmp_out = sinkhorn_lr.LRSinkhornOutput(
        q=q, r=r, g=g, costs=None, errors=None, ot_prob=None
    )

    return quad_prob.update_lr_geom(tmp_out)

  @property
  def rank(self) -> int:
    """Rank of the transport matrix factorization."""
    return self._linear_lr_initializer.rank

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    children, aux_data = super().tree_flatten()
    return children + [self._linear_lr_initializer], aux_data
