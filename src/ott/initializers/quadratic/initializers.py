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
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import geometry

if TYPE_CHECKING:
  from ott.problems.linear import linear_problem
  from ott.problems.quadratic import quadratic_problem

__all__ = ["BaseQuadraticInitializer", "QuadraticInitializer"]


@jax.tree_util.register_pytree_node_class
class BaseQuadraticInitializer(abc.ABC):
  """Base class for quadratic initializers.

  Args:
    kwargs: Keyword arguments.
  """

  def __init__(self, **kwargs: Any):
    self._kwargs = kwargs

  def __call__(
      self, quad_prob: "quadratic_problem.QuadraticProblem", **kwargs: Any
  ) -> "linear_problem.LinearProblem":
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
    assert geom.shape == (n, m), (
        f"Expected geometry of shape `{n, m}`, "
        f"found `{geom.shape}`."
    )
    return linear_problem.LinearProblem(
        geom,
        a=quad_prob.a,
        b=quad_prob.b,
        tau_a=quad_prob.tau_a,
        tau_b=quad_prob.tau_b,
    )

  @abc.abstractmethod
  def _create_geometry(
      self, quad_prob: "quadratic_problem.QuadraticProblem", **kwargs: Any
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_prob: Quadratic problem.
      kwargs: Additional keyword arguments.

    Returns:
      Geometry used to initialize the linearized problem.
    """

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [], self._kwargs

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "BaseQuadraticInitializer":
    return cls(*children, **aux_data)


class QuadraticInitializer(BaseQuadraticInitializer):
  r"""Initialize a linear problem locally around a selected coupling.

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
  `gw_unbalanced_correction` is flag indicating whether the unbalanced
  correction applies. The equation of the local cost can be written as:

  .. math::

    \text{marginal_dep_term} + \text{left}_x(\text{cost_xx}) P
    \text{right}_y(\text{cost_yy}) + \text{unbalanced_correction}

  When working with the fused problem, a linear term is added to the cost
  matrix: `cost_matrix` += `fused_penalty` * `geom_xy.cost_matrix`

  Args:
    init_coupling: The coupling to use for initialization. If :obj:`None`,
      defaults to the product coupling :math:`ab^T`.
  """

  def __init__(
      self, init_coupling: Optional[jnp.ndarray] = None, **kwargs: Any
  ):
    super().__init__(**kwargs)
    self.init_coupling = init_coupling

  def _create_geometry(
      self,
      quad_prob: "quadratic_problem.QuadraticProblem",
      *,
      epsilon: float,
      relative_epsilon: Optional[bool] = None,
      **kwargs: Any,
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_prob: Quadratic OT problem.
      epsilon: Epsilon regularization.
      relative_epsilon: Flag, use `relative_epsilon` or not in geometry.
      kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.

    Returns:
      The initial geometry used to initialize the linearized problem.
    """
    from ott.problems.quadratic import quadratic_problem

    del kwargs

    marginal_cost = quad_prob.marginal_dependent_cost(quad_prob.a, quad_prob.b)
    geom_xx, geom_yy = quad_prob.geom_xx, quad_prob.geom_yy

    h1, h2 = quad_prob.quad_loss
    if self.init_coupling is None:
      tmp1 = quadratic_problem.apply_cost(geom_xx, quad_prob.a, axis=1, fn=h1)
      tmp2 = quadratic_problem.apply_cost(geom_yy, quad_prob.b, axis=1, fn=h2)
      tmp = jnp.outer(tmp1, tmp2)
    else:
      tmp1 = h1.func(geom_xx.cost_matrix)
      tmp2 = h2.func(geom_yy.cost_matrix)
      tmp = tmp1 @ self.init_coupling @ tmp2.T

    if quad_prob.is_balanced:
      cost_matrix = marginal_cost.cost_matrix - tmp
    else:
      # initialize epsilon for Unbalanced GW according to Sejourne et. al (2021)
      init_transport = jnp.outer(quad_prob.a, quad_prob.b)
      marginal_1, marginal_2 = init_transport.sum(1), init_transport.sum(0)

      epsilon *= marginal_1.sum()
      unbalanced_correction = quad_prob.cost_unbalanced_correction(
          init_transport, marginal_1, marginal_2, epsilon=epsilon
      )
      cost_matrix = marginal_cost.cost_matrix - tmp + unbalanced_correction

    cost_matrix += quad_prob.fused_penalty * quad_prob._fused_cost_matrix
    return geometry.Geometry(
        cost_matrix=cost_matrix,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon
    )

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [self.init_coupling], self._kwargs
