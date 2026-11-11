# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import geometry

if TYPE_CHECKING:
  from ott.problems.linear import linear_problem
  from ott.problems.quadratic import quadratic_problem

__all__ = ["BaseQuadraticInitializer", "QuadraticInitializer"]


@jtu.register_pytree_node_class
class BaseQuadraticInitializer(abc.ABC):
  """Base class for quadratic initializers."""

  def __call__(
      self, quad_prob: "quadratic_problem.QuadraticProblem", **kwargs: Any
  ) -> "linear_problem.LinearProblem":
    """Compute the initial linearization of a quadratic problem.

    Args:
      quad_prob: Quadratic problem to linearize.
      kwargs: Additional keyword arguments.

    Returns:
      The linearized problem.
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

  def init_transport_mass(
      self, quad_prob: "quadratic_problem.QuadraticProblem"
  ) -> float:
    """Mass of the coupling around which the problem is linearized.

    Args:
      quad_prob: Quadratic problem.

    Returns:
      The transport mass.
    """
    return quad_prob.init_transport_mass()

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
    return [], {}

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "BaseQuadraticInitializer":
    return cls(*children, **aux_data)


@jtu.register_pytree_node_class
class QuadraticInitializer(BaseQuadraticInitializer):
  r"""Initialize a linear problem locally around a selected coupling.

  If the problem is balanced (``tau_a = 1`` and ``tau_b = 1``), the cost is
  the gradient of the GW objective, i.e., twice eq. 6, p. 1 of
  :cite:`peyre:16`.

  If the problem is unbalanced (``tau_a < 1`` or ``tau_b < 1``), there are two
  possible cases. A first possibility is to introduce a quadratic KL
  divergence on the marginals in the objective as done in :cite:`sejourne:21`
  (``gw_unbalanced_correction = True``), which in turns modifies the
  local cost matrix.

  Alternatively, it could be possible to leave the formulation of the
  local cost unchanged, i.e. use the gradient as in the balanced case
  (``gw_unbalanced_correction = False``) and include the unbalanced terms
  at the level of the linear problem only.

  Let :math:`P` [num_a, num_b] be the transport matrix, `cost_xx`, `cost_yy`
  and `cost_xy` be the cost matrices of `geom_xx`, `geom_yy` and `geom_xy`.
  `left_x` and `right_y` depend on the loss chosen for GW. Without the
  unbalanced correction, the local cost is the gradient of
  ``quadratic + fused_penalty * linear``, i.e.,

  .. math::

    2 (\text{marginal_dep_term} - \text{left}_x(\text{cost_xx}) P
    \text{right}_y(\text{cost_yy})^T)
    + \text{fused_penalty} \cdot \text{cost_xy}.

  With the unbalanced correction, half of this gradient is used and the
  scalar `unbalanced_correction` is added, see :cite:`sejourne:21`, alg. 1,
  and :cite:`thual:22`, alg. 1.

  Args:
    init_coupling: The coupling to use for initialization. If :obj:`None`,
      defaults to the product coupling :math:`ab^T`.
  """

  def __init__(self, init_coupling: Optional[jnp.ndarray] = None):
    super().__init__()
    self.init_coupling = init_coupling

  def _create_geometry(
      self,
      quad_prob: "quadratic_problem.QuadraticProblem",
      *,
      epsilon: float,
      relative_epsilon: Optional[Literal["mean", "std"]] = None,
      **kwargs: Any,
  ) -> geometry.Geometry:
    """Compute initial geometry for linearization.

    Args:
      quad_prob: Quadratic OT problem.
      epsilon: Epsilon regularization.
      relative_epsilon: Flag, use `relative_epsilon` or not in geometry.
        Not supported with the unbalanced correction.
      kwargs: Unused.

    Returns:
      Geometry used to initialize the linearized problem.
    """
    from ott.problems.quadratic import quadratic_problem

    del kwargs
    assert relative_epsilon is None or not quad_prob._is_ugw, \
        "Relative epsilon is not supported with the unbalanced correction."

    geom_xx, geom_yy = quad_prob.geom_xx, quad_prob.geom_yy

    h1, h2 = quad_prob.quad_loss
    if self.init_coupling is None:
      init_transport = jnp.outer(quad_prob.a, quad_prob.b)
      marginal_cost = quad_prob.marginal_dependent_cost(
          quad_prob.a, quad_prob.b
      )
      tmp1 = quadratic_problem.apply_cost(geom_xx, quad_prob.a, axis=1, fn=h1)
      tmp2 = quadratic_problem.apply_cost(geom_yy, quad_prob.b, axis=1, fn=h2)
      tmp = jnp.outer(tmp1, tmp2)
    else:
      init_transport = self.init_coupling
      marginal_cost = quad_prob.marginal_dependent_cost(
          init_transport.sum(1), init_transport.sum(0)
      )
      tmp1 = h1.func(geom_xx.cost_matrix)
      tmp2 = h2.func(geom_yy.cost_matrix)
      tmp = tmp1 @ init_transport @ tmp2.T

    cost_matrix = marginal_cost.cost_matrix - tmp
    cost_matrix += 0.5 * quad_prob.fused_penalty * quad_prob._fused_cost_matrix
    if not quad_prob._is_ugw:
      cost_matrix *= 2.0
    else:
      # initialize epsilon for Unbalanced GW according to Sejourne et. al (2021)
      marginal_1, marginal_2 = init_transport.sum(1), init_transport.sum(0)

      cost_matrix += quad_prob.cost_unbalanced_correction(
          init_transport, marginal_1, marginal_2, epsilon=epsilon
      )
      epsilon *= marginal_1.sum()
    return geometry.Geometry(
        cost_matrix=cost_matrix,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon
    )

  def init_transport_mass(  # noqa: D102
      self, quad_prob: "quadratic_problem.QuadraticProblem"
  ) -> float:
    if self.init_coupling is None:
      return super().init_transport_mass(quad_prob)
    return jax.lax.stop_gradient(self.init_coupling).sum()

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [self.init_coupling], {}
