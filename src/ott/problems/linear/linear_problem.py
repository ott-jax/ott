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
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import geometry, pointcloud
from ott.math import utils as math_utils

__all__ = ["LinearProblem"]

# TODO(michalk8): move to typing.py when refactoring the types
MarginalFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
TransportAppFunc = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
                            jnp.ndarray]


@jax.tree_util.register_pytree_node_class
class LinearProblem:
  r"""Linear OT problem.

  This class describes the main ingredients appearing in a linear OT problem.
  Namely, a ``geom`` object (including cost structure/points) describing point
  clouds or the support of measures, followed by probability masses ``a`` and
  ``b``. Unbalancedness of the problem is also kept track of, through two
  coefficients ``tau_a`` and ``tau_b``, which are both kept between 0 and 1
  (1 corresponding to a balanced OT problem).

  Args:
    geom: The ground geometry cost of the linear problem.
    a: The first marginal. If :obj:`None`, it will be uniform.
    b: The second marginal. If :obj:`None`, it will be uniform.
    tau_a: If :math:`<1`, defines how much unbalanced the problem is
      on the first marginal.
    tau_b: If :math:`< 1`, defines how much unbalanced the problem is
      on the second marginal.
  """

  def __init__(
      self,
      geom: geometry.Geometry,
      a: Optional[jnp.ndarray] = None,
      b: Optional[jnp.ndarray] = None,
      tau_a: float = 1.0,
      tau_b: float = 1.0
  ):
    self.geom = geom
    self._a = a
    self._b = b
    self.tau_a = tau_a
    self.tau_b = tau_b

  @property
  def a(self) -> jnp.ndarray:
    """First marginal."""
    if self._a is not None:
      return self._a
    n, _ = self.geom.shape
    return jnp.full((n,), fill_value=1.0 / n, dtype=self.dtype)

  @property
  def b(self) -> jnp.ndarray:
    """Second marginal."""
    if self._b is not None:
      return self._b
    _, m = self.geom.shape
    return jnp.full((m,), fill_value=1.0 / m, dtype=self.dtype)

  @property
  def is_balanced(self) -> bool:
    """Whether the problem is balanced."""
    return self.tau_a == 1.0 and self.tau_b == 1.0

  @property
  def is_uniform(self) -> bool:
    """True if no weights ``a,b`` were passed, and have defaulted to uniform."""
    return self._a is None and self._b is None

  @property
  def is_equal_size(self) -> bool:
    """True if square shape, i.e. ``n == m``."""
    return self.geom.shape[0] == self.geom.shape[1]

  @property
  def is_assignment(self) -> bool:
    """True if assignment problem."""
    return self.is_equal_size and self.is_uniform and self.is_balanced

  @property
  def epsilon(self) -> float:
    """Entropic regularization."""
    return self.geom.epsilon

  @property
  def dtype(self) -> jnp.dtype:
    """The data type of the geometry."""
    return self.geom.dtype

  def potential_fn_from_dual_vec(
      self,
      fg: jax.Array,
      *,
      epsilon: Optional[float] = None,
      axis: Literal[0, 1],
  ) -> Callable[[jax.Array], jax.Array]:
    r"""Get potential function from a dual vector using the :term:`c-transform`.

    Args:
      fg: Potential vector :math:`\mathbb{f}` if ``axis = 0``
        else :math:`\mathbb{g}` of shape ``[n,]`` or ``[m,]``, respectively.
      epsilon: Epsilon regularization. If :obj:`None`, use in the :attr:`geom`.
      axis: If ``axis = 0``, return the :math:`g`-potential function, otherwise
        return the :math:`f`-potential function.

    Returns:
      The dual potential function.
    """

    def f_potential(x: jax.Array) -> jax.Array:
      x, y = jnp.atleast_2d(x), self.geom.y
      geom = pointcloud.PointCloud(x, y, cost_fn=self.geom.cost_fn)
      prob = LinearProblem(geom, b=self.b)
      f, _ = prob._c_transform(fg, epsilon=epsilon, axis=axis)
      return f.squeeze(0)

    def g_potential(y: jax.Array) -> jax.Array:
      x, y = self.geom.x, jnp.atleast_2d(y)
      geom = pointcloud.PointCloud(x, y, cost_fn=self.geom.cost_fn)
      prob = LinearProblem(geom, a=self.a)
      g, _ = prob._c_transform(fg, epsilon=epsilon, axis=axis)
      return g.squeeze(0)

    assert axis in (0, 1), axis
    epsilon = self.geom.epsilon if epsilon is None else epsilon
    return g_potential if axis == 0 else f_potential

  def _c_transform(
      self,
      fg: jax.Array,
      *,
      epsilon: Optional[float] = None,
      axis: Literal[0, 1],
  ) -> Tuple[jax.Array, jax.Array]:

    def _soft_c_transform(fg: jax.Array) -> Tuple[jax.Array, jax.Array]:
      cost = self.geom.cost_matrix
      z = (fg - cost) / epsilon
      return -epsilon * math_utils.logsumexp(z, b=self.b, axis=axis), z

    def _hard_c_transform(fg: jax.Array) -> Tuple[jax.Array, jax.Array]:
      cost = self.geom.cost_matrix
      z = fg - cost
      pos_weights = self.b[None, :] > 0.0
      return -jnp.max(z, initial=-jnp.inf, where=pos_weights, axis=axis), z

    assert axis in (0, 1), axis
    fg = jnp.expand_dims(fg, 1 - axis)
    epsilon = self.geom.epsilon if epsilon is None else epsilon
    return jax.lax.cond(epsilon > 0.0, _soft_c_transform, _hard_c_transform, fg)

  def get_transport_functions(
      self, lse_mode: bool
  ) -> Tuple[MarginalFunc, MarginalFunc, TransportAppFunc]:
    """Instantiate useful functions for Sinkhorn depending on lse_mode."""
    geom = self.geom
    if lse_mode:
      marginal_a = lambda f, g: geom.marginal_from_potentials(f, g, 1)
      marginal_b = lambda f, g: geom.marginal_from_potentials(f, g, 0)
      app_transport = geom.apply_transport_from_potentials
    else:
      marginal_a = lambda f, g: geom.marginal_from_scalings(
          geom.scaling_from_potential(f), geom.scaling_from_potential(g), 1
      )
      marginal_b = lambda f, g: geom.marginal_from_scalings(
          geom.scaling_from_potential(f), geom.scaling_from_potential(g), 0
      )
      app_transport = lambda f, g, z, axis: geom.apply_transport_from_scalings(
          geom.scaling_from_potential(f), geom.scaling_from_potential(g), z,
          axis
      )
    return marginal_a, marginal_b, app_transport

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return ([self.geom, self._a, self._b], {
        "tau_a": self.tau_a,
        "tau_b": self.tau_b
    })

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "LinearProblem":
    return cls(*children, **aux_data)
