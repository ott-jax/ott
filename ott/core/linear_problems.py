# Copyright 2022 Google LLC.
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
"""Classes defining OT problem(s) (objective function + utilities)."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.geometry import geometry

MarginalFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
TransportAppFunc = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
                            jnp.ndarray]


@jax.tree_util.register_pytree_node_class
class LinearProblem:
  r"""Linear regularized OT problem.

  .. math::

    \min_P \langle C, P \rangle - \varepsilon H(P)

  s.t. :math:`P \mathbf{1}_m = a, P^T \mathbf{1}_n = b`.

  Args:
    geom: The ground geometry cost of the linear problem.
    a: The first marginal. If `None`, it will be uniform.
    b: The second marginal. If `None`, it will be uniform.
    tau_a: If smaller than `1`, defines how much unbalanced the problem is on
      the first marginal.
    tau_b: If smaller than `1`, defines how much unbalanced the problem is on
      the second marginal.
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
    num_a = self.geom.shape[0]
    return jnp.ones((num_a,)) / num_a if self._a is None else self._a

  @property
  def b(self) -> jnp.ndarray:
    """Second marginal."""
    num_b = self.geom.shape[1]
    return jnp.ones((num_b,)) / num_b if self._b is None else self._b

  @property
  def is_balanced(self) -> bool:
    """Whether the problem is balanced."""
    return self.tau_a == 1.0 and self.tau_b == 1.0

  @property
  def epsilon(self) -> float:
    """Entropic regularization."""
    return self.geom.epsilon

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

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return ([self.geom, self._a, self._b], {
        'tau_a': self.tau_a,
        'tau_b': self.tau_b
    })

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "LinearProblem":
    return cls(*children, **aux_data)
