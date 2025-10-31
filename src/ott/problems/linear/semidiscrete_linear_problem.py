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
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import semidiscrete_pointcloud
from ott.problems.linear import linear_problem

__all__ = ["SemidiscreteLinearProblem"]


@jtu.register_pytree_node_class
class SemidiscreteLinearProblem:
  """Semidiscrete linear OT problem.

  Instances of this problem can be sampled using the :meth:`sample` method.

  Args:
    geom: Semidiscrete point cloud geometry.
    b: The second marginal. If :obj:`None`, it will be uniform.
    tau_b: If :math:`< 1`, defines how much unbalanced the problem is
      on the second marginal. Currently not implemented.
  """

  def __init__(
      self,
      geom: semidiscrete_pointcloud.SemidiscretePointCloud,
      b: Optional[jax.Array] = None,
      tau_b: float = 1.0,
  ):
    assert tau_b == 1.0, "Unbalanced semidiscrete problem is not supported."
    self.geom = geom
    self._b = b
    self.tau_b = tau_b

  def sample(
      self,
      rng: jax.Array,
      num_samples: int,
      *,
      epsilon: Optional[float] = None,
  ) -> linear_problem.LinearProblem:
    """Sample a linear OT problem.

    Args:
      rng: Random key used for seeding.
      num_samples: Number of samples.
      epsilon: Epsilon regularization. If :obj:`None`, use :attr:`epsilon`.

    Returns:
      The sampled linear problem.
    """
    if epsilon is None:
      epsilon = self.epsilon
    geom = self.geom.sample(rng, num_samples, epsilon=epsilon)
    return linear_problem.LinearProblem(
        geom, a=None, b=self._b, tau_a=1.0, tau_b=self.tau_b
    )

  def potential_fn_from_dual_vec(
      self,
      g: jax.Array,
      *,
      epsilon: Optional[float] = None
  ) -> Callable[[jax.Array], jax.Array]:
    r"""Get potential function from a dual vector using the :term:`c-transform`.

    Args:
      g: Potential vector :math:`\mathbb{g}` of shape ``[m,]``.
      epsilon: Epsilon regularization. If :obj:`None`, use in the :attr:`geom`.

    Returns:
      The dual potential function :math:`f`.
    """
    # `potential_fn_from_dual_vec` accesses only necessary properties of the
    # problem/geometry, so we can pass the semidiscrete point cloud
    prob = linear_problem.LinearProblem(self.geom, b=self.b)
    return prob.potential_fn_from_dual_vec(g, epsilon=epsilon, axis=1)

  @property
  def b(self) -> jnp.ndarray:
    """Second marginal."""
    if self._b is not None:
      return self._b
    _, m = self.geom.shape
    return jnp.full((m,), fill_value=1.0 / m, dtype=self.geom.y.dtype)

  @property
  def epsilon(self) -> jax.Array:
    """Entropic regularization."""
    return self.geom.epsilon

  def tree_flatten(self):  # noqa: D102
    return (self.geom, self._b), {"tau_b": self.tau_b}

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data, children
  ) -> "SemidiscreteLinearProblem":
    return cls(*children, **aux_data)
