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
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ott.geometry import semidiscrete_pointcloud
from ott.problems.linear import linear_problem

__all__ = ["SemidiscreteLinearProblem"]


@jtu.register_pytree_node_class
class SemidiscreteLinearProblem:
  """TODO."""

  def __init__(
      self,
      geom: semidiscrete_pointcloud.SemidiscretePointCloud,
      b: Optional[jax.Array] = None,
      tau_b: float = 1.0,
  ):
    self.geom = geom
    self._b = b
    self.tau_b = tau_b

  def materialize(
      self, rng: jax.Array, num_samples: int
  ) -> linear_problem.LinearProblem:
    """TODO."""
    geom = self.geom.materialize(rng, num_samples)
    return linear_problem.LinearProblem(
        geom, a=None, b=self._b, tau_a=1.0, tau_b=self.tau_b
    )

  @property
  def b(self) -> jnp.ndarray:
    """Second marginal."""
    if self._b is not None:
      return self._b
    _, m = self.geom.shape
    return jnp.full((m,), fill_value=1.0 / m, dtype=self.geom.y.dtype)

  def tree_flatten(self):  # noqa: D102
    return (self.geom, self._b), {"tau_b": self.tau_b}

  @classmethod
  def tree_unflatten(  # noqa: D102
      cls, aux_data, children
  ) -> "SemidiscreteLinearProblem":
    return cls(children, **aux_data)
