# coding=utf-8
# Copyright 2021 Google LLC.
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

# Lint as: python3
"""A State variable to store iterations and outputs of a Sinkhorn solver."""
from typing import Optional, NamedTuple

import jax
import jax.numpy as jnp
from ott.geometry import geometry


class SinkhornState(NamedTuple):
  """Holds the outputs of the Sinkhorn algorithm."""
  # Output of the algorithm.
  f: Optional[jnp.ndarray] = None
  g: Optional[jnp.ndarray] = None
  reg_ot_cost: Optional[jnp.ndarray] = None  # For backward compatibility.
  errors: Optional[jnp.ndarray] = None
  # Intermediate values.
  fu: Optional[jnp.ndarray] = None
  gv: Optional[jnp.ndarray] = None
  old_fus: Optional[jnp.ndarray] = None
  old_mapped_fus: Optional[jnp.ndarray] = None

  def set(self, **kwargs) -> 'SinkhornState':
    """Returns a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def finalize(self):
    return self.set(fu=None, gv=None, old_fus=None, old_mapped_fus=None)

  def set_cost(self, ot_prob, lse_mode, use_danskin):
    f = jax.lax.stop_gradient(self.f) if use_danskin else self.f
    g = jax.lax.stop_gradient(self.g) if use_danskin else self.g
    cost = ot_prob.ent_reg_cost(f, g, lse_mode)
    return self.set(reg_ot_cost=cost)

  @property
  def converged(self):
    if self.errors is None:
      return False
    return jnp.logical_and(jnp.sum(self.errors == -1) > 0,
                           jnp.sum(jnp.isnan(self.errors)) == 0)

  def scalings(self, geom):
    u = geom.scaling_from_potential(self.f)
    v = geom.scaling_from_potential(self.g)
    return u, v

  def postprocess(self, ot_prob, lse_mode) -> 'SinkhornState':
    fu, gv = self.fu, self.gv
    f = fu if lse_mode else ot_prob.geom.potential_from_scaling(fu)
    g = gv if lse_mode else ot_prob.geom.potential_from_scaling(gv)
    return self.set(f=f, g=g, errors=self.errors[:, 0])

  def matrix(self, geom) -> jnp.ndarray:
    """Transport matrix if it can be instantiated."""
    try:
      return geom.transport_from_potentials(self.f, self.g)
    except ValueError:
      return geom.transport_from_scalings(*self.scalings(geom))

  def apply(self, geom, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Applies the transport to a ndarray; axis=1 for its transpose."""
    try:
      return geom.apply_transport_from_potentials(
          self.f, self.g, inputs, axis=axis)
    except ValueError:
      u, v = self.scalings(geom)
      return geom.apply_transport_from_scalings(u, v, inputs, axis=axis)

  def marginal(self, geom: geometry.Geometry, axis: int) -> jnp.ndarray:
    return geom.marginal_from_potentials(self.f, self.g, axis=axis)
