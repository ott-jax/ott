# coding=utf-8
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

"""Some utility functions for transport computation.

This module is primarily made for new users who are looking for one-liners.
For instance, solving the transport between two point clouds.

>>> x = jax.random.uniform((34, 3))
>>> y = jax.random.uniform((34, 3)) + 1
>>> ot = ott.transport.solve(x, y)
>>> Tz = ot.apply(z)

Even if the transport.solve sole function can support many complex use cases, we
suggest more advanced users to instantiate directly their problem (see
ott.core.problems) and their solvers (see ott.core.sinkhorn and
ott.core.gromov_wasserstein) for better control over the parameters.
"""

from typing import Any, NamedTuple
import jax.numpy as jnp
from ott.core import gromov_wasserstein
from ott.core import problems
from ott.core import quad_problems
from ott.core import sinkhorn


class Transport(NamedTuple):
  """Implements a core.problems.Transport interface to transport solutions."""
  problem: Any = None
  solver_output: Any = None

  @property
  def linear(self):
    return isinstance(self.problem, problems.LinearProblem)

  @property
  def geom(self):
    return self.problem.geom if self.linear else self.solver_output.geom

  @property
  def a(self):
    return self.problem.a

  @property
  def b(self):
    return self.problem.b

  @property
  def linear_output(self):
    out = self.solver_output
    return out if self.linear else out.linear_state

  @property
  def matrix(self):
    return self.linear_output.matrix

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    return self.linear_output.apply(inputs, axis)

  def marginal(self, axis: int = 0) -> jnp.ndarray:
    return self.linear_output.marginal(axis)


def solve(*args, a=None, b=None, objective=None, **kwargs) -> Transport:
  """Generic interface to transport problem.

  The geometries can be passed as arrays, geometry.Geometry or directly as a
  problem. The solver is passed via kwargs.

  Args:
    *args: can be either a single argument, the geometry.Geometry instance, or
      for convenience only two jnp.ndarray<float> corresponding to two point
      clouds. In that case the regularization parameter epsilon must be set in
      the kwargs.
    a: the weights of the source. Uniform by default.
    b: the weights of the target. Uniform by default.
    objective: Optional[str], 'linear', 'quadratic', 'fused' or None. None
      means that the objective will be chosen based on the dimensionalities
      of the arrays.
    **kwargs: the keyword arguments passed to the point clouds and/or the
      solvers.

  Returns:
    A Transport object.
  """
  tau_a, tau_b = kwargs.get('tau_a', 1.0), kwargs.get('tau_b', 1.0)
  gw_unbalanced_correction = kwargs.pop('gw_unbalanced_correction', True)
  fused_penalty = kwargs.pop('fused_penalty', None)
  eps_keys = ['epsilon', 'init', 'target', 'decay']
  pb_kwargs = {k: v for k, v in kwargs.items() if k in eps_keys}
  pb = quad_problems.make(
      *args,
      objective=objective,
      a=a,
      b=b,
      tau_a=tau_a,
      tau_b=tau_b,
      gw_unbalanced_correction=gw_unbalanced_correction,
      fused_penalty=fused_penalty,
      **pb_kwargs)
  linear = isinstance(pb, problems.LinearProblem)
  solver_fn = sinkhorn.make if linear else gromov_wasserstein.make
  geom_keys = ['cost_fn', 'power', 'online']
  remove_keys = geom_keys + eps_keys if linear else geom_keys
  for key in remove_keys:
    kwargs.pop(key, None)
  solver = solver_fn(**kwargs)
  output = solver(pb)
  return Transport(pb, output)
