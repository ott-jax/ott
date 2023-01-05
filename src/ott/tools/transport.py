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

Even if the `transport.solve` sole function can support many complex use cases,
we suggest more advanced users to instantiate directly their :mod:`ott.problems`
and their :mod:`ott.solvers` for better control over the parameters.
"""

from typing import Any, NamedTuple, Optional, Union

import jax.numpy as jnp
import numpy as np
from typing_extensions import Literal

from ott import utils
from ott.geometry import geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

__all__ = ["Transport"]


class Transport(NamedTuple):
  """Transport interface to transport solutions."""

  problem: Any = None
  solver_output: Any = None

  @property
  def linear(self) -> bool:
    return isinstance(self.problem, linear_problem.LinearProblem)

  @property
  def geom(self) -> geometry.Geometry:
    return self.problem.geom if self.linear else self.solver_output.geom

  @property
  def a(self) -> jnp.ndarray:
    return self.problem.a

  @property
  def b(self) -> jnp.ndarray:
    return self.problem.b

  @property
  def linear_output(self):
    out = self.solver_output
    return out if self.linear else out.linear_state

  @property
  def matrix(self) -> jnp.ndarray:
    return self.linear_output.matrix

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    return self.linear_output.apply(inputs, axis)

  def marginal(self, axis: int = 0) -> jnp.ndarray:
    return self.linear_output.marginal(axis)


@utils.deprecate(version="0.3.2")
def solve(
    *args: Any,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    init_dual_a: Optional[jnp.ndarray] = None,
    init_dual_b: Optional[jnp.ndarray] = None,
    objective: Optional[Literal['linear', 'quadratic', 'fused']] = None,
    **kwargs: Any
) -> Transport:
  """Generic interface to transport problem.

  .. note:

    This function has been deprecated and will be removed in ``0.3.2`` release.
    Please use :mod:`ott.solvers` module directly.

  The geometries can be passed as arrays, geometry.Geometry or directly as a
  problem. The solver is passed via kwargs.

  Args:
    args: can be either a single argument, the geometry.Geometry instance, or
      for convenience only two jnp.ndarray<float> corresponding to two point
      clouds. In that case the regularization parameter epsilon must be set in
      the kwargs.
    a: the weights of the source. Uniform by default.
    b: the weights of the target. Uniform by default.
    objective: Optional[str], 'linear', 'quadratic', 'fused' or None. None
      means that the objective will be chosen based on the dimensionalities
      of the arrays.
    kwargs: the keyword arguments passed to the point clouds and/or the
      solvers.

  Returns:
    A Transport object.
  """  # noqa: D401
  tau_a, tau_b = kwargs.get('tau_a', 1.0), kwargs.get('tau_b', 1.0)
  gw_unbalanced_correction = kwargs.pop('gw_unbalanced_correction', True)
  fused_penalty = kwargs.pop('fused_penalty', None)
  eps_keys = ['epsilon', 'init', 'target', 'decay']
  pb_kwargs = {k: v for k, v in kwargs.items() if k in eps_keys}
  pb = make(
      *args,
      objective=objective,
      a=a,
      b=b,
      tau_a=tau_a,
      tau_b=tau_b,
      gw_unbalanced_correction=gw_unbalanced_correction,
      fused_penalty=fused_penalty,
      **pb_kwargs
  )
  linear = isinstance(pb, linear_problem.LinearProblem)
  solver_fn = sinkhorn.Sinkhorn if linear else gromov_wasserstein.GromovWasserstein
  geom_keys = ['cost_fn', 'online']

  remove_keys = geom_keys + eps_keys if linear else geom_keys
  for key in remove_keys:
    kwargs.pop(key, None)
  solver = solver_fn(**kwargs)
  output = solver(pb, (init_dual_a, init_dual_b))
  return Transport(pb, output)


def make(
    *args: Union[jnp.ndarray, geometry.Geometry, linear_problem.LinearProblem,
                 quadratic_problem.QuadraticProblem],
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    objective: Optional[str] = None,
    gw_unbalanced_correction: Optional[bool] = True,
    fused_penalty: Optional[float] = None,
    scale_cost: Optional[Union[bool, float, str]] = False,
    **kwargs: Any,
):
  """Make a problem from arrays, assuming PointCloud geometries."""
  if isinstance(args[0], (jnp.ndarray, np.ndarray)):
    x = args[0]
    y = args[1] if len(args) > 1 else args[0]
    if ((objective == 'linear') or
        (objective is None and x.shape[1] == y.shape[1])):  # noqa: E129
      geom_xy = pointcloud.PointCloud(x, y, **kwargs)
      return linear_problem.LinearProblem(
          geom_xy, a=a, b=b, tau_a=tau_a, tau_b=tau_b
      )
    elif ((objective == 'quadratic') or
          (objective is None and x.shape[1] != y.shape[1])):
      geom_xx = pointcloud.PointCloud(x, x, **kwargs)
      geom_yy = pointcloud.PointCloud(y, y, **kwargs)
      return quadratic_problem.QuadraticProblem(
          geom_xx=geom_xx,
          geom_yy=geom_yy,
          geom_xy=None,
          scale_cost=scale_cost,
          a=a,
          b=b,
          tau_a=tau_a,
          tau_b=tau_b,
          gw_unbalanced_correction=gw_unbalanced_correction
      )
    elif objective == 'fused':
      geom_xx = pointcloud.PointCloud(x, x, **kwargs)
      geom_yy = pointcloud.PointCloud(y, y, **kwargs)
      geom_xy = pointcloud.PointCloud(x, y, **kwargs)
      return quadratic_problem.QuadraticProblem(
          geom_xx=geom_xx,
          geom_yy=geom_yy,
          geom_xy=geom_xy,
          fused_penalty=fused_penalty,
          scale_cost=scale_cost,
          a=a,
          b=b,
          tau_a=tau_a,
          tau_b=tau_b,
          gw_unbalanced_correction=gw_unbalanced_correction
      )
    else:
      raise ValueError(f'Unknown transport problem `{objective}`')
  elif isinstance(args[0], geometry.Geometry):
    if len(args) == 1:
      return linear_problem.LinearProblem(
          *args, a=a, b=b, tau_a=tau_a, tau_b=tau_b
      )
    return quadratic_problem.QuadraticProblem(
        *args, a=a, b=b, tau_a=tau_a, tau_b=tau_b, scale_cost=scale_cost
    )
  elif isinstance(
      args[0],
      (linear_problem.LinearProblem, quadratic_problem.QuadraticProblem)
  ):
    return args[0]
  else:
    raise ValueError('Cannot instantiate a transport problem.')
