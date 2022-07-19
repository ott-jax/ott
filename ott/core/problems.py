# Copyright 2022 The OTT Authors
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
"""Utility to make a problem class from arrays."""
from typing import Any, Optional, Union

import jax.numpy as jnp
import numpy as np

from ott.core import linear_problems, quad_problems
from ott.geometry import geometry, pointcloud


def make(
    *args: Union[jnp.ndarray, geometry.Geometry, linear_problems.LinearProblem,
                 quad_problems.QuadraticProblem],
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
      return linear_problems.LinearProblem(
          geom_xy, a=a, b=b, tau_a=tau_a, tau_b=tau_b
      )
    elif ((objective == 'quadratic') or
          (objective is None and x.shape[1] != y.shape[1])):
      geom_xx = pointcloud.PointCloud(x, x, **kwargs)
      geom_yy = pointcloud.PointCloud(y, y, **kwargs)
      return quad_problems.QuadraticProblem(
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
      return quad_problems.QuadraticProblem(
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
      return linear_problems.LinearProblem(
          *args, a=a, b=b, tau_a=tau_a, tau_b=tau_b
      )
    return quad_problems.QuadraticProblem(
        *args, a=a, b=b, tau_a=tau_a, tau_b=tau_b, scale_cost=scale_cost
    )
  elif isinstance(
      args[0], (linear_problems.LinearProblem, quad_problems.QuadraticProblem)
  ):
    return args[0]
  else:
    raise ValueError('Cannot instantiate a transport problem.')
