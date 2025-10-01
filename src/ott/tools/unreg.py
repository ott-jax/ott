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
from typing import Tuple

import jax.numpy as jnp

from optax import assignment

from ott.geometry import costs, geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import semidiscrete

__all__ = ["hungarian", "wassdis_p"]


def hungarian(
    geom: geometry.Geometry
) -> Tuple[jnp.ndarray, semidiscrete.HardAssignmentOutput]:
  """Solve matching problem using the :term:`Hungarian algorithm`.

  Uses the implementation from :mod:`optax`.

  Args:
    geom: Geometry object with square (shape ``[n, n]``)
      :attr:`~ott.geometry.geometry.Geometry.cost_matrix`.

  Returns:
    The value of the unregularized OT problem, along with an output
    object listing relevant information on outputs.
  """
  n, m = geom.shape
  assert n == m, f"Hungarian can only match same # of points, got {n} and {m}."
  cost_matrix = geom.cost_matrix
  i, j = assignment.hungarian_algorithm(cost_matrix)
  prob = linear_problem.LinearProblem(geom)
  out = semidiscrete.HardAssignmentOutput(
      prob, paired_indices=jnp.stack([i, j])
  )
  transport_cost = cost_matrix[i, j].sum() / n
  return transport_cost, out


def wassdis_p(x: jnp.ndarray, y: jnp.ndarray, *, p: float = 2.0) -> float:
  """Compute the :term:`Wasserstein distance`, uses :term:`Hungarian algorithm`.

  Uses :func:`hungarian` to solve the :term:`optimal matching problem` between
  two point clouds of the same size, to compute a :term:`Wasserstein distance`
  estimator.

  Note:
    At the moment, only supports point clouds of the same size to be easily
    cast as an optimal matching problem.

  Args:
    x: ``[n, d]`` point cloud.
    y: ``[n, d]`` point cloud of the same size.
    p: order of the Wasserstein distance, non-negative float.

  Returns:
    The `p`-Wasserstein distance between these point clouds.
  """
  geom = pointcloud.PointCloud(x, y, cost_fn=costs.EuclideanP(p=p))
  cost, _ = hungarian(geom)
  return cost ** (1.0 / p)
