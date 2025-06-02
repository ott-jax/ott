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
from typing import NamedTuple, Optional, Tuple

import jax.experimental.sparse as jesp
import jax.numpy as jnp

from optax import assignment

from ott.geometry import costs, geometry, pointcloud

__all__ = ["hungarian"]


class HungarianOutput(NamedTuple):
  r"""Output of the Hungarian solver.

  Args:
    geom: geometry object
    paired_indices: Array of shape ``[2, n]``, of :math:`n` pairs
      of indices, for which the optimal transport assigns mass. Namely, for each
      index :math:`0 \leq k < n`, if one has
      :math:`i := \text{paired_indices}[0, k]` and
      :math:`j := \text{paired_indices}[1, k]`, then point :math:`i` in
      the first geometry sends mass to point :math:`j` in the second.
  """
  geom: geometry.Geometry
  paired_indices: Optional[jnp.ndarray] = None

  @property
  def matrix(self) -> jesp.BCOO:
    """``[n, n]`` transport matrix  in sparse format, with ``n`` NNZ entries."""
    n, _ = self.geom.shape
    unit_mass = jnp.ones((n,)) / n
    indices = self.paired_indices.swapaxes(0, 1)
    return jesp.BCOO((unit_mass, indices), shape=(n, n))


def hungarian(geom: geometry.Geometry) -> Tuple[jnp.ndarray, HungarianOutput]:
  """Solve matching problem using :term:`Hungarian algorithm` from :mod:`optax`.

  Args:
    geom: Geometry object with square (shape ``[n,n]``)
      :attr:`~ott.geometry.geometry.Geomgetry.cost matrix`.

  Returns:
    The value of the unregularized OT problem, along with an output
    object listing relevant information on outputs.
  """
  n, m = geom.shape
  assert n == m, f"Hungarian can only match same # of points, got {n} and {m}."
  i, j = assignment.hungarian_algorithm(geom.cost_matrix)

  hungarian_out = HungarianOutput(geom=geom, paired_indices=jnp.stack((i, j)))
  return jnp.sum(geom.cost_matrix[i, j]) / n, hungarian_out


def wassdis_p(x: jnp.ndarray, y: jnp.ndarray, *, p: float = 2.0) -> float:
  """Compute the :term:`Wasserstein distance`, uses :term:`Hungarian algorithm`.

  Uses :func:`hungarian` to solve the :term:`optimal matching problem` between
  two point clouds of the same size, to compute a :term:`Wasserstein distance`
  estimator.

  Note:
    At the moment, only supports point clouds of the same size to be easily
    cast as an optimal matching problem.

  Args:
    x: ``[n,d]`` point cloud
    y: ``[n,d]`` point cloud of the same size
    p: order of the Wasserstein distance, non-negative float.

  Returns:
    The `p`-Wasserstein distance between these point clouds.
  """
  geom = pointcloud.PointCloud(x, y, cost_fn=costs.EuclideanP(p=p))
  cost, _ = hungarian(geom)
  return cost ** (1. / p)
