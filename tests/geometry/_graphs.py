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
"""Random graphs shared by the graph and geodesic geometry tests."""
import functools
from typing import Optional, Union

import networkx as nx
from networkx.algorithms import shortest_paths
from networkx.generators import random_graphs

import jax.experimental.sparse as jesp
import jax.numpy as jnp
import numpy as np

from ott.geometry import geometry

__all__ = ["random_graph", "gt_geometry"]


@functools.lru_cache(maxsize=None)
def random_graph(
    n: int,
    p: float = 0.3,
    seed: Optional[int] = 0,
    is_sparse: bool = False,
    *,
    return_laplacian: bool = False,
    directed: bool = False,
) -> jnp.ndarray:
  """Sample a connected random graph with uniformly weighted edges."""
  G = random_graphs.fast_gnp_random_graph(n, p, seed=seed, directed=directed)
  if not directed:
    assert nx.is_connected(G), "Generated graph is not connected."

  rng = np.random.RandomState(seed)
  for _, _, w in G.edges(data=True):
    w["weight"] = rng.uniform(0, 10)

  G = nx.linalg.laplacian_matrix(
      G
  ) if return_laplacian else nx.linalg.adjacency_matrix(G)

  if is_sparse:
    return jesp.BCOO.from_scipy_sparse(G)
  return jnp.asarray(G.toarray())


def gt_geometry(
    G: Union[jnp.ndarray, nx.Graph],
    *,
    epsilon: float = 1e-2
) -> geometry.Geometry:
  """Geometry whose cost is the squared shortest-path distance on ``G``."""
  if not isinstance(G, nx.Graph):
    G = nx.from_numpy_array(np.asarray(G))

  n = len(G)
  cost = np.zeros((n, n))

  path = dict(
      shortest_paths.all_pairs_bellman_ford_path_length(G, weight="weight")
  )
  for i, src in enumerate(G.nodes):
    for j, tgt in enumerate(G.nodes):
      cost[i, j] = path[src][tgt] ** 2

  cost = jnp.asarray(cost)
  kernel = jnp.asarray(np.exp(-cost / epsilon))
  return geometry.Geometry(cost_matrix=cost, kernel_matrix=kernel, epsilon=1.0)
