from typing import Optional, Union

import jax.experimental.sparse as jesp
import jax.numpy as jnp
import networkx as nx
import pytest
from networkx.generators import random_graphs
from typing_extensions import Literal

from ott.core import decomposition

sksparse = pytest.importorskip("sksparse")


def random_graph(
    n: int,
    p: float = 0.05,
    seed: Optional[int] = None,
    return_laplacian: bool = False,
    directed: bool = False,
    format: Optional[Literal["csr", "csc", "coo"]] = None
) -> Union[jnp.ndarray, jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]:
  graph = random_graphs.fast_gnp_random_graph(
      n, p, seed=seed, directed=directed
  )

  if return_laplacian:
    graph = nx.linalg.laplacianmatrix(graph)
  else:
    graph = nx.linalg.adjacency_matrix(graph)

  if format is None:
    return jnp.asarray(graph.A)

  graph = getattr(graph, f"to{format}")()
  return decomposition._scipy_sparse_to_jax(graph)


class TestGraph:

  def test_pytree(self):
    pass

  def test_init_graph(self):
    pass

  def test_init_laplacian(self):
    pass

  def test_numerical_scheme(self):
    pass

  def test_sparse_formats(self):
    pass

  def test_sparse_correctness(self):
    pass

  def test_sparse_memory_efficiency(self):
    pass
