from typing import Optional, Union

import jax.experimental.sparse as jesp
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from networkx.generators import random_graphs
from typing_extensions import Literal

from ott.core import decomposition
from ott.geometry import graph

sksparse = pytest.importorskip("sksparse")


def random_graph(
    n: int,
    p: float = 0.05,
    seed: Optional[int] = 0,
    *,
    return_laplacian: bool = False,
    directed: bool = False,
    fmt: Optional[Literal["csr", "csc", "coo"]] = None
) -> Union[jnp.ndarray, jesp.CSR, jesp.CSC, jesp.COO, jesp.BCOO]:
  graph = random_graphs.fast_gnp_random_graph(
      n, p, seed=seed, directed=directed
  )

  if return_laplacian:
    graph = nx.linalg.laplacian_matrix(graph)
  else:
    graph = nx.linalg.adjacency_matrix(graph)

  if fmt is None:
    return jnp.asarray(graph.A)

  graph = getattr(graph, f"to{fmt}")()
  return decomposition._scipy_sparse_to_jax(graph)


class TestGraph:

  @pytest.mark.parametrize("empty", [False, True])
  def test_invalid_initialization(self, empty):
    with pytest.raises(AssertionError, match="Please provide the graph"):
      if empty:
        _ = graph.Graph(graph=None, laplacian=None)
      else:
        G = random_graph(100, 0.1)
        _ = graph.Graph(graph=G, laplacian=G)

  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_init_graph(self, fmt: Optional[str]):
    n, p = 100, 0.1
    G = random_graph(n, p, fmt=fmt)

    geom = graph.Graph(G)

    assert geom.shape == (n, n)
    assert geom.graph is G
    assert geom._laplacian is None
    # compute the laplacian on the fly
    assert isinstance(geom.laplacian, type(geom.graph))

  @pytest.mark.parametrize("fmt", [None, "csr", "csc", "coo"])
  def test_init_laplacian(self, fmt: Optional[str]):
    n, p = 20, 0.05
    L = random_graph(n, p, return_laplacian=True, fmt=fmt)

    geom = graph.Graph(laplacian=L)

    assert geom.shape == (n, n)
    assert geom.laplacian is L
    assert geom.graph is None

  @pytest.mark.parametrize("as_laplacian", [False, True])
  @pytest.mark.parametrize("fmt", [None, "coo"])
  def test_pytree(self, fmt: Optional[str], as_laplacian: bool):
    G = random_graph(25, fmt=fmt, return_laplacian=as_laplacian)

    geom1 = graph.Graph(laplacian=G) if as_laplacian else graph.Graph(graph=G)
    children, aux_data = geom1.tree_flatten()
    geom2 = graph.Graph.tree_unflatten(aux_data, children)

    assert geom1.graph is geom2.graph
    if as_laplacian:
      assert geom1.laplacian is geom2.laplacian
    elif fmt is None:
      np.testing.assert_array_equal(geom1.laplacian, geom2.laplacian)
    else:
      lap1, lap2 = geom1.laplacian, geom2.laplacian
      np.testing.assert_array_equal(lap1.data, lap2.data)
      np.testing.assert_array_equal(lap1.indices, lap2.indices)

    assert geom1.epsilon == geom2.epsilon
    assert geom1.n_steps == geom2.n_steps
    assert geom1.numerical_scheme == geom2.numerical_scheme
    assert geom1.directed == geom2.directed
    assert geom1._solver == geom2._solver

  @pytest.mark.parametrize("fmt", [None, "csr", "csc", "coo"])
  def test_solver(self, fmt: Optional[str]):
    n = 27
    G = random_graph(n, fmt=fmt)
    geom1 = graph.Graph(laplacian=G)

    assert geom1._solver is None
    solver = geom1.solver
    assert geom1.solver is solver  # cached

    if geom1.is_sparse:
      assert isinstance(solver, decomposition.SparseCholeskySolver)
      L = solver.L  # trigger the computation
      assert L is None  # we're interested in the side-effect below
      assert hash(geom1) in decomposition.SparseCholeskySolver._FACTOR_CACHE
    else:
      assert isinstance(solver, decomposition.DenseCholeskySolver)
      np.testing.assert_array_equal(solver.A, geom1._M)
      assert isinstance(solver.L, jnp.ndarray)
      assert solver.L.shape == (n, n)

  def test_numerical_scheme(self):
    pass

  def test_sparse_formats(self):
    pass

  def test_sparse_correctness(self):
    pass

  def test_sparse_memory_efficiency(self):
    pass
