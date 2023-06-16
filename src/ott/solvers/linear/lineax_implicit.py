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
from typing import Any, Callable, Optional, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from jaxtyping import Array, Float, PyTree

_T = TypeVar("_T")
_FlatPyTree = tuple[list[_T], jtu.PyTreeDef]

__all__ = ["CustomTransposeLinearOperator"]


class CustomTransposeLinearOperator(lx.FunctionLinearOperator):
  """Implement a linear operator that can specify its transpose directly."""
  fn: Callable[[PyTree[Float[Array, "..."]]], PyTree[Float[Array, "..."]]]
  fn_t: Callable[[PyTree[Float[Array, "..."]]], PyTree[Float[Array, "..."]]]
  input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.static_field()
  input_structure_t: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.static_field()
  tags: frozenset[object]

  def __init__(self, fn, fn_t, input_structure, input_structure_t, tags=()):
    super().__init__(fn, input_structure, tags)
    self.fn_t = eqx.filter_closure_convert(fn_t, input_structure_t)
    self.input_structure_t = input_structure_t

  def transpose(self):
    """Provide custom transposition operator from function."""
    return lx.FunctionLinearOperator(self.fn_t, self.input_structure_t)


def solve_lineax(
    lin: Callable,
    b: jnp.ndarray,
    lin_t: Optional[Callable] = None,
    symmetric: Optional[bool] = False,
    nonsym_solver: Optional[lx.AbstractLinearSolver] = None,
    **kwargs: Any
):
  """Wrapper around lineax solvers.

  Args:
    lin: Linear operator
    b: vector such that sought `x` is such that `lin(x)=b`
    lin_t: Linear operator, corresponding to transpose of `lin`.
    symmetric: whether `lin` is symmetric.
    nonsym_solver: `lineax` solver used when handling non symmetric cases. Note
      that `CG` is used by default in the symmetric case.
    kwargs: arguments passed to `lineax`'s linear solver.
  """
  input_structure = jax.eval_shape(lambda: b)
  kwargs.setdefault("rtol", 1e-6)
  kwargs.setdefault("atol", 1e-6)
  # Ridge parameters passed to JAX solvers are ignored in lineax.
  _ = kwargs.pop("ridge_identity", None)
  _ = kwargs.pop("ridge_kernel", None)

  if symmetric:
    solver = lx.CG(**kwargs)
    fn_operator = lx.FunctionLinearOperator(
        lin, input_structure, tags=lx.positive_semidefinite_tag
    )
    return lx.linear_solve(fn_operator, b, solver).value
  # In the nonsymmetric case, use NormalCG by default, but consider
  # user defined choice of alternative lx solver.
  solver_type = lx.NormalCG if nonsym_solver is None else nonsym_solver
  solver = solver_type(**kwargs)
  fn_operator = CustomTransposeLinearOperator(
      lin, lin_t, input_structure, input_structure
  )
  return lx.linear_solve(fn_operator, b, solver).value
