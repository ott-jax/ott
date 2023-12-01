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
import abc
from typing import Callable, Literal, NamedTuple, Optional

import jax.numpy as jnp
from jaxopt import LBFGS

from ott import utils

__all__ = [
    "ConjugateResults",
    "FenchelConjugateSolver",
    "FenchelConjugateLBFGS",
    "DEFAULT_CONJUGATE_SOLVER",
]


class ConjugateResults(NamedTuple):
  r"""Holds the results of numerically conjugating a function.

  Args:
    val: the conjugate value, i.e., :math:`f^\star(y)`
    grad: the gradient, i.e., :math:`\nabla f^\star(y)`
    num_iter: the number of iterations taken by the solver
  """
  val: float
  grad: jnp.ndarray
  num_iter: int


class FenchelConjugateSolver(abc.ABC):
  r"""Abstract conjugate solver class.

  Given a function :math:`f`, numerically estimate the Fenchel conjugate
  :math:`f^\star(y) := -\inf_{x\in\mathbb{R}^n} f(x)-\langle x, y\rangle`.
  """

  @abc.abstractmethod
  def solve(
      self,
      f: Callable[[jnp.ndarray], jnp.ndarray],
      y: jnp.ndarray,
      x_init: Optional[jnp.ndarray] = None
  ) -> ConjugateResults:
    """Solve for the conjugate.

    Args:
      f: function to conjugate
      y: point to conjugate
      x_init: initial point to search over

    Returns:
      The solution to the conjugation.
    """


@utils.register_pytree_node
class FenchelConjugateLBFGS(FenchelConjugateSolver):
  """Solve for the conjugate using :class:`~jaxopt.LBFGS`.

  Args:
    gtol: gradient tolerance
    max_iter: maximum number of iterations
    max_linesearch_iter: maximum number of line search iterations
    linesearch_type: type of line search
    linesearch_init: strategy for line search initialization
    increase_factor: factor by which to increase the step size during
      the line search
  """

  gtol: float = 1e-3
  max_iter: int = 10
  max_linesearch_iter: int = 10
  linesearch_type: Literal["zoom", "backtracking",
                           "hager-zhang"] = "backtracking"
  linesearch_init: Literal["increase", "max", "current"] = "increase"
  increase_factor: float = 1.5

  def solve(  # noqa: D102
      self,
      f: Callable[[jnp.ndarray], jnp.ndarray],
      y: jnp.ndarray,
      x_init: Optional[jnp.array] = None
  ) -> ConjugateResults:
    assert y.ndim == 1, y.ndim

    solver = LBFGS(
        fun=lambda x: f(x) - x.dot(y),
        tol=self.gtol,
        maxiter=self.max_iter,
        linesearch=self.linesearch_type,
        linesearch_init=self.linesearch_init,
        increase_factor=self.increase_factor,
        implicit_diff=False,
        unroll=False
    )

    out = solver.run(y if x_init is None else x_init)
    return ConjugateResults(
        val=-out.state.value, grad=out.params, num_iter=out.state.iter_num
    )


DEFAULT_CONJUGATE_SOLVER = FenchelConjugateLBFGS(
    gtol=1e-5,
    max_iter=20,
    max_linesearch_iter=20,
    linesearch_type="backtracking",
)
