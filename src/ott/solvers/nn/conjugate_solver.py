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
"""Implementation of :cite:`amos:17` input convex neural networks (ICNN)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple, Optional

from jaxopt import LBFGS

import jax.numpy as jnp

from ott import utils


class ConjugateResults(NamedTuple):
  val: float
  grad: jnp.array
  num_iter: int


class ConjugateSolver(ABC):

  @abstractmethod
  def solve(
      self,
      f: Callable,
      y: jnp.ndarray,
      x_init: Optional[jnp.array] = None
  ) -> ConjugateResults:
    pass


@dataclass
@utils.register_pytree_node
class FenchelConjugateLBFGS(ConjugateSolver):
  gtol: float = 1e-3
  max_iter: int = 10
  max_linesearch_iter: int = 10
  linesearch_type: Literal['zoom', 'backtracking'] = 'backtracking'
  decrease_factor: float = 2. / 3.
  ls_method: Literal['wolf', 'strong-wolfe'] = 'strong-wolfe'

  def solve(
      self,
      f: Callable,
      y: jnp.ndarray,
      x_init: Optional[jnp.array] = None
  ) -> ConjugateResults:
    assert y.ndim == 1

    if x_init is None:
      x_init = y

    conj_min_obj = lambda x: f(x) - x.dot(y)
    solver = LBFGS(
        fun=conj_min_obj,
        tol=self.gtol,
        maxiter=self.max_iter,
        decrease_factor=self.decrease_factor,
        linesearch=self.linesearch_type,
        condition=self.ls_method,
        implicit_diff=False,
        unroll=False
    )

    out = solver.run(x_init)
    return ConjugateResults(
        val=-out.state.value, grad=out.params, num_iter=out.state.iter_num
    )
