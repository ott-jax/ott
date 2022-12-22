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

import functools
from collections import namedtuple
from dataclasses import dataclass

from jaxopt import LBFGS

ConjugateResults = namedtuple("ConjugateResults", "val grad num_iter")


@dataclass
class ConjugateSolverLBFGS:
  gtol: float = 1e-3
  max_iter: int = 10
  max_linesearch_iter: int = 10
  linesearch_type: str = 'backtracking'
  decrease_factor: float = 2. / 3.
  ls_method: str = 'strong-wolfe'

  def conj_min_obj(self, x, f, y):
    # f^*(y) = -inf_x f(x) - y^T x
    return f(x) - x.dot(y)

  def solve(self, f, y, x_init=None):
    assert y.ndim == 1

    if x_init is None:
      x_init = y

    conj_min_obj = functools.partial(self.conj_min_obj, y=y, f=f)
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
