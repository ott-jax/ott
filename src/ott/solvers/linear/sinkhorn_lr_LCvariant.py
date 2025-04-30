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
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ott.geometry import geometry
from ott.initializers.linear import initializers_lr
from ott.math import fixed_point_loop
from ott.math import utils as mu
from ott.problems.linear import linear_problem
from ott.solvers.linear import lr_utils, sinkhorn

__all__ = ["LR_LC_Sinkhorn", "LR_LC_SinkhornOutput"]


ProgressFunction = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "LRSinkhornState"]], None]

class LR_LC_SinkhornOutput(NamedTuple):
    """Transport interface for a low-rank latent-coupling Sinkhorn solution."""

class LR_LC_Sinkhorn(sinkhorn.Sinkhorn):
  """Low-rank latent-coupling Sinkhorn solver.

  TODO : end
  """

  def __init__(
    self,
    rank: int,
    gamma: float = 10.0,
    gamma_rescale: bool = True,
    epsilon: float = 0.0,
    initializer: Optional[initializers_lr.LRInitializer] = None,
    lse_mode: bool = True,
    inner_iterations: int = 10,
    use_danskin: bool = True,
    kwargs_dys: Optional[Mapping[str, Any]] = None,
    progress_fn: Optional[ProgressFunction] = None,
    **kwargs: Any,
  ):
    kwargs["implicit_diff"] = None  # not yet implemented
    super().__init__(
        lse_mode=lse_mode,
        inner_iterations=inner_iterations,
        use_danskin=use_danskin,
        **kwargs
    )
    self.rank = rank
    self.gamma = gamma
    self.gamma_rescale = gamma_rescale
    self.epsilon = epsilon
    self.initializer = initializers_lr.RandomInitializer(
        rank
    ) if initializer is None else initializer
    self.progress_fn = progress_fn
    self.kwargs_dys = {} if kwargs_dys is None else kwargs_dys
  
  def __call__(
      self,
      ot_prob: linear_problem.LinearProblem,
      init: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = None,
      **kwargs: Any,
  ) -> LR_LC_SinkhornOutput:
    """Run low-rank Sinkhorn.

    Args:
      ot_prob: Linear OT problem.
      init: Initial values for the low-rank factors:

        - :attr:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornOutput.q`.
        - :attr:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornOutput.r`.
        - :attr:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornOutput.g`.

        If :obj:`None`, run the initializer.
      kwargs: Keyword arguments for the initializer.

    Returns:
      The low-rank Sinkhorn output.
    """
    if init is None:
      init = self.initializer(ot_prob, **kwargs)
    return run(ot_prob, self, init)
  
def run(
    ot_prob: linear_problem.LinearProblem,
    solver: LR_LC_Sinkhorn,
    init: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> LR_LC_SinkhornOutput:
  """Run loop of the solver, outputting a state upgraded to an output."""
  out = sinkhorn.iterations(ot_prob, solver, init)
  out = out.set_cost(
      ot_prob, lse_mode=solver.lse_mode, use_danskin=solver.use_danskin
  )
  return out.set(ot_prob=ot_prob)