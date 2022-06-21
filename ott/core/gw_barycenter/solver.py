from typing import Any, NamedTuple

import jax

from ott.core import gromov_wasserstein
from ott.core.gw_barycenter.problem import GromovWassersteinBarycenterProblem


class GWBarycenterState(NamedTuple):
  pass


@jax.tree_util.register_pytree_node_class
class GromovWassersteinBarycenter(gromov_wasserstein.GromovWasserstein):

  def __call__(
      self, problem: GromovWassersteinBarycenterProblem, **kwargs: Any
  ):
    bar_fn = jax.jit(iterations, static_argnums=1) if self.jit else iterations
    state = self.init_state(problem)
    state = bar_fn(solver=self, problem=problem, state=state)  # TODO
    out = self.output_from_state(state)
    return out

  def init_state(
      self,
      problem: GromovWassersteinBarycenterProblem,
      **kwargs: Any,
  ) -> GWBarycenterState:
    pass

  def update_state(self, state: GWBarycenterState) -> GWBarycenterState:
    pass

  def output_from_state(self, state: GWBarycenterState) -> GWBarycenterState:
    # TODO(michalk8)
    return state


def iterations(
    solver: GromovWassersteinBarycenter,
    problem: GromovWassersteinBarycenterProblem, state: GWBarycenterState
) -> GWBarycenterState:
  return state
