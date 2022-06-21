from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from ott.core import gromov_wasserstein
from ott.core.gw_barycenter.problem import GromovWassersteinBarycenterProblem
from ott.geometry import geometry


class GWBarycenterState(NamedTuple):
  x: Optional[geometry.Geometry] = None
  a: Optional[jnp.ndarray] = None
  converged: bool = False
  errors: Optional[jnp.ndarray] = None
  reg_gw_cost: Optional[float] = None


@jax.tree_util.register_pytree_node_class
class GromovWassersteinBarycenter(gromov_wasserstein.GromovWasserstein):

  def __call__(
      self, problem: GromovWassersteinBarycenterProblem, **kwargs: Any
  ):
    bar_fn = jax.jit(iterations, static_argnums=1) if self.jit else iterations
    state = self.init_state(problem, **kwargs)
    state = bar_fn(solver=self, problem=problem, init_state=state)  # TODO
    out = self.output_from_state(state)
    return out

  def init_state(
      self,
      problem: GromovWassersteinBarycenterProblem,
      *,
      bar_init: Union[int, geometry.Geometry],
      num_iter: int,
      a: Optional[jnp.ndarray] = None,
  ) -> GWBarycenterState:
    bar_size = bar_init if isinstance(bar_init, int) else bar_init.shape[0]

    if a is None:
      a = jnp.ones((bar_size,)) / bar_size
    if not isinstance(bar_init, geometry.Geometry):
      bar_init = geometry.Geometry(cost_matrix=a[:, None] * a[None, :])

    assert a.shape == (bar_size,), (a.shape, (bar_size,))
    assert a.shape == (bar_init.shape[0],), (a.shape, bar_init.shape[0])

    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones(
          (num_iter, problem.size, self.linear_ot_solver.outer_iterations)
      )
    else:
      errors = None

    return GWBarycenterState(x=bar_init, a=a, errors=errors)

  def update_state(self, state: GWBarycenterState) -> GWBarycenterState:
    return state

  def output_from_state(self, state: GWBarycenterState) -> GWBarycenterState:
    # TODO(michalk8)
    return state


def iterations(
    solver: GromovWassersteinBarycenter,
    problem: GromovWassersteinBarycenterProblem, init_state: GWBarycenterState
) -> GWBarycenterState:
  return init_state
