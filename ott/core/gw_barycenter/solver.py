from functools import partial
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ott.core import fixed_point_loop, gromov_wasserstein
from ott.core.gw_barycenter.problem import GromovWassersteinBarycenterProblem
from ott.geometry import geometry, low_rank


class GWBarycenterState(NamedTuple):
  x: Optional[geometry.Geometry] = None
  a: Optional[jnp.ndarray] = None
  converged: bool = False
  errors: Optional[jnp.ndarray] = None
  costs: Optional[jnp.ndarray] = None
  reg_gw_cost: Optional[float] = None


@jax.tree_util.register_pytree_node_class
class GromovWassersteinBarycenter:

  # TODO(michalk8): consider subclassing + calling parent in `update_state`
  def __init__(self, **kwargs: Any):
    self._quad_solver = gromov_wasserstein.GromovWasserstein(**kwargs)
    self._kwargs = kwargs
    assert not self._quad_solver.is_low_rank, "Low rank not yet implemented."

  def __call__(
      self, problem: GromovWassersteinBarycenterProblem, **kwargs: Any
  ):
    bar_fn = jax.jit(
        iterations, static_argnums=1
    ) if self._quad_solver.jit else iterations
    state = self.init_state(problem, **kwargs)
    state = bar_fn(solver=self, problem=problem, init_state=state)  # TODO
    out = self.output_from_state(state)
    return out

  def init_state(
      self,
      problem: GromovWassersteinBarycenterProblem,
      *,
      bar_init: Union[int, geometry.Geometry],
      a: Optional[jnp.ndarray] = None,
  ) -> GWBarycenterState:
    bar_size = bar_init if isinstance(bar_init, int) else bar_init.shape[0]

    if a is None:
      a = jnp.ones((bar_size,)) / bar_size
    if not isinstance(bar_init, geometry.Geometry):
      bar_init = low_rank.LRCGeometry(cost_1=a[:, None], cost_2=a[:, None])

    assert a.shape == (bar_size,), (a.shape, (bar_size,))
    assert a.shape == (bar_init.shape[0],), (a.shape, bar_init.shape[0])

    num_iter = self._quad_solver.max_iterations
    if self._quad_solver.store_inner_errors:
      errors = -jnp.ones((
          num_iter, problem.size,
          self._quad_solver.linear_ot_solver.outer_iterations
      ))
    else:
      errors = None

    costs = -jnp.ones((num_iter,))
    return GWBarycenterState(x=bar_init, a=a, errors=errors, costs=costs)

  def update_state(
      self, state: GWBarycenterState, iteration: int,
      problem: GromovWassersteinBarycenterProblem
  ) -> GWBarycenterState:
    from ott.core import gromov_wasserstein, quad_problems

    @partial(jax.vmap, in_axes=[None, 0])
    def solve_gw(
        bar: geometry.Geometry, geom_x: geometry.Geometry
    ) -> gromov_wasserstein.GWOutput:
      quad_prob = quad_problems.QuadraticProblem(
          geom_xx=bar, geom_xy=geom_x, a=None, b=None
      )
      sol = self._quad_solver(quad_prob)
      return sol

    return state

  def output_from_state(self, state: GWBarycenterState) -> GWBarycenterState:
    # TODO(michalk8)
    return state

  def tree_flatten(self) -> Tuple[Sequence[Any], Mapping[str, Any]]:
    return [], self._kwargs

  @classmethod
  def tree_unflatten(
      cls, aux_data: Mapping[str, Any], children: Sequence[Any]
  ) -> "GromovWassersteinBarycenter":
    del children
    return cls(**aux_data)


def iterations(
    solver: GromovWassersteinBarycenter,
    problem: GromovWassersteinBarycenterProblem, init_state: GWBarycenterState
) -> GWBarycenterState:

  def cond_fn(
      iteration: int, constants: GromovWassersteinBarycenter,
      state: GWBarycenterState
  ) -> bool:
    solver, _ = constants
    return solver._quad_solver._continue(state, iteration)

  def body_fn(
      iteration, constants: Tuple[GromovWassersteinBarycenter,
                                  GromovWassersteinBarycenterProblem],
      state: GWBarycenterState, compute_error: bool
  ) -> GWBarycenterState:
    del compute_error  # always assumed true
    solver, problem = constants
    return solver.update_state(state, iteration, problem)

  state = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver._quad_solver.min_iterations,
      max_iterations=solver._quad_solver.max_iterations,
      inner_iterations=1,
      constants=(solver, problem),
      state=init_state,
  )
  return state
