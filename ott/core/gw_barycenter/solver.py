from functools import partial
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ott.core import fixed_point_loop, gromov_wasserstein
from ott.core.gw_barycenter.problem import GromovWassersteinBarycenterProblem
from ott.geometry import geometry


class GWBarycenterState(NamedTuple):
  """TODO.

  Attributes:
    x: TODO.
    a: TODO.
    converged: TODO.
    errors: TODO.
    costs: TODO.
    reg_gw_cost: TODO.
  """
  x: Optional[geometry.Geometry] = None
  a: Optional[jnp.ndarray] = None
  converged: bool = False
  errors: Optional[jnp.ndarray] = None
  costs: Optional[jnp.ndarray] = None
  reg_gw_cost: float = -1

  def set(self, **kwargs: Any) -> 'GWBarycenterState':
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)


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
    return self.output_from_state(state)

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
    if isinstance(bar_init, int):
      # TODO(michalk8): sample from each measure
      raise NotImplementedError(bar_init)
    elif not isinstance(bar_init, geometry.Geometry):
      # TODO(michalk8): think about low rank, also fails if PC is passed
      bar_init = geometry.Geometry(
          cost=a[:, None] * a[:, None], epsilon=problem.epsilon
      )

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
      self,
      state: GWBarycenterState,
      iteration: int,
      problem: GromovWassersteinBarycenterProblem,
      store_errors: bool = True,
  ) -> GWBarycenterState:
    from ott.core import gromov_wasserstein, quad_problems

    # TODO(michalk8): make sure geometries are padded to the same shape
    # TODO(michalk8): think about low rank
    @partial(jax.vmap, in_axes=[None, None, 0, 0])
    def solve_gw(
        a: jnp.ndarray,
        bar: geometry.Geometry,
        b: jnp.ndarray,
        cost: jnp.ndarray,
    ) -> gromov_wasserstein.GWOutput:
      assert isinstance(cost, jnp.ndarray), cost
      geom = geometry.Geometry(cost, epsilon=problem.epsilon)
      quad_problem = quad_problems.QuadraticProblem(
          geom_xx=bar, geom_yy=geom, a=a, b=b
      )
      out = self._quad_solver(quad_problem)
      return (
          out.reg_gw_cost, out.convergence, out.matrix,
          out.errors if store_errors else None
      )

    costs, convs, transports, errors = solve_gw(
        state.a, state.x, problem.b, problem.geometries
    )

    cost = jnp.sum(costs * problem.weights)
    costs = state.costs.at[iteration].set(cost)

    x_new = compute_baycenter(problem, transports, state.a)

    return state.set(x=x_new, costs=costs)

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


def compute_baycenter(
    problem: GromovWassersteinBarycenterProblem,
    transports: jnp.ndarray,
    a: jnp.ndarray,
) -> geometry.Geometry:
  """TODO.

  Args:
    problem: the GW barycenter problem.
    transports: (num_measures, )
    a: barycenter weights.
  """

  @partial(jax.vmap, in_axes=[0, 0, None])
  def project(cost: jnp.ndarray, transport: jnp.ndarray, fn) -> jnp.ndarray:
    # TODO(michalk8): use geometries
    print(cost.shape, transport.shape)
    return transport @ (fn(cost) @ transport.T)

  scale = 1.0 / jnp.vdot(a, a)
  weights = problem.weights[:, None, None]

  h2 = problem.loss[1][1]
  barycenter = jnp.sum(
      weights * project(problem.geometries, transports, h2), axis=0
  )

  if problem._loss == 'sqeucl':
    # divide by `2` to adjust for the scale in `h2`
    barycenter = barycenter * (scale / 2.0)
  elif problem._loss == 'kl':
    barycenter = jnp.exp(barycenter * scale)
  else:
    raise NotImplementedError(problem._loss)

  return geometry.Geometry(cost_matrix=barycenter)


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
