from functools import partial
from types import MappingProxyType
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
  x: Optional[jnp.ndarray] = None
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
    state = bar_fn(solver=self, problem=problem, init_state=state)
    return self.output_from_state(state)

  def init_state(
      self,
      problem: GromovWassersteinBarycenterProblem,
      *,
      bar_init: Union[int, jnp.ndarray],
      a: Optional[jnp.ndarray] = None,
  ) -> GWBarycenterState:
    bar_size = bar_init if isinstance(bar_init, int) else bar_init.shape[0]

    if a is None:
      a = jnp.ones((bar_size,)) / bar_size
    if isinstance(bar_init, int):
      # TODO(michalk8): initializer
      raise NotImplementedError(bar_init)
    # TODO(michalk8): think about low rank

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

    # TODO(michalk8): use point clouds instead of geometries
    # TODO(michalk8): think about low rank
    @partial(jax.vmap, in_axes=[None, None, 0, 0])
    def solve_gw(
        a: jnp.ndarray,
        bar: jnp.ndarray,
        b: jnp.ndarray,
        cost: jnp.ndarray,
    ) -> gromov_wasserstein.GWOutput:
      assert isinstance(cost, jnp.ndarray), cost
      bar = geometry.Geometry(bar, epsilon=problem.epsilon)
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
) -> jnp.ndarray:
  """TODO.

  Args:
    problem: the GW barycenter problem.
    transports: (num_measures, )
    a: barycenter weights.
  """

  @partial(jax.vmap, in_axes=[0, 0, None])
  def project(cost: jnp.ndarray, transport: jnp.ndarray, fn) -> jnp.ndarray:
    # TODO(michalk8): use geometries/outputs
    cost = cost if fn is None else fn(cost)
    return transport @ (cost @ transport.T)

  if problem._loss == 'sqeucl':
    fn = None
  elif problem._loss == 'kl':
    fn = problem.loss[1][1]  # log(x)
  else:
    raise NotImplementedError(problem._loss)

  barycenter = jnp.sum(
      problem.weights[:, None, None] *
      project(problem.geometries, transports, fn),
      axis=0
  ) / jnp.vdot(a, a)

  if problem._loss == 'kl':
    barycenter = jnp.exp(barycenter)
  return barycenter


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


def pad_along_axis(
    x: Sequence[jnp.ndarray],
    max_pad_size: Mapping[int, Optional[int]] = MappingProxyType({}),
    constant_values: Any = 0.0
) -> jnp.ndarray:
  """TODO.

  Args:
    x: sequence of arrays to pad.
    max_pad_size: maximum padding size along axis. Always pads after.
    constant_values: value to pad with.

  Returns:
    TODO.
  """
  shapes = jnp.asarray([arr.shape for arr in x])
  res = []

  for arr in x:
    pad_width = []
    for dim in range(arr.ndim):
      max_size = max_pad_size.get(dim, arr.shape[dim])
      if max_size is None:
        max_size = jnp.max(shapes[:, dim])
      pad_width.append((0, max_size - arr.shape[dim]))
    padded = jnp.pad(
        arr,
        pad_width=pad_width,
        mode='constant',
        constant_values=constant_values
    )
    res.append(padded)

  return jnp.asarray(res)
