from functools import partial
from types import MappingProxyType
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ott.core import (
    fixed_point_loop,
    gromov_wasserstein,
    problems,
    quad_problems,
    was_solver,
)
from ott.core.gw_barycenter.problem import GWBarycenterProblem
from ott.geometry import geometry, pointcloud


class GWBarycenterState(NamedTuple):
  """TODO.

  Attributes:
    x: Barycenter features. Only in fused case.
    c: Barycenter cost matrix.
    a: TODO.
    converged: TODO.
    errors: TODO.
    costs: TODO.
    reg_gw_cost: TODO.
  """
  x: Optional[jnp.ndarray] = None
  c: Optional[jnp.ndarray] = None
  a: Optional[jnp.ndarray] = None
  converged: bool = False
  errors: Optional[jnp.ndarray] = None
  costs: Optional[jnp.ndarray] = None
  reg_gw_cost: float = -1

  def set(self, **kwargs: Any) -> 'GWBarycenterState':
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)


@jax.tree_util.register_pytree_node_class
class GromovWassersteinBarycenter(was_solver.WassersteinSolver):

  def __init__(
      self,
      epsilon: Optional[float] = None,
      min_iterations: int = 5,
      max_iterations: int = 50,
      threshold: float = 1e-3,
      jit: bool = True,
      store_inner_errors: bool = False,
      gw_kwargs: Mapping[str, Any] = MappingProxyType({}),
  ):
    super().__init__(
        epsilon=epsilon,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        threshold=threshold,
        jit=jit,
        store_inner_errors=store_inner_errors
    )
    self._quad_solver = gromov_wasserstein.GromovWasserstein(**gw_kwargs)
    assert not self._quad_solver.is_low_rank, "Low rank not yet implemented."

  def __call__(self, problem: GWBarycenterProblem, **kwargs: Any):
    bar_fn = jax.jit(iterations, static_argnums=1) if self.jit else iterations
    state = self.init_state(problem, **kwargs)
    state = bar_fn(solver=self, problem=problem, init_state=state)
    return self.output_from_state(state)

  def init_state(
      self,
      problem: GWBarycenterProblem,
      bar_init: Union[int, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
      a: Optional[jnp.ndarray] = None,
      seed: int = 0,
  ) -> GWBarycenterState:
    """TODO.

    Args:
      problem: The barycenter problem.
      bar_init: TODO.
      a: Barycenter weights.
      seed: TODO.

    Returns:
      TODO.
    """
    if isinstance(bar_init, int):
      if a is None:
        a = jnp.ones((bar_init,)) / bar_init

      rng = jax.random.PRNGKey(seed)
      _, b = problem.segmented_y_b
      keys = jax.random.split(rng, len(b))

      linear_solver = self._quad_solver.linear_ot_solver
      transports = init_transports(linear_solver, keys, b, a, problem.epsilon)

      x = problem.update_features(transports, a)
      c = problem.update_barycenter(transports, a)
    else:
      c, x = bar_init if isinstance(bar_init, tuple) else (bar_init, None)
      bar_size = c.shape[0]

      if a is None:
        a = jnp.ones((bar_size,)) / bar_size
      assert c.shape == (bar_size, bar_size)
      assert a.shape == (bar_size,)

      if problem.is_fused:
        assert x is not None, "barycenter features are not initialized"
        _, _, d = problem.segmented_y_fused.shape
        assert x.shape == (bar_size, d)

    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones((
          num_iter, problem.max_measure_size,
          self.linear_ot_solver.outer_iterations
      ))
    else:
      errors = None

    costs = -jnp.ones((num_iter,))
    return GWBarycenterState(x=x, c=c, a=a, errors=errors, costs=costs)

  def update_state(
      self,
      state: GWBarycenterState,
      iteration: int,
      problem: GWBarycenterProblem,
      store_errors: bool = True,
  ) -> Tuple[float, bool, jnp.ndarray, Optional[jnp.ndarray]]:

    def solve_gw(
        state: GWBarycenterState, b: jnp.ndarray, y: jnp.ndarray,
        f: Optional[jnp.ndarray]
    ) -> Any:
      geom_xx = geometry.Geometry(state.c, epsilon=problem.epsilon)
      if problem.is_cost:
        geom_yy = geometry.Geometry(y, epsilon=problem.epsilon)
      else:
        geom_yy = pointcloud.PointCloud(y, epsilon=problem.epsilon)

      if problem.is_fused:
        geom_xy = pointcloud.PointCloud(state.x, f, epsilon=problem.epsilon)
      else:
        geom_xy = None

      quad_problem = quad_problems.QuadraticProblem(
          geom_xx=geom_xx,
          geom_yy=geom_yy,
          geom_xy=geom_xy,
          a=state.a,
          b=b,
          fused_penalty=problem.fused_penalty,
      )
      out = self._quad_solver(quad_problem)

      return (
          out.reg_gw_cost, out.convergence, out.matrix,
          out.errors if store_errors else None
      )

    in_axes = [None, 0, 0]
    in_axes += [0] if problem.is_fused else [None]
    solve_fn = jax.vmap(solve_gw, in_axes=in_axes)

    y, b = problem.segmented_y_b
    y_f = problem.segmented_y_fused
    costs, convs, transports, errors = solve_fn(state, b, y, y_f)

    cost = jnp.sum(costs * problem.weights)
    costs = state.costs.at[iteration].set(cost)

    x_new = problem.update_features(transports, state.a)
    c_new = problem.update_barycenter(transports, state.a)
    # TODO(michalk8): set other flags
    return state.set(x=x_new, c=c_new, costs=costs)

  def output_from_state(self, state: GWBarycenterState) -> GWBarycenterState:
    # for consistency with cont. barycenter, will be refactored in the future
    return state

  def tree_flatten(self) -> Tuple[Sequence[Any], Mapping[str, Any]]:
    children, aux = super().tree_flatten()
    aux['_gw_kwargs'] = self._quad_solver._kwargs
    return children, aux

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "GromovWassersteinBarycenter":
    gw_kwargs = aux_data.pop("_gw_kwargs")
    aux_data = {**aux_data, **gw_kwargs}
    return super().tree_unflatten(aux_data, children)


@partial(jax.vmap, in_axes=[None, 0, 0, None, None])
def init_transports(
    solver, key: jnp.ndarray, b: jnp.ndarray, a: jnp.ndarray, eps: float
) -> jnp.ndarray:
  cost = jax.random.uniform(key, shape=(len(a), len(b)), minval=0, maxval=1)
  geom = geometry.Geometry(cost, epsilon=eps)
  problem = problems.LinearProblem(geom, a=a, b=b)
  return solver(problem).matrix


def iterations(
    solver: GromovWassersteinBarycenter, problem: GWBarycenterProblem,
    init_state: GWBarycenterState
) -> GWBarycenterState:

  def cond_fn(
      iteration: int, constants: GromovWassersteinBarycenter,
      state: GWBarycenterState
  ) -> bool:
    solver, _ = constants
    return solver._continue(state, iteration)

  def body_fn(
      iteration, constants: Tuple[GromovWassersteinBarycenter,
                                  GWBarycenterProblem],
      state: GWBarycenterState, compute_error: bool
  ) -> GWBarycenterState:
    del compute_error  # always assumed true
    solver, problem = constants
    return solver.update_state(state, iteration, problem)

  state = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=(solver, problem),
      state=init_state,
  )
  return state
