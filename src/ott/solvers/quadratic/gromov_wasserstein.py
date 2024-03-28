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
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np

from ott import utils
from ott.geometry import geometry
from ott.initializers.quadratic import initializers as quad_initializers
from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn, sinkhorn_lr

__all__ = ["GromovWasserstein", "GWOutput"]

LinearOutput = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput]

ProgressCallbackFn_t = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "GWState"]], None]


class GWOutput(NamedTuple):
  """Holds the output of the Gromov-Wasserstein solver.

  Args:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    converged: Convergence flag for the outer GW iterations.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
    linear_state: State used to solve and store solutions to the local
      linearization of GW.
    geom: The geometry underlying the local linearization.
    old_transport_mass: Holds total mass of transport at previous iteration.
  """

  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  converged: bool = False
  errors: Optional[jnp.ndarray] = None
  linear_state: Optional[LinearOutput] = None
  geom: Optional[geometry.Geometry] = None
  # Intermediate values.
  old_transport_mass: float = 1.0

  def set(self, **kwargs: Any) -> "GWOutput":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix."""
    return self._rescale_factor * self.linear_state.matrix

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply the transport to an array; axis=1 for its transpose."""
    return self._rescale_factor * self.linear_state.apply(inputs, axis=axis)

  @property
  def reg_gw_cost(self) -> float:
    """Regularized optimal transport cost of the linearization."""
    return self.linear_state.reg_ot_cost

  @property
  def _rescale_factor(self) -> float:
    return jnp.sqrt(self.old_transport_mass / self.linear_state.transport_mass)

  @property
  def primal_cost(self) -> float:
    """Return transport cost of current linear OT solution at geometry."""
    return self.linear_state.transport_cost_at_geom(other_geom=self.geom)

  @property
  def n_iters(self) -> int:  # noqa: D102
    if self.errors is None:
      return -1
    return jnp.sum(self.errors[:, 0] != -1)


class GWState(NamedTuple):
  """State of the Gromov-Wasserstein solver.

  Attributes:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    linear_state: State used to solve and store solutions to the local
      linearization of GW.
    linear_pb: Local linearization of the quadratic GW problem.
    old_transport_mass: Intermediary value of the mass of the transport matrix.
    rngs: Random keys passed to low-rank initializers at every GW iteration
      when not using warm start.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
  """

  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_state: LinearOutput
  linear_pb: linear_problem.LinearProblem
  old_transport_mass: float
  rngs: Optional[jax.Array] = None
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "GWState":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(  # noqa: D102
      self, iteration: int, linear_sol: LinearOutput,
      linear_pb: linear_problem.LinearProblem, store_errors: bool,
      old_transport_mass: float
  ) -> "GWState":
    costs = self.costs.at[iteration].set(linear_sol.reg_ot_cost)
    errors = None
    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :].set(linear_sol.errors)
    linear_convergence = self.linear_convergence.at[iteration].set(
        linear_sol.converged
    )

    return self.set(
        linear_state=linear_sol,
        linear_pb=linear_pb,
        costs=costs,
        linear_convergence=linear_convergence,
        errors=errors,
        old_transport_mass=old_transport_mass
    )


@jax.tree_util.register_pytree_node_class
class GromovWasserstein(was_solver.WassersteinSolver):
  """Gromov-Wasserstein solver :cite:`peyre:16`.

  .. seealso::
    Low-rank Gromov-Wasserstein :cite:`scetbon:23` is implemented in
    :class:`~ott.solvers.quadratic.gromov_wasserstein_lr.LRGromovWasserstein`.

  Args:
    args: Positional arguments for
      :class:`~ott.solvers.was_solver.WassersteinSolver`.
    warm_start: Whether to initialize Sinkhorn calls using values
      from the previous iteration. If :obj:`None`, warm starts are not used for
      standard Sinkhorn.
    relative_epsilon: Whether to use relative epsilon in the linearized
      geometry.
    quad_initializer: Quadratic initializer. If the solver is entropic,
      :class:`~ott.initializers.quadratic.initializers.QuadraticInitializer`
      is always used.
    progress_fn: callback function which gets called during the
      Gromov-Wasserstein iterations, so the user can display the error at each
      iteration, e.g., using a progress bar.
      See :func:`~ott.utils.default_progress_fn` for a basic implementation.
    kwargs_init: Keyword arguments when creating the initializer.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.was_solver.WassersteinSolver`.
  """

  def __init__(
      self,
      *args: Any,
      warm_start: Optional[bool] = None,
      relative_epsilon: Optional[bool] = None,
      quad_initializer: Optional[
          Union[Literal["random", "rank2", "k-means", "generalized-k-means"],
                quad_initializers.BaseQuadraticInitializer]] = None,
      progress_fn: Optional[ProgressCallbackFn_t] = None,
      kwargs_init: Optional[Mapping[str, Any]] = None,
      **kwargs: Any
  ):
    super().__init__(*args, **kwargs)
    assert not self.is_low_rank, \
      "For low-rank GW, use " \
      "`ott.solvers.quadratic.gromov_wasserstein_lr.LRGromovWasserstein`."
    self._warm_start = warm_start
    self.relative_epsilon = relative_epsilon
    self.quad_initializer = quad_initializer
    self.progress_fn = progress_fn
    self.kwargs_init = {} if kwargs_init is None else kwargs_init

  def __call__(
      self,
      prob: quadratic_problem.QuadraticProblem,
      init: Optional[linear_problem.LinearProblem] = None,
      rng: Optional[jax.Array] = None,
      **kwargs: Any,
  ) -> GWOutput:
    """Run the Gromov-Wasserstein solver.

    Args:
      prob: Quadratic OT problem.
      init: Initial linearization of the quadratic problem. If `None`, it will
        be computed using the initializer.
      rng: Random number key.
      kwargs: Keyword arguments used when calling the initializer.

    Returns:
      The Gromov-Wasserstein output.
    """
    rng = utils.default_prng_key(rng)
    rng1, rng2 = jax.random.split(rng, 2)

    if prob._is_low_rank_convertible:
      prob = prob.to_low_rank()

    if init is None:
      initializer = self.create_initializer(prob)
      init = initializer(
          prob,
          epsilon=self.epsilon,
          rng=rng1,
          relative_epsilon=self.relative_epsilon,
          **kwargs
      )

    out = iterations(self, prob, init, rng2)
    # TODO(lpapaxanthoos): remove stop_gradient when using backprop
    if self.is_low_rank:
      linearization = prob.update_lr_linearization(
          jax.lax.stop_gradient(out.linear_state),
          relative_epsilon=self.relative_epsilon,
      )
    else:
      linearization = prob.update_linearization(
          jax.lax.stop_gradient(out.linear_state),
          epsilon=self.epsilon,
          old_transport_mass=jax.lax.stop_gradient(out.old_transport_mass),
          relative_epsilon=self.relative_epsilon,
      )

    linear_state = out.linear_state.set_cost(linearization, True, True)
    iteration = jnp.sum(out.costs != -1)
    converged = jnp.logical_and(
        iteration < self.max_iterations, jnp.all(out.linear_convergence)
    )
    return out.set(
        linear_state=linear_state, geom=linearization.geom, converged=converged
    )

  def init_state(
      self,
      prob: quadratic_problem.QuadraticProblem,
      init: linear_problem.LinearProblem,
      rng: jax.Array,
  ) -> GWState:
    """Initialize the state of the Gromov-Wasserstein iterations.

    Args:
      prob: Quadratic OT problem.
      init: Initial linearization of the quadratic problem.
      rng: Random key for low-rank initializers. Only used when
        :attr:`warm_start` is `False`.

    Returns:
      The initial Gromov-Wasserstein state.
    """
    linear_state = self.linear_ot_solver(init)
    num_iter = self.max_iterations
    transport_mass = prob.init_transport_mass()
    if self.store_inner_errors:
      errors = -jnp.ones((num_iter, self.linear_ot_solver.outer_iterations))
    else:
      errors = None

    return GWState(
        costs=-jnp.ones((num_iter,)),
        linear_convergence=-jnp.ones((num_iter,)),
        linear_state=linear_state,
        linear_pb=init,
        old_transport_mass=transport_mass,
        rngs=jax.random.split(rng, num_iter),
        errors=errors,
    )

  def output_from_state(
      self,
      state: GWState,
  ) -> GWOutput:
    """Create an output from a loop state.

    Arguments:
      state: A GWState.

    Returns:
      A GWOutput.
    """
    return GWOutput(
        costs=state.costs,
        linear_convergence=state.linear_convergence,
        errors=state.errors,
        linear_state=state.linear_state,
        geom=state.linear_pb.geom,
        old_transport_mass=state.old_transport_mass
    )

  def create_initializer(
      self, prob: quadratic_problem.QuadraticProblem
  ) -> quad_initializers.BaseQuadraticInitializer:
    """Create quadratic, possibly low-rank initializer.

    Args:
      prob: Quadratic OT problem used to determine the initializer.

    Returns:
      The initializer.
    """
    if isinstance(
        self.quad_initializer, quad_initializers.BaseQuadraticInitializer
    ):
      return self.quad_initializer
    # no other options implemented, use the default
    return quad_initializers.QuadraticInitializer(**self.kwargs_init)

  @property
  def warm_start(self) -> bool:
    """Whether to initialize Sinkhorn using previous solutions."""
    return self.is_low_rank if self._warm_start is None else self._warm_start

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    children, aux_data = super().tree_flatten()
    aux_data["warm_start"] = self._warm_start
    aux_data["progress_fn"] = self.progress_fn
    aux_data["relative_epsilon"] = self.relative_epsilon
    aux_data["quad_initializer"] = self.quad_initializer
    aux_data["kwargs_init"] = self.kwargs_init
    return children, aux_data


def iterations(
    solver: GromovWasserstein,
    prob: quadratic_problem.QuadraticProblem,
    init: linear_problem.LinearProblem,
    rng: jax.Array,
) -> GWOutput:
  """Jittable Gromov-Wasserstein outer loop."""

  def cond_fn(
      iteration: int, solver: GromovWasserstein, state: GWState
  ) -> bool:
    return solver._continue(state, iteration)

  def body_fn(
      iteration: int, solver: GromovWasserstein, state: GWState,
      compute_error: bool
  ) -> GWState:
    del compute_error  # always assumed true for the outer loop of GW

    lin_state = state.linear_state
    if solver.is_low_rank:
      rng = state.rngs[iteration]
      init = (lin_state.q, lin_state.r,
              lin_state.g) if solver.warm_start else (None, None, None)
      linear_pb = prob.update_lr_linearization(
          state.linear_state, relative_epsilon=solver.relative_epsilon
      )
      out = solver.linear_ot_solver(linear_pb, init=init, rng=rng)
    else:
      init = (lin_state.f, lin_state.g) if solver.warm_start else (None, None)
      linear_pb = prob.update_linearization(
          lin_state,
          solver.epsilon,
          state.old_transport_mass,
          relative_epsilon=solver.relative_epsilon,
      )
      out = solver.linear_ot_solver(linear_pb, init=init)

    old_transport_mass = jax.lax.stop_gradient(
        state.linear_state.transport_mass
    )
    new_state = state.update(
        iteration, out, linear_pb, solver.store_inner_errors, old_transport_mass
    )

    # Inner iterations is currently fixed to 1.
    inner_iterations = 1
    if solver.progress_fn is not None:
      jax.debug.callback(
          solver.progress_fn,
          (iteration, inner_iterations, solver.max_iterations, state)
      )

    return new_state

  state = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=solver,
      state=solver.init_state(prob, init, rng=rng)
  )

  return solver.output_from_state(state)
