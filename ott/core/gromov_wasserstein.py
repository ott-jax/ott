# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A Jax version of the regularised GW Solver (Peyre et al. 2016)."""
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import (
    fixed_point_loop,
    initializers_lr,
    linear_problems,
    quad_initializers,
    quad_problems,
    sinkhorn,
    sinkhorn_lr,
    was_solver,
)
from ott.geometry import epsilon_scheduler, geometry, low_rank, pointcloud

LinearOutput = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput]


class GWOutput(NamedTuple):
  """Holds the output of the Gromov-Wasserstein solver.

  Args:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    converged: Bool convergence flag for the outer GW iterations.
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

  def set(self, **kwargs: Any) -> 'GWOutput':
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
    return jnp.sqrt(
        self.old_transport_mass / self.linear_state.transport_mass()
    )


class GWState(NamedTuple):
  """Holds the state of the Gromov-Wasserstein solver.

  Attributes:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
    linear_state: State used to solve and store solutions to the local
      linearization of GW.
    linear_pb: Local linearization of the quadratic GW problem.
    old_transport_mass: Intermediary value of the mass of the transport matrix.
    keys: Random keys passed to low-rank initializers at every GW iteration
      when not using warm start.
  """

  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_state: LinearOutput
  linear_pb: linear_problems.LinearProblem
  old_transport_mass: float
  keys: Optional[jnp.ndarray] = None
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> 'GWState':
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(
      self, iteration: int, linear_sol: LinearOutput,
      linear_pb: linear_problems.LinearProblem, store_errors: bool,
      old_transport_mass: float
  ) -> 'GWState':
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
  """Gromov-Wasserstein solver.

  Args:
    args: Positional_arguments for
      :class:`~ott.core.was_solver.WassersteinSolver`.
    warm_start: Whether to initialize (low-rank) Sinkhorn calls using values
      from the previous iteration. If `None`, warm starts are not used for
      standard Sinkhorn, but used for low-rank Sinkhorn.
    quad_initializer: Quadratic initializer. If the solver is entropic,
      :class:`~ott.core.quad_initializers.QuadraticInitializer` is always used.
      Otherwise, the quadratic initializer wraps low-rank Sinkhorn initializers:

        - `'random'` - :class:`~ott.core.initializers_lr.RandomInitializer`.
        - `'rank2'` - :class:`~ott.core.initializers_lr.Rank2Initializer`.
        - `'k-means'` - :class:`~ott.core.initializers_lr.KMeansInitializer`.
        - `'generalized-k-means'` -
          :class:`~ott.core.initializers_lr.GeneralizedKMeansInitializer`.

      If `None`, the low-rank initializer will be selected in a problem-specific
      manner:

        - if both :attr:`~ott.core.quad_problems.QuadraticProblem.geom_xx` and
          :attr:`~ott.core.quad_problems.QuadraticProblem.geom_yy` are
          :class:`~ott.geometry.pointcloud.PointCloud`  or
          :class:`~ott.geometry.low_rank.LRCGeometry`,
          :class:`~ott.core.initializers_lr.KMeansInitializer` is used.
        - otherwise, use :class:`~ott.core.initializers_lr.RandomInitializer`.

    kwargs_init: Keyword arguments when creating the initializer.
    kwargs: Keyword arguments for
      :class:`~ott.core.was_solver.WassersteinSolver`.
  """

  def __init__(
      self,
      *args: Any,
      warm_start: Optional[bool] = None,
      quad_initializer: Optional[
          Union[Literal["random", "rank2", "k-means", "generalized-k-means"],
                quad_initializers.BaseQuadraticInitializer]] = None,
      kwargs_init: Optional[Mapping[str, Any]] = None,
      **kwargs: Any
  ):
    super().__init__(*args, **kwargs)
    self._warm_start = warm_start
    self.quad_initializer = quad_initializer
    self.kwargs_init = {} if kwargs_init is None else kwargs_init

  def __call__(
      self,
      prob: quad_problems.QuadraticProblem,
      init: Optional[linear_problems.LinearProblem] = None,
      key: Optional[jnp.ndarray] = None,
      **kwargs: Any,
  ) -> GWOutput:
    """Run the Gromov-Wasserstein solver.

    Args:
      prob: Quadratic OT problem.
      init: Initial linearization of the quadratic problem. If `None`, it will
        be computed using the initializer.
      kwargs: Keyword arguments used when calling the initializer.

    Returns:
      The Gromov-Wasserstein output.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)

    # consider converting problem first if using low-rank solver
    if self.is_low_rank and prob._is_low_rank_convertible:
      prob = prob.to_low_rank()

    if init is None:
      initializer = self.create_initializer(prob)
      init = initializer(prob, epsilon=self.epsilon, key=key1, **kwargs)

    gromov_fn = jax.jit(iterations) if self.jit else iterations
    out = gromov_fn(self, prob, init, key2)
    # TODO(lpapaxanthos): remove stop_gradient when using backprop
    if self.is_low_rank:
      linearization = prob.update_lr_linearization(
          jax.lax.stop_gradient(out.linear_state)
      )
    else:
      linearization = prob.update_linearization(
          jax.lax.stop_gradient(out.linear_state), self.epsilon,
          jax.lax.stop_gradient(out.old_transport_mass)
      )
    linear_state = out.linear_state.set_cost(linearization, True, True)
    iteration = jnp.sum(out.costs != -1)
    converged = jnp.logical_and(
        iteration < self.max_iterations, jnp.all(out.linear_convergence)
    )
    return out.set(linear_state=linear_state, converged=converged)

  def init_state(
      self,
      prob: quad_problems.QuadraticProblem,
      init: linear_problems.LinearProblem,
      key: jnp.ndarray,
  ) -> GWState:
    """Initialize the state of the Gromov-Wasserstein iterations.

    Args:
      prob: Quadratic OT problem.
      init: Initial linearization of the quadratic problem.
      key: Random key for low-rank initializers. Only used when
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
        keys=jax.random.split(key, num_iter),
        errors=errors,
    )

  def output_from_state(self, state: GWState) -> GWOutput:
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
      self, prob: quad_problems.QuadraticProblem
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
      if self.is_low_rank:
        assert isinstance(
            self.quad_initializer, quad_initializers.LRQuadraticInitializer
        ), f"Expected quadratic initializer to be low rank, " \
           f"found `{type(self.quad_initializer).__name___}`."
        assert self.quad_initializer.rank == self.rank, \
            f"Expected quadratic initializer of rank `{self.rank}`, " \
            f"found `{self.quad_initializer.rank}`."
      return self.quad_initializer

    if self.is_low_rank:
      if self.quad_initializer is None:
        types = (pointcloud.PointCloud, low_rank.LRCGeometry)
        kind = "k-means" if isinstance(prob.geom_xx, types) and isinstance(
            prob.geom_yy, types
        ) else "random"
      else:
        kind = self.quad_initializer
      linear_lr_init = initializers_lr.LRInitializer.from_solver(
          self, kind=kind, **self.kwargs_init
      )
      return quad_initializers.LRQuadraticInitializer(linear_lr_init)

    return quad_initializers.QuadraticInitializer(**self.kwargs_init)

  @property
  def warm_start(self) -> bool:
    """Whether to initialize (low-rank) Sinkhorn using previous solutions."""
    return self.is_low_rank if self._warm_start is None else self._warm_start

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    children, aux_data = super().tree_flatten()
    aux_data["warm_start"] = self._warm_start
    aux_data["quad_initializer"] = self.quad_initializer
    aux_data["kwargs_init"] = self.kwargs_init
    return children, aux_data


def iterations(
    solver: GromovWasserstein,
    prob: quad_problems.QuadraticProblem,
    init: linear_problems.LinearProblem,
    key: jnp.ndarray,
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
      key = state.keys[iteration]
      init = (lin_state.q, lin_state.r,
              lin_state.g) if solver.warm_start else (None, None, None)
      linear_pb = prob.update_lr_linearization(state.linear_state)
      out = solver.linear_ot_solver(linear_pb, init=init, key=key)
    else:
      init = (lin_state.f, lin_state.g) if solver.warm_start else (None, None)
      linear_pb = prob.update_linearization(
          lin_state, solver.epsilon, state.old_transport_mass
      )
      out = solver.linear_ot_solver(linear_pb, init=init)

    old_transport_mass = jax.lax.stop_gradient(
        state.linear_state.transport_mass()
    )
    return state.update(
        iteration, out, linear_pb, solver.store_inner_errors, old_transport_mass
    )

  state = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=solver,
      state=solver.init_state(prob, init, key=key)
  )

  return solver.output_from_state(state)


def make(
    epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.,
    rank: int = -1,
    max_iterations: int = 50,
    jit: bool = False,
    warm_start: Optional[bool] = None,
    store_inner_errors: bool = False,
    linear_ot_solver_kwargs: Optional[Mapping[str, Any]] = None,
    threshold: float = 1e-2,
    min_iterations: int = 1,
    **kwargs: Any,
) -> GromovWasserstein:
  """Create a GromovWasserstein solver.

  Args:
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    rank: integer used to constrain the rank of GW solutions if >0.
    max_iterations: the maximum number of outer iterations for
      Gromov Wasserstein.
    jit: bool, if True, jits the function.
    warm_start: Whether to initialize (low-rank) Sinkhorn calls using values
      from the previous iteration. If `None`, it's enabled when using low-rank.
    store_inner_errors: whether or not to return all the errors of the inner
      Sinkhorn iterations.
    linear_ot_solver_kwargs: Optionally a dictionary containing the keywords
      arguments for the linear OT solver (e.g. sinkhorn)
    threshold: threshold (progress between two iterate costs) used to stop GW.
    min_iterations: see fixed_point_loop.
    kwargs: additional kwargs for epsilon.

  Returns:
    A GromovWasserstein solver.
  """
  if linear_ot_solver_kwargs is None:
    linear_ot_solver_kwargs = {}

  if rank == -1:
    sink = sinkhorn.make(**linear_ot_solver_kwargs)
  elif rank > 0:
    # `rank` and `epsilon` are arguments of the `sinkhorn_lr` solver. As we are
    # passing them to make, we should not pass them in `linear_ot_solver_kwargs`
    # Therefore, the `rank` or `epsilon` passed to `linear_ot_solver_kwargs` are
    # deleted.
    _ = linear_ot_solver_kwargs.pop('rank', None)
    _ = linear_ot_solver_kwargs.pop('epsilon', None)
    sink = sinkhorn_lr.make(
        rank=rank, epsilon=epsilon, **linear_ot_solver_kwargs
    )
  else:
    raise ValueError(f"Invalid value for `rank={rank}`.")

  return GromovWasserstein(
      epsilon=epsilon,
      rank=rank,
      linear_ot_solver=sink,
      threshold=threshold,
      min_iterations=min_iterations,
      max_iterations=max_iterations,
      jit=jit,
      store_inner_errors=store_inner_errors,
      warm_start=warm_start,
      **kwargs
  )


def gromov_wasserstein(
    geom_xx: geometry.Geometry,
    geom_yy: geometry.Geometry,
    geom_xy: Optional[geometry.Geometry] = None,
    fused_penalty: float = 1.0,
    scale_cost: Optional[Union[bool, float, str]] = False,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    loss: Union[Literal['sqeucl', 'kl'], quad_problems.GWLoss] = 'sqeucl',
    tau_a: Optional[float] = 1.0,
    tau_b: Optional[float] = 1.0,
    gw_unbalanced_correction: bool = True,
    ranks: Union[int, Tuple[int, ...]] = -1,
    tolerances: Union[float, Tuple[float, ...]] = 1e-2,
    **kwargs: Any,
) -> GWOutput:
  """Solve a Gromov Wasserstein problem.

  Wrapper that instantiates a quadratic problem (possibly with linear term
  if the problem is fused) and calls a solver to output a solution.

  Args:
    geom_xx: a Geometry object for the first view.
    geom_yy: a second Geometry object for the second view.
    geom_xy: a Geometry object representing the linear cost in FGW.
    fused_penalty: multiplier of the linear term in Fused Gromov Wasserstein,
      i.e. loss = quadratic_loss + fused_penalty * linear_loss.
      Ignored if ``geom_xy`` is not specified.
    scale_cost: option to rescale the cost matrices:

      - if `True`, use the default for each geometry.
      - if `False`, keep the original scaling in geometries.
      - if :class:`str`, use a specific method available in
        :class:`ott.geometry.geometry.Geometry` or
        :class:`ott.geometry.pointcloud.PointCloud`.
      - if `None`, do not scale the cost matrices.

    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    loss: defaults to the square Euclidean distance. Can also pass 'kl'
      to define the GW loss as KL loss.
      See :class:`~ott.core.gromov_wasserstein.GromovWasserstein` on how to pass
      custom loss.
    tau_a: float between 0 and 1.0, parameter that controls the strength of the
      KL divergence constraint between the weights and marginals of the
      transport for the first view. If set to 1.0, then it is equivalent to a
      hard constraint and if smaller to a softer constraint.
    tau_b: float between 0 and 1.0, parameter that controls the strength of the
      KL divergence constraint between the weights and marginals of the
      transport for the second view. If set to 1.0, then it is equivalent to a
      hard constraint and if smaller to a softer constraint.
    gw_unbalanced_correction: True (default) if the unbalanced version of
      :cite:`sejourne:21` is used, False if tau_a and tau_b
      only affect the inner Sinkhorn loop.
    ranks: Switch to a low rank approximation of all cost matrices, using
      :meth:`~ott.geometry.geometry.Geometry.to_LRCGeometry`, to gain speed.
      This is only relevant if the geometries of interest are *not*
      :class:`~ott.geometry.pointcloud.PointCloud` with `'sqeucl'` cost
      function, in which case they would be low-rank by construction (as long
      as the sizes of these point clouds is larger than dimension).
      If `-1`, geometries are left as they are, and not converted.
      If :class:`tuple`, these 2 or 3 :class:`int` specify the ranks of
      ``geom_xx``, ``geom_yy`` and ``geom_xy``, respectively. If :class:`int`,
      all 3 geometries are converted using that rank.
    tolerances: Tolerances used when converting geometries to low-rank. Used when
      geometries are *not* :class:`~ott.geometry.pointcloud.PointCloud` with
      `'sqeucl'` cost. If :class:`float`, that tolerance is shared across all
      3 geometries.
    kwargs: Keyword arguments to
      :class:`~ott.core.gromov_wasserstein.GromovWasserstein`.

  Returns:
    A GromovWassersteinState named tuple.
  """
  prob = quad_problems.QuadraticProblem(
      geom_xx,
      geom_yy,
      geom_xy=geom_xy,
      fused_penalty=fused_penalty,
      scale_cost=scale_cost,
      a=a,
      b=b,
      loss=loss,
      tau_a=tau_a,
      tau_b=tau_b,
      gw_unbalanced_correction=gw_unbalanced_correction,
      ranks=ranks,
      tolerances=tolerances
  )
  solver = make(**kwargs)
  return solver(prob)
