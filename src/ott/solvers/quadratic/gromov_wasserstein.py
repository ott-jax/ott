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
"""A Jax version of the regularised GW Solver (Peyre et al. 2016)."""
from typing import (
    Any,
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

from ott.geometry import geometry, low_rank, pointcloud
from ott.initializers.linear import initializers_lr
from ott.initializers.quadratic import initializers as quad_initializers
from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_costs, quadratic_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn, sinkhorn_lr

__all__ = ["GWOutput", "GromovWasserstein", "solve"]

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
    return jnp.sqrt(self.old_transport_mass / self.linear_state.transport_mass)


class GWState(NamedTuple):
  """Holds the state of the Gromov-Wasserstein solver.

  Attributes:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    linear_state: State used to solve and store solutions to the local
      linearization of GW.
    linear_pb: Local linearization of the quadratic GW problem.
    old_transport_mass: Intermediary value of the mass of the transport matrix.
    keys: Random keys passed to low-rank initializers at every GW iteration
      when not using warm start.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
  """

  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_state: LinearOutput
  linear_pb: linear_problem.LinearProblem
  old_transport_mass: float
  keys: Optional[jnp.ndarray] = None
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> 'GWState':
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(
      self, iteration: int, linear_sol: LinearOutput,
      linear_pb: linear_problem.LinearProblem, store_errors: bool,
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
    args: Positional arguments for
      :class:`~ott.solvers.was_solver.WassersteinSolver`.
    warm_start: Whether to initialize (low-rank) Sinkhorn calls using values
      from the previous iteration. If `None`, warm starts are not used for
      standard Sinkhorn, but used for low-rank Sinkhorn.
    quad_initializer: Quadratic initializer. If the solver is entropic,
      :class:`~ott.initializers.quadratic.initializers.QuadraticInitializer`
      is always used. Otherwise, the quadratic initializer wraps the low-rank
      Sinkhorn initializers:

        - `'random'` - :class:`~ott.initializers.linear.initializers_lr.RandomInitializer`.
        - `'rank2'` - :class:`~ott.initializers.linear.initializers_lr.Rank2Initializer`.
        - `'k-means'` - :class:`~ott.initializers.linear.initializers_lr.KMeansInitializer`.
        - `'generalized-k-means'` - :class:`~ott.initializers.linear.initializers_lr.GeneralizedKMeansInitializer`.

      If `None`, the low-rank initializer will be selected in a problem-specific
      manner:

        - if both :attr:`~ott.problems.quadratic.quadratic_problem.QuadraticProblem.geom_xx`
          and :attr:`~ott.problems.quadratic.quadratic_problem.QuadraticProblem.geom_yy`
          are :class:`~ott.geometry.pointcloud.PointCloud` or :class:`~ott.geometry.low_rank.LRCGeometry`,
          :class:`~ott.initializers.linear.initializers_lr.KMeansInitializer`
          is used.
        - otherwise, use :class:`~ott.initializers.linear.initializers_lr.RandomInitializer`.

    kwargs_init: Keyword arguments when creating the initializer.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.was_solver.WassersteinSolver`.
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
      prob: quadratic_problem.QuadraticProblem,
      init: Optional[linear_problem.LinearProblem] = None,
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

    if prob._is_low_rank_convertible:
      prob = prob.to_low_rank()

    if init is None:
      initializer = self.create_initializer(prob)
      init = initializer(prob, epsilon=self.epsilon, key=key1, **kwargs)

    run_fn = jax.jit(iterations) if self.jit else iterations
    out = run_fn(self, prob, init, key2)
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
      prob: quadratic_problem.QuadraticProblem,
      init: linear_problem.LinearProblem,
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
      if self.is_low_rank:
        assert isinstance(
            self.quad_initializer, quad_initializers.LRQuadraticInitializer
        ), f"Expected quadratic initializer to be low rank, " \
           f"found `{type(self.quad_initializer).__name__}`."
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
    prob: quadratic_problem.QuadraticProblem,
    init: linear_problem.LinearProblem,
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
        state.linear_state.transport_mass
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


def solve(
    geom_xx: geometry.Geometry,
    geom_yy: geometry.Geometry,
    geom_xy: Optional[geometry.Geometry] = None,
    fused_penalty: float = 1.0,
    scale_cost: Optional[Union[bool, float, str]] = False,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    loss: Union[Literal['sqeucl', 'kl'], quadratic_costs.GWLoss] = 'sqeucl',
    tau_a: Optional[float] = 1.0,
    tau_b: Optional[float] = 1.0,
    gw_unbalanced_correction: bool = True,
    ranks: Union[int, Tuple[int, ...]] = -1,
    tolerances: Union[float, Tuple[float, ...]] = 1e-2,
    **kwargs: Any,
) -> GWOutput:
  r"""Solve quadratic regularized OT problem.

  The quadratic loss of a single OT matrix is assumed to
  have the form given in :cite:`peyre:16`, eq. 4.

  The two geometries below parameterize matrices :math:`C` and :math:`\bar{C}`
  in that equation. The function :math:`L` (of two real values) in that equation
  is assumed to match the form given in eq. 5., with our notations:

  .. math::

    L(x, y) = lin1(x) + lin2(y) - quad1(x) * quad2(y)

  Args:
    geom_xx: Ground geometry of the first space.
    geom_yy: Ground geometry of the second space.
    geom_xy: Geometry defining the linear penalty term for
      Fused Gromov-Wasserstein. If `None`, the problem reduces to
      a plain Gromov-Wasserstein problem.
    fused_penalty: multiplier of the linear term in Fused Gromov-Wasserstein,
      i.e. problem = purely quadratic + fused_penalty * linear problem.
      Ignored if ``geom_xy`` is not specified.
    scale_cost: option to rescale the cost matrices:

      - if :obj:`True`, use the default for each geometry.
      - if :obj:`False`, keep the original scaling in geometries.
      - if :class:`str`, use a specific method available in
        :class:`~ott.geometry.geometry.Geometry` or
        :class:`~ott.geometry.pointcloud.PointCloud`.
      - if :obj:`None`, do not scale the cost matrices.

    a: array representing the probability weights of the samples
      from ``geom_xx``. If `None`, it will be uniform.
    b: array representing the probability weights of the samples
      from ``geom_yy``. If `None`, it will be uniform.
    loss: a 2-tuple of 2-tuples of Callable. The first tuple is the linear
      part of the loss. The second one is the quadratic part (quad1, quad2).
      By default, the loss is set as the 4 functions representing the squared
      Euclidean loss, and this property is taken advantage of in subsequent
      computations. Alternatively, KL loss can be specified in no less optimized
      way.
    tau_a: if `< 1.0`, defines how much unbalanced the problem is on
      the first marginal.
    tau_b: if `< 1.0`, defines how much unbalanced the problem is on
      the second marginal.
    gw_unbalanced_correction: Whether the unbalanced version of
      :cite:`sejourne:21` is used. Otherwise, ``tau_a`` and ``tau_b`` only affect
      the inner Sinkhorn loop.
    ranks: Ranks of the cost matrices, see
      :meth:`~ott.geometry.geometry.Geometry.to_LRCGeometry`. Used when
      geometries are *not* :class:`~ott.geometry.pointcloud.PointCloud` with
      `'sqeucl'` cost function. If `-1`, the geometries will not be converted
      to low-rank. If :class:`tuple`, it specifies the ranks of ``geom_xx``,
      ``geom_yy`` and ``geom_xy``, respectively. If :class:`int`, rank is shared
      across all geometries.
    tolerances: Tolerances used when converting geometries to low-rank. Used
      when geometries are not :class:`~ott.geometry.pointcloud.PointCloud` with
      `'sqeucl'` cost. If :class:`float`, it is shared across all geometries.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein`.

  Returns:
    Gromov-Wasserstein output.
  """
  prob = quadratic_problem.QuadraticProblem(
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
  solver = GromovWasserstein(**kwargs)
  return solver(prob)
