# coding=utf-8
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
import functools
from typing import Any, Dict, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from ott.core import fixed_point_loop
from ott.core import problems
from ott.core import quad_problems
from ott.core import sinkhorn
from ott.core import sinkhorn_lr

from ott.core import was_solver
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry
from ott.geometry import low_rank
from ott.geometry import pointcloud


class GWOutput(NamedTuple):
  """Holds the output of the Gromov-Wasserstein solver.

  Attributes:
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    convergence: Bool convergence flag for the outer GW iterations.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
    linear_state: State used to solve and store solutions to the local
      linearization of GW.
    geom: The geometry underlying the local linearization.
    old_transport_mass: Holds total mass of transport at previous iteration.
    transport: The transport matrix.
    reg_gw_cost: Regularized optimal transport cost of the linearization.
  """
  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  convergence: bool = False
  errors: Optional[jnp.ndarray] = None
  linear_state: Any = None
  geom: geometry.Geometry = None
  # Intermediate values.
  old_transport_mass: Optional[float] = 1.0

  def set(self, **kwargs) -> 'GWOutput':
    """Returns a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  @property
  def matrix(self):
    """Transport matrix."""
    rescale_factor = jnp.sqrt(
        self.old_transport_mass / self.linear_state.transport_mass())
    return self.linear_state.matrix * rescale_factor

  @property
  def reg_gw_cost(self):
    return self.linear_state.reg_ot_cost


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
  """
  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  errors: Optional[jnp.ndarray] = None
  linear_state: Any = None
  linear_pb: Optional[problems.LinearProblem] = None
  # Intermediate values.
  old_transport_mass: Optional[float] = 1.0

  def set(self, **kwargs) -> 'GWState':
    """Returns a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(self, iteration: int, linear_sol, linear_pb, store_errors: bool,
             old_transport_mass: float):
    costs = self.costs.at[iteration].set(linear_sol.reg_ot_cost)
    errors = None
    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :].set(linear_sol.errors)
    linear_convergence = self.linear_convergence.at[iteration].set(
        linear_sol.converged)
    return self.set(linear_state=linear_sol,
                    linear_pb=linear_pb,
                    costs=costs,
                    linear_convergence=linear_convergence,
                    errors=errors,
                    old_transport_mass=old_transport_mass)


@jax.tree_util.register_pytree_node_class
class GromovWasserstein(was_solver.WassersteinSolver):
  """A Gromov Wasserstein solver, built on generic template."""

  def __call__(self, prob: quad_problems.QuadraticProblem) -> GWOutput:
    # Consider converting problem first if using low-rank solver
    if self.is_low_rank:
      convert = (
          isinstance(prob.geom_xx, pointcloud.PointCloud) and
          prob.geom_xx.is_squared_euclidean and
          isinstance(prob.geom_yy, pointcloud.PointCloud) and
          prob.geom_yy.is_squared_euclidean
          )
      # Consider converting
      if convert:
        if not prob.is_fused or isinstance(prob.geom_xy,
                                           low_rank.LRCGeometry):
          prob.geom_xx = prob.geom_xx.to_LRCGeometry()
          prob.geom_yy = prob.geom_yy.to_LRCGeometry()
        else:
          if (isinstance(prob.geom_xy, pointcloud.PointCloud) and
              prob.geom_xy.is_squared_euclidean):
            prob.geom_xy = prob.geom_xy.to_LRCGeometry(prob.fused_penalty)
            prob.geom_xx = prob.geom_xx.to_LRCGeometry()
            prob.geom_yy = prob.geom_yy.to_LRCGeometry()

    # Possibly jit iteration functions and run. Closure on rank to
    # avoid jitting issues, since rank value will be used to branch between
    # a default entropic GW or a low-rank GW.
    iterations_fn = functools.partial(iterations, rank=self.rank)
    gromov_fn = jax.jit(iterations_fn) if self.jit else iterations_fn
    out = gromov_fn(self, prob)
    # TODO(lpapaxanthos): remove stop_gradient when using backprop
    if self.is_low_rank:
      linearization = prob.update_lr_linearization(
          jax.lax.stop_gradient(out.linear_state))
    else:
      linearization = prob.update_linearization(
          jax.lax.stop_gradient(out.linear_state),
          self.epsilon,
          jax.lax.stop_gradient(out.old_transport_mass))
    linear_state = out.linear_state.set_cost(linearization, True, True)
    iteration = jnp.sum(out.costs != -1)
    convergence = jnp.logical_and(iteration < self.max_iterations,
                                  jnp.all(out.linear_convergence))
    return out.set(linear_state=linear_state,
                   convergence=convergence)

  def init_state(self, prob: quad_problems.QuadraticProblem,
                 rank: int) -> GWState:
    """Initializes the state of the Gromov-Wasserstein iterations."""
    if rank > 0:
      linearization = prob.init_lr_linearization(rank)
    else:
      linearization = prob.init_linearization(self.epsilon)

    linear_state = self.linear_ot_solver(linearization)
    num_iter = self.max_iterations
    transport_mass = prob.init_transport_mass()
    if self.store_inner_errors:
      errors = -jnp.ones((num_iter, self.linear_ot_solver.outer_iterations))
    else:
      errors = None
    return GWState(-jnp.ones((num_iter,)), -jnp.ones((num_iter,)),
                   errors, linear_state, linearization, transport_mass)

  def output_from_state(self, state):
    """Create an output from a loop state.

    Arguments:
      state: A GWState.

    Returns:
      A GWOutput.
    """
    geom = state.linear_pb.geom
    return GWOutput(costs=state.costs,
                    linear_convergence=state.linear_convergence,
                    errors=state.errors,
                    linear_state=state.linear_state,
                    geom=geom,
                    old_transport_mass=state.old_transport_mass)


def iterations(solver: GromovWasserstein,
               prob: quad_problems.QuadraticProblem,
               rank: int) -> GWOutput:
  """A jittable Gromov-Wasserstein outer loop."""

  def cond_fn(iteration, constants, state):
    solver = constants
    return solver._continue(state, iteration)

  def body_fn(iteration, constants, state, compute_error):
    del compute_error  # Always assumed True for outer loop of GW.
    solver = constants
    if rank > 0:
      linear_pb = prob.update_lr_linearization(state.linear_state)
    else:
      linear_pb = prob.update_linearization(state.linear_state, solver.epsilon,
                                            state.old_transport_mass)

    out = solver.linear_ot_solver(linear_pb)
    old_transport_mass = jax.lax.stop_gradient(
        state.linear_state.transport_mass())
    return state.update(
        iteration,
        out,
        linear_pb,
        solver.store_inner_errors,
        old_transport_mass)

  state = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=solver,
      state=solver.init_state(prob, rank))

  return solver.output_from_state(state)


def make(
    epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.,
    rank: int = -1,
    max_iterations: int = 50,
    jit: bool = False,
    warm_start: bool = True,
    store_inner_errors: bool = False,
    linear_ot_solver_kwargs: Optional[Dict[str, Any]] = None,
    threshold: float = 1e-2,
    min_iterations: int = 1,
    **kwargs) -> GromovWasserstein:
  """Creates a GromovWasserstein solver.

  Args:
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    rank: integer used to constrain the rank of GW solutions if >0.
    max_iterations: int32, the maximum number of outer iterations for
      Gromov Wasserstein.
    jit: bool, if True, jits the function.
    warm_start: deprecated.
    store_inner_errors: whether or not to return all the errors of the inner
      Sinkhorn iterations.
    linear_ot_solver_kwargs: Optionally a dictionary containing the keywords
      arguments for the linear OT solver (e.g. sinkhorn)
    threshold: threshold (progress between two iterate costs) used to stop GW.
    min_iterations: see fixed_point_loop.
    **kwargs: additional kwargs for epsilon.

  Returns:
    A GromovWasserstein solver.
  """
  del warm_start
  if linear_ot_solver_kwargs is None:
    linear_ot_solver_kwargs = {}

  if rank == -1:
    sink = sinkhorn.make(**linear_ot_solver_kwargs)
  elif rank > 0:
    # `rank` and `epsilon` are arguments of the `sinkhorn_lr` solver. As we are
    # passing them to make, we should not pass them in `linear_ot_solver_kwargs`
    # Therefore, the `rank` or `epsilon` passed to `linear_ot_solver_kwargs` are
    # deleted.
    linear_ot_solver_kwargs.pop('rank', None)
    linear_ot_solver_kwargs.pop('epsilon', None)
    sink = sinkhorn_lr.make(
        rank=rank, epsilon=epsilon, **linear_ot_solver_kwargs)

  return GromovWasserstein(
      epsilon, rank, max_iterations=max_iterations,
      jit=jit, linear_ot_solver=sink, threshold=threshold,
      store_inner_errors=store_inner_errors,
      min_iterations=min_iterations, **kwargs)


def gromov_wasserstein(
    geom_xx: geometry.Geometry,
    geom_yy: geometry.Geometry,
    geom_xy: Optional[geometry.Geometry] = None,
    fused_penalty: Optional[float] = None,
    scale_cost: Optional[Union[bool, float, str]] = False,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    loss: Optional[str] = None,
    tau_a: Optional[float] = 1.0,
    tau_b: Optional[float] = 1.0,
    gw_unbalanced_correction: bool = True,
    **kwargs) -> GWOutput:
  """Wrapper to solve a Gromov Wasserstein problem.

  Wrapper that instantiates a quadratic problem (possibly with linear term
  if the problem is fused) and calls a solver to output a solution.

  Args:
    geom_xx: a Geometry object for the first view.
    geom_yy: a second Geometry object for the second view.
    geom_xy: a Geometry object representing the linear cost in FGW.
    fused_penalty: multiplier of the linear term in Fused Gromov Wasserstein,
      i.e. loss = quadratic_loss + fused_penalty * linear_loss. If geom_xy is
      None fused_penalty will be ignored, i.e. fused_penalty = 0
    scale_cost: option to rescale the cost matrices:

      - if `True`, use the default for each geometry.
      - if `False`, keep the original scaling in geometries.
      - if :class:`str`, use a specific method available in
        :meth:`ott.geometry.geometry.Geometry.__init__` or
        :meth:`ott.geometry.pointcloud.PointCloud.__init__`.
      - if `None`, do not scale the cost matrices.

    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    loss: str, None defaults to the square Euclidean distance, can also
      receive 'kl' to define the GW loss.
    tau_a: float between 0 and 1.0, parameter that controls the strength of the
      KL divergence constraint between the weights and marginals of the
      transport for the first view. If set to 1.0, then it is equivalent to a
      hard constraint and if smaller to a softer constraint.
    tau_b: float between 0 and 1.0, parameter that controls the strength of the
      KL divergence constraint between the weights and marginals of the
      transport for the second view. If set to 1.0, then it is equivalent to a
      hard constraint and if smaller to a softer constraint.
    gw_unbalanced_correction: True (default) if the unbalanced version of
      Sejourne et al (Neurips 2021) is used, False if tau_a and tau_b
      only affect the inner Sinhkorn loop.
    **kwargs: keyword arguments to make.

  Returns:
    A GromovWassersteinState named tuple.
  """
  losses = {'kl': quad_problems.make_kl_loss}
  loss_fn = losses.get(loss, None)
  prob = quad_problems.QuadraticProblem(
      geom_xx,
      geom_yy,
      geom_xy=geom_xy,
      fused_penalty=fused_penalty,
      scale_cost=scale_cost,
      a=a,
      b=b,
      loss=(loss_fn() if loss_fn is not None else None),
      tau_a=tau_a,
      tau_b=tau_b,
      gw_unbalanced_correction=gw_unbalanced_correction)
  solver = make(**kwargs)
  return solver(prob)
