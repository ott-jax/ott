# coding=utf-8
# Copyright 2021 Google LLC.
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
"""A Jax version of Sinkhorn's algorithm."""
import collections
import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
from ott.core import fixed_point_loop
from ott.core import problems
from ott.core import sinkhorn
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry


GromovWassersteinOutput = collections.namedtuple(
    'GromovWassersteinOutput', ['f', 'g', 'transport', 'cost_matrix', 'gw_cost',
                                'reg_gw_cost', 'reg_gw_cost_arr',
                                'errors_sinkhorn', 'converged_sinkhorn'])


def sinkhornsol_to_gwsol(out: sinkhorn.SinkhornState,
                         transport, cost_matrix, gw_cost,
                         reg_gw_cost_arr, errors, converged):
  return GromovWassersteinOutput(
      out.f, out.g, transport, cost_matrix, gw_cost,
      out.reg_ot_cost, reg_gw_cost_arr, errors, converged)


def gwsol_to_sinkhornsol(out: GromovWassersteinOutput):
  return sinkhorn.SinkhornState(
      out.f, out.g, out.reg_gw_cost, None)


@jax.tree_util.register_pytree_node_class
class GromovWasserstein:
  """A Gromov Wasserstein solver."""

  def __init__(self,
               epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.0,
               min_iterations: int = 5,
               max_iterations: int = 50,
               threshold: float = 1e-3,
               jit: bool = True,
               warm_start: bool = False,
               store_sinkhorn_errors = False,
               linear_ot_solver: sinkhorn.Sinkhorn = sinkhorn.Sinkhorn(),
               **kwargs):
    self.epsilon = epsilon
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.threshold = threshold
    self.jit = jit
    self.warm_start = warm_start
    self.store_sinkhorn_errors = store_sinkhorn_errors
    self.linear_ot_solver = linear_ot_solver
    self._kwargs = kwargs
    self._make_geom_fn = functools.partial(
        geometry.Geometry, epsilon=epsilon, **kwargs)

  def tree_flatten(self):
    return ([self.epsilon, self.linear_ot_solver, self.threshold],
            dict(
                min_iterations=self.min_iterations,
                max_iterations=self.max_iterations,
                jit=self.jit,
                warm_start=self.warm_start,
                store_sinkhorn_errors=self.store_sinkhorn_errors,
                **self._kwargs))

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(epsilon=children[0],
               linear_ot_solver=children[1],
               threshold=children[2],
               **aux_data)

  def not_converged(self, state, iteration):
    objs = state.reg_gw_cost_arr
    i = iteration
    tol = self.threshold
    return jnp.logical_or(
        i <= 2,
        jnp.logical_and(
            jnp.isfinite(objs[i - 1]),
            jnp.logical_not(jnp.isclose(objs[i - 2], objs[i - 1], rtol=tol))))

  def init_linearization(
      self, prob: problems.QuadraticProblem) -> problems.LinearProblem:
    """Initialises the cost matrix for the geometry object for GW.

    The equation follows Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf.

    Args:
      prob: a quadratic problem.

    Returns:
      A LinearProblem, representing local linearization of GW problem.
    """
    a = jax.lax.stop_gradient(prob.a)
    b = jax.lax.stop_gradient(prob.b)
    # TODO(oliviert, cuturi): consider passing a custom initialization.
    ab = a[:, None] * b[None, :]
    marginal_1 = ab.sum(1)
    marginal_2 = ab.sum(0)
    marginal_term = prob.marginal_dependent_cost(marginal_1, marginal_2)
    tmp = prob.geom_1.apply_cost(ab, axis=1, fn=prob.quad_loss[0])
    cost_matrix = marginal_term - prob.geom_2.apply_cost(
        tmp.T, axis=1, fn=prob.quad_loss[1]).T
    return problems.LinearProblem(
        self._make_geom_fn(cost_matrix=cost_matrix), prob.a, prob.b)

  def update_linearization(self,
                           prob: problems.QuadraticProblem,
                           linearization: problems.LinearProblem,
                           gw_solution: sinkhorn.SinkhornState
                           ) -> problems.LinearProblem:
    """Updates linearization of GW problem by updating cost matrix.

    The cost matrix equation follows Equation 6, Proposition 1 of
    http://proceedings.mlr.press/v48/peyre16.pdf.

    Let :math:`p` [num_a,] be the marginal of the transport matrix for samples
    from geom_x and :math:`q` [num_b,] be the marginal of the transport matrix
    for samples from geom_y. Let :math:`T` [num_a, num_b] be the transport
    matrix. The cost matrix equation can be written as:

    cost_matrix = marginal_dep_term
                + left_x(cost_x) :math:`T` right_y(cost_y):math:`^T`

    Args:
      prob: the quadratic problem.
      linearization: a LinearProblem object, linearizing the quadratic one.
      gw_solution: solution of the linearization of the quadratic problem.

    Returns:
      Updated linear OT problem, a new local linearization of GW problem.
    """
    geom, f, g = linearization.geom, gw_solution.f, gw_solution.g
    # Computes tmp = cost_matrix_x * transport
    # When the transport can be instantiated and a low rank structure
    # of the cost can be taken advantage of, it is preferable to do the product
    # between transport and cost matrix by instantiating first the transport
    # and applying the cost to it on the left.
    # TODO(cuturi,oliviert,lpapaxanthos): handle online & sqEuc geom_1 better
    if not prob.geom_1.is_online or prob.geom_1.is_squared_euclidean:
      transport = geom.transport_from_potentials(f, g)
      tmp = prob.geom_1.apply_cost(transport, axis=1, fn=prob.quad_loss[0])
    else:
      # When on the contrary the transport is difficult to instantiate
      # we default back on the application of the transport to the cost matrix.
      tmp = geom.apply_transport_from_potentials(
          f, g, prob.quad_loss[0](prob.geom_1.cost_matrix), axis=0)

    marginal_1 = geom.marginal_from_potentials(f, g, axis=1)
    marginal_2 = geom.marginal_from_potentials(f, g, axis=0)
    marginal_term = prob.marginal_dependent_cost(marginal_1, marginal_2)
    # TODO(cuturi,oliviert,lpapaxanthos): handle low rank products for geom_2's.
    cost_matrix = marginal_term - prob.geom_2.apply_cost(
        tmp.T, axis=1, fn=prob.quad_loss[1]).T
    return problems.LinearProblem(
        self._make_geom_fn(cost_matrix=cost_matrix), prob.a, prob.b)

  def __call__(
      self, prob: problems.QuadraticProblem) -> GromovWassersteinOutput:
    if not prob.is_balanced:
      raise ValueError('Unbalanced Gromov-Wasserstein is not supported yet.')

    gromov_fn = jax.jit(iterations) if self.jit else iterations
    linearization, linear_sol = gromov_fn(self, prob)
    f, g = linear_sol.f, linear_sol.g

    # TODO(lpapaxanthos): remove stop_gradient when using backprop
    linearization = self.update_linearization(
        prob,
        jax.lax.stop_gradient(linearization),
        jax.lax.stop_gradient(linear_sol))
    geom_gw = linearization.geom
    transport = geom_gw.transport_from_potentials(f, g)
    cost_matrix = 0.5 * geom_gw.cost_matrix
    gw_cost = jnp.sum(transport * cost_matrix)
    reg_gw_cost = linearization.ent_reg_cost(
        jax.lax.stop_gradient(f), jax.lax.stop_gradient(g), lse_mode=True)
    # TODO(oliviert): revisit the definition of the GW output.
    return GromovWassersteinOutput(
        f, g, transport, cost_matrix, gw_cost, reg_gw_cost,
        linear_sol.reg_gw_cost_arr, linear_sol.errors_sinkhorn,
        linear_sol.converged_sinkhorn)


def iterations(solver: GromovWasserstein, prob: problems.QuadraticProblem):
  """A jittable Gromov-Wasserstein outer loop."""
  lse_mode = solver.linear_ot_solver.lse_mode

  def cond_fn(iteration, constants, state):
    solver = constants
    _, gw_solution = state
    return solver.not_converged(gw_solution, iteration)

  def body_fn(iteration, constants, state, compute_error):
    solver = constants
    del compute_error  # Always assumed True for outer loop of GW.
    linear_pb, gw_solution = state
    sinkhorn_solution = gwsol_to_sinkhornsol(gw_solution)
    linear_pb = solver.update_linearization(prob, linear_pb, sinkhorn_solution)
    f, g = sinkhorn_solution.f, gw_solution.g
    if solver.warm_start:
      init_dual_a = f if lse_mode else linear_pb.geom.scaling_from_potential(f)
      init_dual_b = g if lse_mode else linear_pb.geom.scaling_from_potential(g)
    else:
      init_dual_a, init_dual_b = None, None
    out = solver.linear_ot_solver(linear_pb, init_dual_a, init_dual_b)
    reg_gw_cost_arr = gw_solution.reg_gw_cost_arr
    reg_gw_cost_arr = reg_gw_cost_arr.at[iteration].set(out.reg_ot_cost)


    errors_sinkhorn = gw_solution.errors_sinkhorn
    if solver.store_sinkhorn_errors:
      errors_sinkhorn = errors_sinkhorn.at[iteration, :].set(out.errors)

    converged_sinkhorn = gw_solution.converged_sinkhorn
    converged_sinkhorn = converged_sinkhorn.at[iteration].set(out.converged)
    out = sinkhornsol_to_gwsol(
        out, None, None, None,
        reg_gw_cost_arr, errors_sinkhorn, converged_sinkhorn)
    return linear_pb, out

  linearization = solver.init_linearization(prob)
  sinkhorn_solution = solver.linear_ot_solver(linearization)
  # initialize the GW State
  gw_solution = sinkhornsol_to_gwsol(
      sinkhorn_solution,
      None, None, None,
      jnp.zeros((solver.max_iterations,)),
      None if not solver.store_sinkhorn_errors else jnp.zeros(
          (solver.max_iterations, solver.linear_ot_solver.outer_iterations)),
      jnp.zeros((solver.max_iterations,)))

  (linearization, gw_solution) = fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=solver,
      state=(linearization, gw_solution))
  return linearization, gw_solution


def gromov_wasserstein(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.,
    loss: str = 'sqeucl',
    max_iterations: int = 50,
    jit: bool = False,
    warm_start: bool = True,
    store_sinkhorn_errors: bool = False,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    threshold: float = 1e-2,
    min_iterations: int = 1,
    **kwargs) -> GromovWassersteinOutput:
  """For backward compatibility."""
  sinkhorn_kwargs = {} if sinkhorn_kwargs is None else sinkhorn_kwargs
  sink = sinkhorn.make(**sinkhorn_kwargs)
  solver = GromovWasserstein(
      epsilon, max_iterations=max_iterations, warm_start=warm_start,
      jit=jit, linear_ot_solver=sink, threshold=threshold,
      store_sinkhorn_errors=store_sinkhorn_errors,
      min_iterations=min_iterations, **kwargs)
  losses = {'sqeucl': problems.make_square_loss, 'kl': problems.make_kl_loss}
  loss_fn = losses.get(loss, None)
  tau_a = sinkhorn_kwargs.get('tau_a', 1.0)  # For backward compatibility.
  tau_b = sinkhorn_kwargs.get('tau_b', 1.0)  # For backward compatibility.
  prob = problems.QuadraticProblem(
      geom_x, geom_y, a=a, b=b, loss=loss_fn(), tau_a=tau_a, tau_b=tau_b)
  return solver(prob)
