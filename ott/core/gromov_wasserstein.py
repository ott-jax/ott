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
import functools
from typing import Any, Dict, Optional, NamedTuple, Union

import jax
import jax.numpy as jnp
from ott.core import fixed_point_loop
from ott.core import problems
from ott.core import sinkhorn
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry


class GWState(NamedTuple):
  """Holds the state of the Gromov-Wasserstein solver.

  Attributes:
    cost: Holds the sequence of regularized GW costs seen through the outer loop
      of the solver.
    convergence: Holds the sequence of bool convergence flags of the inner
      Sinkhorn iterations.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm at
      each iteration.
    linear_state: state used to solve and store solutions to the local
      linearization of GW.
    linear_pb: local linearizationg of the quadratic GW problem.
    geom: the geometry underlying the local linearization.
    transport: the transport matrix.
    reg_gw_cost: regularized optimal transport cost of the linearization.
  """
  costs: Optional[jnp.ndarray] = None
  convergence: Optional[jnp.ndarray] = None
  errors: Optional[jnp.ndarray] = None
  linear_state: Any = None
  linear_pb: Optional[problems.LinearProblem] = None

  def set(self, **kwargs) -> 'GWState':
    """Returns a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(self, iteration: int, linear_sol, linear_pb, store_errors: bool):
    costs = self.costs.at[iteration].set(linear_sol.reg_ot_cost)
    errors = None
    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :].set(linear_sol.errors)
    convergence = self.convergence.at[iteration].set(linear_sol.converged)
    return self.set(linear_state=linear_sol, linear_pb=linear_pb,
                    costs=costs, convergence=convergence, errors=errors)

  @property
  def geom(self):
    return self.linear_pb.geom

  @property
  def transport(self):
    return self.linear_state.matrix(self.geom)

  @property
  def reg_gw_cost(self):
    return self.linear_state.reg_ot_cost


@jax.tree_util.register_pytree_node_class
class GromovWasserstein:
  """A Gromov Wasserstein solver."""

  def __init__(self,
               epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.0,
               min_iterations: int = 5,
               max_iterations: int = 50,
               threshold: float = 1e-3,
               jit: bool = True,
               store_sinkhorn_errors: bool = False,
               linear_ot_solver: sinkhorn.Sinkhorn = sinkhorn.Sinkhorn(),
               **kwargs):
    self.epsilon = epsilon
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.threshold = threshold
    self.jit = jit
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
                store_sinkhorn_errors=self.store_sinkhorn_errors,
                **self._kwargs))

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(epsilon=children[0],
               linear_ot_solver=children[1],
               threshold=children[2],
               **aux_data)

  def not_converged(self, state, iteration):
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_or(
        i <= 2,
        jnp.logical_and(
            jnp.isfinite(costs[i - 1]),
            jnp.logical_not(jnp.isclose(costs[i - 2], costs[i - 1], rtol=tol))))

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
                           linear_solution: sinkhorn.SinkhornState
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
      linear_solution: solution of the linearization of the quadratic problem.

    Returns:
      Updated linear OT problem, a new local linearization of GW problem.
    """
    geom = linearization.geom
    # Computes tmp = cost_matrix_x * transport
    # When the transport can be instantiated and a low rank structure
    # of the cost can be taken advantage of, it is preferable to do the product
    # between transport and cost matrix by instantiating first the transport
    # and applying the cost to it on the left.
    # TODO(cuturi,oliviert,lpapaxanthos): handle online & sqEuc geom_1 better
    if not prob.geom_1.is_online or prob.geom_1.is_squared_euclidean:
      transport = linear_solution.matrix(geom)
      tmp = prob.geom_1.apply_cost(transport, axis=1, fn=prob.quad_loss[0])
    else:
      # When on the contrary the transport is difficult to instantiate
      # we default back on the application of the transport to the cost matrix.
      tmp = linear_solution.apply(
          geom, prob.quad_loss[0](prob.geom_1.cost_matrix), axis=0)

    marginal_1 = linear_solution.marginal(geom, axis=1)
    marginal_2 = linear_solution.marginal(geom, axis=0)
    marginal_term = prob.marginal_dependent_cost(marginal_1, marginal_2)
    # TODO(cuturi,oliviert,lpapaxanthos): handle low rank products for geom_2's.
    cost_matrix = marginal_term - prob.geom_2.apply_cost(
        tmp.T, axis=1, fn=prob.quad_loss[1]).T
    return problems.LinearProblem(
        self._make_geom_fn(cost_matrix=cost_matrix), prob.a, prob.b)

  def __call__(self, prob: problems.QuadraticProblem) -> GWState:
    if not prob.is_balanced:
      raise ValueError('Unbalanced Gromov-Wasserstein is not supported yet.')

    gromov_fn = jax.jit(iterations) if self.jit else iterations
    state = gromov_fn(self, prob)
    # # TODO(lpapaxanthos): remove stop_gradient when using backprop
    linearization = self.update_linearization(
        prob,
        jax.lax.stop_gradient(state.linear_pb),
        jax.lax.stop_gradient(state.linear_state))
    linear_state = state.linear_state.set_cost(linearization, True, True)
    return state.set(linear_pb=linearization, linear_state=linear_state)

  def init_state(self, prob: problems.QuadraticProblem) -> GWState:
    """Initializes the state of the Gromov-Wasserstein iterations."""
    linearization = self.init_linearization(prob)
    linear_state = self.linear_ot_solver(linearization)
    num_iter = self.max_iterations
    if self.store_sinkhorn_errors:
      errors = jnp.zeros((num_iter, self.linear_ot_solver.outer_iterations))
    else:
      errors = None
    return GWState(jnp.zeros((num_iter,)), jnp.zeros((num_iter,)), errors,
                   linear_state, linearization)


def iterations(solver: GromovWasserstein,
               prob: problems.QuadraticProblem) -> GWState:
  """A jittable Gromov-Wasserstein outer loop."""

  def cond_fn(iteration, constants, state):
    solver = constants
    return solver.not_converged(state, iteration)

  def body_fn(iteration, constants, state, compute_error):
    del compute_error  # Always assumed True for outer loop of GW.
    solver = constants
    linear_pb = solver.update_linearization(
        prob, state.linear_pb, state.linear_state)
    out = solver.linear_ot_solver(linear_pb)
    return state.update(iteration, out, linear_pb, solver.store_sinkhorn_errors)

  return fixed_point_loop.fixpoint_iter(
      cond_fn=cond_fn,
      body_fn=body_fn,
      min_iterations=solver.min_iterations,
      max_iterations=solver.max_iterations,
      inner_iterations=1,
      constants=solver,
      state=solver.init_state(prob))


def make(
    epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.,
    max_iterations: int = 50,
    jit: bool = False,
    warm_start: bool = True,
    store_sinkhorn_errors: bool = False,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    threshold: float = 1e-2,
    min_iterations: int = 1,
    **kwargs) -> GromovWasserstein:
  """Creates a GromovWasserstein solver.

  Args:
    epsilon: a regularization parameter or a epsilon_scheduler.Epsilon object.
    max_iterations: int32, the maximum number of outer iterations for
      Gromov Wasserstein.
    jit: bool, if True, jits the function.
    warm_start: deprecated.
    store_sinkhorn_errors: whether or not to return all the errors of the inner
      Sinkhorn iterations.
    sinkhorn_kwargs: Optionally a dictionary containing the keywords arguments
     for calls to the sinkhorn function.
    threshold: threshold (progress between two iterate costs) used to stop GW.
    min_iterations: see fixed_point_loop.
    **kwargs: additional kwargs for epsilon.

  Returns:
    A GromovWasserstein solver.
  """
  del warm_start
  sinkhorn_kwargs = {} if sinkhorn_kwargs is None else sinkhorn_kwargs
  sink = sinkhorn.make(**sinkhorn_kwargs)
  return GromovWasserstein(
      epsilon, max_iterations=max_iterations,
      jit=jit, linear_ot_solver=sink, threshold=threshold,
      store_sinkhorn_errors=store_sinkhorn_errors,
      min_iterations=min_iterations, **kwargs)


def gromov_wasserstein(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    loss: str = 'sqeucl',
    **kwargs) -> GWState:
  """Fits Gromov Wasserstein.

  Args:
    geom_x: a Geometry object for the first view.
    geom_y: a second Geometry object for the second view.
    a: jnp.ndarray<float>[num_a,] or jnp.ndarray<float>[batch,num_a] weights.
    b: jnp.ndarray<float>[num_b,] or jnp.ndarray<float>[batch,num_b] weights.
    loss: str 'sqeucl' or 'kl' to define the GW loss.
    **kwargs: keyword arguments to make.

  Returns:
    A GromovWassersteinState named tuple.
  """
  losses = {'sqeucl': problems.make_square_loss, 'kl': problems.make_kl_loss}
  loss_fn = losses.get(loss, None)
  prob = problems.QuadraticProblem(geom_x, geom_y, a=a, b=b, loss=loss_fn())
  solver = make(**kwargs)
  return solver(prob)
