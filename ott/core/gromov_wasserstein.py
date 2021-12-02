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
from typing import Optional, Union, Any, Dict

import jax
import jax.numpy as jnp
from ott.core import problems
from ott.core import sinkhorn
from ott.geometry import epsilon_scheduler
from ott.geometry import geometry


GromovWassersteinOutput = collections.namedtuple(
    'GromovWassersteinOutput', ['f', 'g', 'transport', 'cost_matrix', 'gw_cost',
                                'reg_gw_cost', 'reg_gw_cost_arr',
                                'errors_sinkhorn', 'converged_sinkhorn'])


@jax.tree_util.register_pytree_node_class
class GromovWasserstein:
  """A Gromov Wasserstein solver."""

  def __init__(self,
               epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.0,
               max_iterations: int = 20,
               jit: bool = True,
               warm_start: bool = False,
               linear_ot_solver: sinkhorn.Sinkhorn = sinkhorn.Sinkhorn(),
               **kwargs):
    self.epsilon = epsilon
    self.max_iterations = max_iterations
    self.warm_start = warm_start
    self.linear_ot_solver = linear_ot_solver
    self.jit = jit
    self._kwargs = kwargs
    self._make_geom_fn = functools.partial(
        geometry.Geometry, epsilon=epsilon, **kwargs)

  def tree_flatten(self):
    return ([self.epsilon, self.linear_ot_solver],
            dict(max_iterations=self.max_iterations,
                 jit=self.jit,
                 warm_start=self.warm_start, **self._kwargs)
            )

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(epsilon=children[0], linear_ot_solver=children[1], **aux_data)

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
                           linear_solution: sinkhorn.SinkhornOutput
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
    geom, f, g = linearization.geom, linear_solution.f, linear_solution.g
    # Computes tmp = cost_matrix_x * transport
    # When the transport can be instantiated and a low rank structure
    # of the cost can be taken advantage of, it is preferable to do the product
    # between transport and cost matrix by instantiating first the transport
    # and applying the cost to it on the left.
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
    cost_matrix = marginal_term - prob.geom_2.apply_cost(
        tmp.T, axis=1, fn=prob.quad_loss[1]).T
    return problems.LinearProblem(
        self._make_geom_fn(cost_matrix=cost_matrix), prob.a, prob.b)

  def __call__(
      self, prob: problems.QuadraticProblem) -> GromovWassersteinOutput:
    if not prob.is_balanced:
      raise ValueError('Unbalanced Gromov-Wasserstein is not supported yet.')

    gromov_fn = jax.jit(iterations) if self.jit else iterations
    linearization, linear_sol, (cost, errors, converged) = gromov_fn(self, prob)
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
        f, g, transport, cost_matrix, gw_cost,
        reg_gw_cost, cost, errors, converged)


def iterations(solver: GromovWasserstein, prob: problems.QuadraticProblem):
  """A jittable Gromov-Wasserstein outer loop."""
  lse_mode = solver.linear_ot_solver.lse_mode

  def body_fn(carry, x=None):
    del x
    linear_pb, linear_solution = carry
    linear_pb = solver.update_linearization(prob, linear_pb, linear_solution)
    f, g = linear_solution.f, linear_solution.g
    if solver.warm_start:
      init_dual_a = f if lse_mode else linear_pb.geom.scaling_from_potential(f)
      init_dual_b = g if lse_mode else linear_pb.geom.scaling_from_potential(g)
    else:
      init_dual_a, init_dual_b = None, None
    out = solver.linear_ot_solver(linear_pb, init_dual_a, init_dual_b)
    return (linear_pb, out), (out.reg_ot_cost, out.errors, out.converged)

  # TODO(oliviert): use fixed_point_loop instead.
  linearization = solver.init_linearization(prob)
  linear_solution = solver.linear_ot_solver(linearization)
  carry = linearization, linear_solution
  (linearization, linear_solution), out = jax.lax.scan(
      f=body_fn, init=carry, xs=None, length=solver.max_iterations - 1)
  return linearization, linear_solution, out


def gromov_wasserstein(
    geom_x: geometry.Geometry,
    geom_y: geometry.Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    epsilon: Union[epsilon_scheduler.Epsilon, float] = 1.,
    loss: str = 'sqeucl',
    max_iterations: int = 20,
    jit: bool = False,
    warm_start: bool = True,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs) -> GromovWassersteinOutput:
  """For backward compatibility."""
  sinkhorn_kwargs = {} if sinkhorn_kwargs is None else sinkhorn_kwargs
  sink = sinkhorn.make(**sinkhorn_kwargs)
  solver = GromovWasserstein(
      epsilon, max_iterations=max_iterations, warm_start=warm_start,
      jit=jit, linear_ot_solver=sink, **kwargs)
  losses = {'sqeucl': problems.make_square_loss, 'kl': problems.make_kl_loss}
  loss_fn = losses.get(loss, None)
  tau_a = sinkhorn_kwargs.get('tau_a', 1.0)  # For backward compatibility.
  tau_b = sinkhorn_kwargs.get('tau_b', 1.0)  # For backward compatibility.
  prob = problems.QuadraticProblem(
      geom_x, geom_y, a=a, b=b, loss=loss_fn(), tau_a=tau_a, tau_b=tau_b)
  return solver(prob)
