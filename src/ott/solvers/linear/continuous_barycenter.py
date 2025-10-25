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
import functools
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott import utils
from ott.geometry import pointcloud
from ott.math import fixed_point_loop
from ott.math import utils as mu
from ott.problems.linear import barycenter_problem, linear_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn, sinkhorn_lr

__all__ = ["FreeBarycenterState", "FreeWassersteinBarycenter"]

LinearOutput = Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput]


class FreeBarycenterState(NamedTuple):
  """Holds the state of the Wasserstein barycenter solver.

  Args:
    x: Barycenter points.
    a: Barycenter weights.
    costs: Holds the sequence of regularized costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    linear_outputs: Holds latest output objects returned by linear solver when
      recomputing transport between barycenter vs. each of the input measures.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
  """

  x: jnp.ndarray
  a: jnp.ndarray
  costs: Optional[jnp.ndarray] = None
  linear_convergence: Optional[jnp.ndarray] = None
  linear_outputs: Optional[LinearOutput] = None
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "FreeBarycenterState":
    """Return a copy of self, possibly with overwrites."""
    return self._replace(**kwargs)

  def update(
      self, iteration: int, bar_prob: barycenter_problem.FreeBarycenterProblem,
      linear_solver: Any, store_errors: bool
  ) -> "FreeBarycenterState":
    """Update the state of the solver.

    Args:
      iteration: the current iteration of the outer loop.
      bar_prob: the barycenter problem.
      linear_solver: the linear OT solver to use.
      store_errors: whether to store the errors of the inner loop.

    Returns:
      The updated state.
    """
    seg_y, seg_b, _ = bar_prob.segmented_y_b

    @functools.partial(jax.vmap, in_axes=[None, None, 0, 0])
    def solve_linear_ot(
        a: Optional[jnp.ndarray], x: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray
    ):
      geom = pointcloud.PointCloud(
          x, y, cost_fn=bar_prob.cost_fn, epsilon=bar_prob.epsilon
      )
      prob = linear_problem.LinearProblem(geom, a=a, b=b)
      out = linear_solver(prob)
      # instantiate matrix since it is a property of out.
      return out, out.matrix

    outs, matrices = solve_linear_ot(self.a, self.x, seg_b, seg_y)
    reg_ot_costs = outs.reg_ot_cost
    convergeds = outs.converged
    errors = outs.errors

    cost = jnp.sum(reg_ot_costs * bar_prob.weights)
    updated_costs = self.costs.at[iteration].set(cost)
    converged = jnp.all(convergeds)
    linear_convergence = self.linear_convergence.at[iteration].set(converged)

    if store_errors and self.errors is not None:
      errors = self.errors.at[iteration, :, :].set(errors)
    else:
      errors = None

    # Approximation of barycenter as barycenter of barycenters per measure.

    barycenters_per_measure = mu.barycentric_projection(
        matrices, seg_y, bar_prob.cost_fn
    )

    x_new = jax.vmap(
        lambda w, y: bar_prob.cost_fn.barycenter(w, y)[0], in_axes=[None, 1]
    )(bar_prob.weights, barycenters_per_measure)

    return self.set(
        x=x_new,
        costs=updated_costs,
        linear_convergence=linear_convergence,
        linear_outputs=outs,
        errors=errors
    )


class FreeBarycenterOutput(NamedTuple):
  """Holds the output of a Free Wasserstein barycenter solver.

  Objects of this class contain both solutions and problem definition of a
  regularized Free Barycenter OT problem, along several methods that can be used
  to access its content.

  Args:
    x: barycenter points.
    a: barycenter weights.
    costs: Holds the sequence of regularized GW costs seen through the outer
      loop of the solver.
    linear_convergence: Holds the sequence of bool convergence flags of the
      inner Sinkhorn iterations.
    linear_outputs: Holds latest output objects returned by linear solver when
      recomputing transport between barycenter vs. each of the input measures.
    errors: Holds sequence of vectors of errors of the Sinkhorn algorithm
      at each iteration.
  """

  x: jnp.ndarray
  a: jnp.ndarray
  bar_prob: barycenter_problem.FreeBarycenterProblem
  costs: jnp.ndarray
  linear_convergence: jnp.ndarray
  linear_outputs: LinearOutput
  errors: Optional[jnp.ndarray] = None

  @property
  def all_linear_solvers_converged(self) -> bool:
    """Whether all linear convergence flags converged."""
    return jnp.all(self.linear_convergence[self.linear_convergence != -1])

  def matrix_at_index(self, measure_index: int) -> jnp.ndarray:
    """Return the transport matrix from barycenter to measure_index measure."""
    size_measure = self.bar_prob.num_per_measure[measure_index]
    matrix = self.linear_output_at_index(measure_index).matrix
    return jax.lax.dynamic_slice_in_dim(matrix, 0, size_measure, axis=-1)

  @property
  def bar_size(self) -> int:
    """Size of the barycenter."""
    return self.x.shape[0]

  @property
  def num_iters(self) -> int:
    """Number of outer iterations performed to converge."""
    return jnp.sum(self.linear_convergence != -1)

  @property
  def costs_along_iterations(self) -> jnp.ndarray:
    """Costs vector with superfluous values removed."""
    return self.costs[:self.num_iters]

  def linear_output_at_index(self, i: int) -> LinearOutput:
    """Linear solver output of transport from barycenter to measure i."""
    return jax.tree.map(lambda x: x[i], self.linear_outputs)


@jax.tree_util.register_pytree_node_class
class FreeWassersteinBarycenter(was_solver.WassersteinSolver):
  """Continuous Wasserstein barycenter solver :cite:`cuturi:14`."""

  def __call__(  # noqa: D102
      self,
      bar_prob: barycenter_problem.FreeBarycenterProblem,
      bar_size: int = 100,
      x_init: Optional[jnp.ndarray] = None,
      rng: Optional[jax.Array] = None,
  ) -> FreeBarycenterState:
    rng = utils.default_prng_key(rng)
    return self.iterations(bar_size, bar_prob, x_init, rng)

  def init_state(
      self,
      bar_prob: barycenter_problem.FreeBarycenterProblem,
      bar_size: int,
      x_init: Optional[jnp.ndarray] = None,
      rng: Optional[jax.Array] = None,
  ) -> FreeBarycenterState:
    """Initialize the state of the Wasserstein barycenter iterations.

    Args:
      bar_prob: The barycenter problem.
      bar_size: Size of the barycenter.
      x_init: Initial barycenter estimate of shape ``[bar_size, ndim]``.
        If `None`, ``bar_size`` points will be sampled from the input
        measures according to their weights
        :attr:`~ott.problems.linear.barycenter_problem.FreeBarycenterProblem.flattened_y`.
      rng: Random key for seeding.

    Returns:
      The initial barycenter state.
    """
    if x_init is not None:
      assert bar_size == x_init.shape[0]
      x = x_init
    else:
      # sample randomly points in the support of the y measures
      rng = utils.default_prng_key(rng)
      indices_subset = jax.random.choice(
          rng,
          a=bar_prob.flattened_y.shape[0],
          shape=(bar_size,),
          replace=False,
          p=bar_prob.flattened_b
      )
      x = bar_prob.flattened_y[indices_subset, :]

    # TODO(cuturi) expand to non-uniform weights for barycenter.
    a = jnp.ones((bar_size,)) / bar_size
    num_iter = self.max_iterations
    if self.store_inner_errors:
      errors = -jnp.ones((
          num_iter, bar_prob.num_measures, self.linear_solver.outer_iterations
      ))
    else:
      errors = None
    state = FreeBarycenterState(
        x=x,
        a=a,
        costs=-jnp.ones((num_iter,)),
        linear_convergence=-jnp.ones((num_iter,)),
        errors=errors
    )
    abstract_tree = jax.eval_shape(
        functools.partial(state.update, store_errors=self.store_inner_errors),
        0, bar_prob, self.linear_solver
    )

    linear_outputs = jax.tree.map(jnp.zeros_like, abstract_tree.linear_outputs)

    return FreeBarycenterState(
        x=x,
        a=a,
        costs=-jnp.ones((num_iter,)),
        linear_convergence=-jnp.ones((num_iter,)),
        linear_outputs=linear_outputs,
        errors=errors
    )

  def output_from_state(  # noqa: D102
      self,
      state: FreeBarycenterState,
      bar_prob: barycenter_problem.FreeBarycenterProblem
  ) -> FreeBarycenterOutput:
    """Create an output from a barycenter state."""
    return FreeBarycenterOutput(
        x=state.x,
        a=state.a,
        bar_prob=bar_prob,
        costs=state.costs,
        linear_convergence=state.linear_convergence,
        linear_outputs=state.linear_outputs,
        errors=state.errors,
    )

  def iterations(
      self, bar_size: int, bar_prob: barycenter_problem.FreeBarycenterProblem,
      x_init: jnp.ndarray, rng: jax.Array
  ) -> FreeBarycenterState:
    """Wasserstein barycenter outer loop."""

    def cond_fn(
        iteration: int,
        constants: Tuple[FreeWassersteinBarycenter,
                         barycenter_problem.FreeBarycenterProblem],
        state: FreeBarycenterState
    ) -> bool:
      return self._continue(state, iteration)

    def body_fn(
        iteration, constants: Tuple[FreeWassersteinBarycenter,
                                    barycenter_problem.FreeBarycenterProblem],
        state: FreeBarycenterState, compute_error: bool
    ) -> FreeBarycenterState:
      del compute_error  # Always assumed True
      bar_prob = constants
      return state.update(
          iteration, bar_prob, self.linear_solver, self.store_inner_errors
      )

    state = fixed_point_loop.fixpoint_iter(
        cond_fn=cond_fn,
        body_fn=body_fn,
        min_iterations=self.min_iterations,
        max_iterations=self.max_iterations,
        inner_iterations=1,
        constants=bar_prob,
        state=self.init_state(bar_prob, bar_size, x_init, rng)
    )

    return self.output_from_state(state, bar_prob)
