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
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ott.geometry import costs, pointcloud
from ott.math import fixed_point_loop
from ott.math import utils as mu

__all__ = ["MMSinkhornOutput", "MMSinkhorn"]


class MMSinkhornState(NamedTuple):
  potentials: Tuple[jnp.ndarray, ...]
  errors: jnp.ndarray

  def solution_error(
      self,
      cost_t: jnp.ndarray,
      a_s: Tuple[jnp.ndarray, ...],
      epsilon: float,
      norm_error: float = 1.0
  ) -> float:
    coupl_tensor = coupling_tensor(self.potentials, cost_t, epsilon)
    marginals = tensor_marginals(coupl_tensor)
    errors = jnp.array([
        jnp.sum(jnp.abs(a - marginal) ** norm_error) ** (1.0 / norm_error)
        for a, marginal in zip(a_s, marginals)
    ])
    return jnp.sum(errors)

  def set(self, **kwargs: Any) -> "MMSinkhornState":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)


class MMSinkhornOutput(NamedTuple):
  r"""Output of the MMSinkhorn solver used on :math:`k` point clouds.

  This class contains both solutions and problem definition of a regularized
  MM-OT problem involving :math:`k` weighted point clouds of varying sizes,
  along with methods and properties that can use or describe the solution.

  Args:
    potentials: Tuple of :math:`k` optimal dual variables, vectors of sizes
      equal to the number of points in each of the :math:`k` point clouds.
    errors: Vector of errors, along iterations. This vector is of size
      ``max_iterations // inner_iterations`` where those were the parameters
      passed on to the :class:`~ott.experimental.mmsinkhorn.MMSinkhorn` solver.
      Follows the conventions used in
      :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.errors`
    x_s: Tuple of :math:`k` point clouds, ``x_s[i]`` is a matrix of size
      :math:`n_i \times d` where `d` is common to all point clouds.
    a_s: Tuple of :math:`k` probability vectors, each of size :math:`n_i`.
    cost_fns: Cost function, or a tuple of :math:`k(k-1)/2` such instances.
    epsilon: Entropic regularization used to solve the multimarginal Sinkhorn
      problem.
    ent_reg_cost: The regularized optimal transport cost, the linear
      contribution (dot product between optimal tensor and cost) minus entropy
      times ``epsilon``.
    threshold: Convergence threshold used to control the termination of the
      algorithm.
    converged: Whether the output corresponds to a solution whose error is
      below the convergence threshold.
    inner_iterations: Number of iterations that were run between two
      computations of errors.
  """
  potentials: Tuple[jnp.ndarray, ...]
  errors: jnp.ndarray
  x_s: Optional[jnp.ndarray] = None
  a_s: Optional[Tuple[jnp.ndarray, ...]] = None
  cost_fns: Optional[Union[costs.CostFn, Tuple[costs.CostFn, ...]]] = None
  epsilon: Optional[float] = None
  ent_reg_cost: Optional[jnp.ndarray] = None
  threshold: Optional[jnp.ndarray] = None
  converged: Optional[bool] = None
  inner_iterations: Optional[int] = None

  def set(self, **kwargs: Any) -> "MMSinkhornOutput":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  @property
  def n_iters(self) -> int:  # noqa: D102
    """Total number of iterations that were needed to terminate."""
    return jnp.sum(self.errors != -1) * self.inner_iterations

  @property
  def cost_t(self) -> jnp.ndarray:
    """Cost tensor."""
    return cost_tensor(self.x_s, self.cost_fns)

  @property
  def tensor(self) -> jnp.ndarray:
    """Transport tensor."""
    return jnp.exp(
        -remove_tensor_sum(self.cost_t, self.potentials) / self.epsilon
    )

  @property
  def marginals(self) -> Tuple[jnp.ndarray, ...]:
    """:math:`k` marginal probability weight vectors."""
    return tensor_marginals(self.tensor)

  def marginal(self, k: int) -> jnp.ndarray:
    """Return the marginal probability weight vector at slice :math:`k`."""
    return tensor_marginal(self.tensor, k)

  @property
  def transport_mass(self) -> float:
    """Sum of transport tensor."""
    return jnp.sum(self.tensor)


def cost_tensor(
    x_s: Tuple[jnp.ndarray, ...], cost_fns: Union[costs.CostFn,
                                                  Tuple[costs.CostFn, ...]]
) -> jnp.ndarray:
  r"""Create a cost tensor from a tuple of :math:`k` :math:`d`-dim point clouds.

  Args:
    x_s: Tuple of :math:`k` point clouds, each described as a
      :math:`n_i \times d` matrix of batched vectors.
    cost_fns: Either a single :ott:`ott.geometry.costs.CostFn` object, or a
      tuple of :math:`k (k-1)/2` of them. Current implementation only works for
      symmetric and definite cost functions (i.e. such that
      :math:`c(x, y) = c(y, x)` and :math:`c(x, x) = 0`).
  """

  def c_fn_pair(i: int, j: int) -> costs.CostFn:
    if isinstance(cost_fns, costs.CostFn):
      return cost_fns
    return cost_fns[i * k - (i * (i + 1)) // 2 + j - i - 1]

  k = len(x_s)  # TODO(cuturi) padded version
  ns = [x.shape[0] for x in x_s]
  cost_t = jnp.zeros(ns)

  for i in range(k):
    for j in range(i + 1, k):
      cost_m = pointcloud.PointCloud(
          x_s[i], x_s[j], cost_fn=c_fn_pair(i, j)
      ).cost_matrix
      axis = list(range(i)) + list(range(i + 1, j)) + list(range(j + 1, k))
      cost_t += jnp.expand_dims(cost_m, axis=axis)
  return cost_t


def remove_tensor_sum(
    c: jnp.ndarray, u: Tuple[jnp.ndarray, ...]
) -> jnp.ndarray:
  r"""Remove the tensor sum of :math:`k` vectors to tensor of :math:`k` dims.

  Args:
    c: :math:`n_1 \times \cdots n_k` tensor.
    u: Tuple of :math:`k` vectors, each of size :math:`n_i`.

  Return:
    Tensor :math:`c - u[0] \oplus u[1] \oplus ... \oplus u[n]`.
  """
  k = len(u)
  for i in range(k):
    c -= jnp.expand_dims(u[i], axis=list(range(i)) + list(range(i + 1, k)))
  return c


def tensor_marginals(coupling: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
  return tuple(tensor_marginal(coupling, ix) for ix in range(coupling.ndim))


def tensor_marginal(coupling: jnp.ndarray, slice_index: int) -> jnp.ndarray:
  k = coupling.ndim
  axis = list(range(slice_index)) + list(range(slice_index + 1, k))
  return coupling.sum(axis=axis)


@jtu.register_pytree_node_class
class MMSinkhorn:
  r"""Multimarginal Sinkhorn solver, aligns :math:`k \,d`-dim point clouds.

  This solver implements the entropic multimarginal solver presented in
  :cite:`benamou:15` and described in :cite:`piran:24`, Algorithm 1.
  The current implementation follows largely the template of the
  :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver, with a much reduced
  set of hyperparameters, controlling the number of iterations and convergence
  threshold, along with the application of the :cite:`danskin:67` theorem to
  instantiate the OT cost. The iterations are done by default in log-space.

  Args:
    threshold: tolerance used to stop the Sinkhorn iterations. This is
      typically the deviation between a target marginal and the marginal of the
      current primal solution.
    norm_error: power used to define p-norm of error for marginal/target.
    inner_iterations: the Sinkhorn error is not recomputed at each
      iteration but every ``inner_iterations`` instead.
    min_iterations: the minimum number of Sinkhorn iterations carried
      out before the error is computed and monitored.
    max_iterations: the maximum number of Sinkhorn iterations. If
      ``max_iterations`` is equal to ``min_iterations``, Sinkhorn iterations are
      run by default using a :func:`~jax.lax.scan` loop rather than a custom,
      unroll-able :func:`~jax.lax.while_loop` that monitors convergence.
      In that case the error is not monitored and the ``converged``
      flag will return :obj:`False` as a consequence.
    use_danskin: when :obj:`True`, it is assumed the entropy regularized cost
      is evaluated using optimal potentials that are frozen, i.e. whose
      gradients have been stopped. This is useful when carrying out first order
      differentiation, and is only valid mathematically when the algorithm has
      converged with a low tolerance.
  """

  def __init__(
      self,
      threshold: float = 1e-3,
      norm_error: float = 1.0,
      inner_iterations: int = 10,
      min_iterations: int = 0,
      max_iterations: int = 2000,
      use_danskin: bool = True,
  ):
    self.threshold = threshold
    self.inner_iterations = inner_iterations
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.norm_error = norm_error
    self.use_danskin = use_danskin

  def __call__(
      self,
      x_s: Tuple[jnp.ndarray, ...],
      a_s: Optional[Tuple[jnp.ndarray, ...]] = None,
      cost_fns: Optional[Union[costs.CostFn, Tuple[costs.CostFn, ...]]] = None,
      epsilon: Optional[float] = None
  ) -> MMSinkhornOutput:
    r"""Solve multimarginal OT for :math:`k` :math:`d`-dim point clouds.

    Takes :math:`k` weighted :math:`d`-dim point clouds and computes their
    multimarginal optimal transport tensor. The :math:`d` dimensional point
    clouds are stored in ``x_s``, along with :math:`k` probability vectors,
    stored in ``a_s``, as well as a :class:`~ott.geometry.costs.CostFn`
    instance (or :math:`k(k-1)/2` of them, one for each pair of point clouds
    ``x_s[i]`` and ``x_s[j]``, ``i<j``.)

    The solver also uses ``epsilon`` as an input, with the default rule set to
    one twentieth of the mean of the cost tensor resulting from these inputs.

    Args:
      x_s: Tuple of :math:`k` point clouds, ``x_s[i]`` is a matrix of size
        :math:`n_i \times d` where :math:`d` is a dimension common to all
        point clouds.
      a_s: Tuple of :math:`k` probability vectors, each of size :math:`n_i`.
      cost_fns: Instance of :class:`~ott.geometry.costs.CostFn`, or a tuple
        of :math:`k(k-1)/2` such instances. Note that the solver currently
        assumes that these cost functions are symmetric. The cost function at
        index :math:`i(k-\tfrac{i+1}{2})+j-i-1` will be used to compare
        point cloud ``x_s[i]`` with point cloud ``x_s[j]``.
      epsilon: entropic regularization used to solve the multimarginal Sinkhorn
        problem.

    Returns:
      Multimarginal Sinkhorn output.
    """
    n_s = [x.shape[0] for x in x_s]
    if cost_fns is None:
      cost_fns = costs.SqEuclidean()
    elif isinstance(cost_fns, Tuple):
      assert len(cost_fns) == (len(n_s) * (len(n_s) - 1)) // 2

    # Default to uniform probability weights for each point cloud.
    if a_s is None:
      a_s = [jnp.ones(n) / n for n in n_s]
    else:
      # Case in which user passes ``None`` weights within tuple.
      a_s = [(jnp.ones(n) / n if a is None else a) for a, n in zip(a_s, n_s)]

    assert len(n_s) == len(a_s), (len(n_s), len(a_s))
    for n, a in zip(n_s, a_s):
      assert n == a.shape[0], (n, a.shape[0])

    cost_t = cost_tensor(x_s, cost_fns)
    state = self.init_state(n_s)
    epsilon = 0.05 * jnp.mean(cost_t) if epsilon is None else epsilon
    const = cost_t, a_s, epsilon
    out = run(const, self, state)
    return out.set(x_s=x_s, a_s=a_s, cost_fns=cost_fns, epsilon=epsilon)

  def init_state(self, n_s: Tuple[int, ...]) -> MMSinkhornState:
    """Return the initial state of the loop."""
    errors = -jnp.ones((self.outer_iterations, 1))
    potentials = tuple(jnp.zeros(n) for n in n_s)
    return MMSinkhornState(potentials=potentials, errors=errors)

  def _converged(self, state: MMSinkhornState, iteration: int) -> bool:
    err = state.errors[iteration // self.inner_iterations - 1, 0]
    return jnp.logical_and(iteration > 0, err < self.threshold)

  def _diverged(self, state: MMSinkhornState, iteration: int) -> bool:
    err = state.errors[iteration // self.inner_iterations - 1, 0]
    return jnp.logical_not(jnp.isfinite(err))

  def _continue(self, state: MMSinkhornState, iteration: int) -> bool:
    """Continue while not(converged) and not(diverged)."""
    return jnp.logical_and(
        jnp.logical_not(self._diverged(state, iteration)),
        jnp.logical_not(self._converged(state, iteration))
    )

  @property
  def outer_iterations(self) -> int:
    """Upper bound on number of times inner_iterations are carried out.

    This integer can be used to set constant array sizes to track the algorithm
    progress, notably errors.
    """
    return np.ceil(self.max_iterations / self.inner_iterations).astype(int)

  def tree_flatten(self):  # noqa: D102
    aux = vars(self).copy()
    aux.pop("threshold")
    return [self.threshold], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(**aux_data, threshold=children[0])


def run(
    const: Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...], float],
    solver: MMSinkhorn, state: MMSinkhornState
) -> MMSinkhornOutput:

  def cond_fn(
      iteration: int, const: Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...], float],
      state: MMSinkhornState
  ) -> bool:
    del const
    return solver._continue(state, iteration)

  def body_fn(
      iteration: int, const: Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...], float],
      state: MMSinkhornState, compute_error: bool
  ) -> MMSinkhornState:
    cost_t, a_s, epsilon = const
    k = len(a_s)

    def one_slice(potentials: Tuple[jnp.ndarray, ...], l: int, a: jnp.ndarray):
      pot = potentials[l]
      axis = list(range(l)) + list(range(l + 1, k))
      app_lse = mu.softmin(
          remove_tensor_sum(cost_t, potentials), epsilon, axis=axis
      )
      pot += epsilon * jnp.log(a) + jnp.where(jnp.isfinite(app_lse), app_lse, 0)
      return potentials[:l] + (pot,) + potentials[l + 1:]

    potentials = state.potentials
    for l in range(k):
      potentials = one_slice(potentials, l, a_s[l])

    state = state.set(potentials=potentials)
    err = jax.lax.cond(
        jnp.logical_or(
            iteration == solver.max_iterations - 1,
            jnp.logical_and(compute_error, iteration >= solver.min_iterations)
        ),
        lambda state, c, a, e: state.solution_error(c, a, e, solver.norm_error),
        lambda *_: jnp.inf, state, cost_t, a_s, epsilon
    )
    errors = state.errors.at[iteration // solver.inner_iterations, :].set(err)
    return state.set(errors=errors)

  fix_point = fixed_point_loop.fixpoint_iter_backprop
  state = fix_point(
      cond_fn, body_fn, solver.min_iterations, solver.max_iterations,
      solver.inner_iterations, const, state
  )
  converged = jnp.logical_and(
      jnp.logical_not(jnp.any(jnp.isnan(state.errors))), state.errors[-1, 0]
      < solver.threshold
  )

  out = MMSinkhornOutput(
      potentials=state.potentials,
      errors=state.errors,
      threshold=solver.threshold,
      converged=converged,
      inner_iterations=solver.inner_iterations
  )

  # Compute cost
  if solver.use_danskin:
    potentials = [jax.lax.stop_gradient(pot) for pot in out.potentials]
  else:
    potentials = out.potentials

  cost_t, a_s, epsilon = const
  ent_reg_cost = 0.0
  for potential, a in zip(potentials, a_s):
    pot = jnp.where(jnp.isfinite(potential), potential, 0)
    ent_reg_cost += jnp.sum(pot * a)

  ent_reg_cost += epsilon * (
      1 - jnp.sum(coupling_tensor(potentials, cost_t, epsilon))
  )
  return out.set(ent_reg_cost=ent_reg_cost)


def coupling_tensor(
    potentials: Tuple[jnp.ndarray], cost_t: jnp.ndarray, epsilon: float
) -> jnp.ndarray:
  return jnp.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)
