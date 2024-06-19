from typing import Any, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.math import fixed_point_loop
from ott.math import utils as mu
from ott.solvers.linear import sinkhorn

__all__ = ["MMSinkhornOutput", "MMSinkhorn"]


class MMSinkhornState(sinkhorn.SinkhornState):

  def solution_error(self, cost_t, a_s, epsilon):
    coupl_tensor = coupling_tensor(self.potentials, cost_t, epsilon)
    marginals = tensor_marginals(coupl_tensor)
    errors = jnp.array([
        jnp.sum(jnp.abs(a - marginal)) for a, marginal in zip(a_s, marginals)
    ])
    return jnp.sum(errors)


class MMSinkhornOutput(NamedTuple):
  r"""Holds the output of a MMSinkhorn solver used on :math:`k` point clouds.

  Objects of this class contain both solutions and problem definition of a
  regularized MM-OT problem involving `k` weighted point clouds of varying size,
  along with methods or properties that can use or describe the solution.

  Args:
    potentials: Tuple of `k` optimal dual variables, vectors of sizes equal to
      the number of points in each of the `k` point clouds.
    errors: vector or errors, along iterations. This vector is of size
      ``max_iterations // inner_iterations`` where those were the parameters
      passed on to the :class:`~ott.experimental.mmsinkhorn.MMSinkhorn` solver.
      Follows the conventions used in
      :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.errors`
    x_s: Tuple of :math:`k` point clouds, ``x_s[i]`` is a matrix of size
      :math:`n_i\times d` where `d` is common to all point clouds.
    a_s: Tuple of :math:`k` probability vectors, each of size :math:`n_i`.
    cost_fns: Instance of :class:`~ott.solvers.geometry.costs.CostFn`, or Tuple
      of :math:`k(k-1)/2` such instances.
    epsilon: entropic regularization used to solve the multimarginal Sinkhorn
      problem.
    ent_reg_cost: the regularized optimal transport cost, the linear
      contribution (dot product between optimal tensor and cost) minus entropy
      times ``epsilon``.
    threshold: convergence threshold used to control the termination of the
      algorithm.
    converged: whether the output corresponds to a solution whose error is
      below the convergence threshold.
    inner_iterations: number of iterations that were run between two
      computations of errors.
  """
  potentials: Tuple[jnp.ndarray, ...] = None
  errors: jnp.ndarray = None
  x_s: jnp.ndarray = None
  a_s: Optional[Tuple[jnp.ndarray, ...]] = None
  cost_fns: Union[costs.CostFn, Tuple[costs.CostFn, ...]] = costs.SqEuclidean
  epsilon: float = None
  ent_reg_cost: jnp.ndarray = None
  threshold: Optional[jnp.ndarray] = None
  converged: Optional[bool] = None
  inner_iterations: Optional[int] = None

  def set(self, **kwargs: Any) -> "MMSinkhornOutput":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  @property
  def n_iters(self) -> int:  # noqa: D102
    """Returns the total number of iterations that were needed to terminate."""
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
  def marginals(self):
    r"""Return a Tuple of :math:`k` marginal probability weight vectors."""
    return tensor_marginals(self.tensor)

  def marginal(self, slice_index: int):
    r"""Return a Tuple of :math:`k` marginal probability weight vectors."""
    return tensor_marginal(self.tensor, slice_index)

  @property
  def transport_mass(self) -> float:
    """Sum of transport tensor."""
    return jnp.sum(self.tensor)


def cost_tensor(
    x_s: Tuple[jnp.ndarray, ...],
    cost_fns: Union[costs.CostFn, Tuple[costs.CostFn, ...]] = costs.SqEuclidean
):
  r"""Creates cost tensor from Tuple of :math:`k` :math:`d`-dim point clouds.

  Args:
    x_s: Tuple of :math:`k` point clouds, each described as a :math:`n_i x d`
      matrix of batched vectors.
    cost_fns: either a single :ott:`ott.geometry.costs.CostFn` object, or a
      tuple of :math:`k (k-1)/2` of them. Current implementation works for
      symmetric and definite cost functions (i.e. such that
  :math:`c(x,y)=c(y,x)` and :math:`c(x,x)=0`).
  """
  k = len(x_s)  #TODO(cuturi) padded version
  ns = [x.shape[0] for x in x_s]

  def c_fn_pair(i, j):
    if isinstance(cost_fns, costs.CostFn):
      return cost_fns
    return cost_fns[i * k - (i * (i + 1)) // 2 + j - i - 1]

  cost_t = jnp.zeros(ns)
  for i in range(k):
    for j in range(i + 1, k):
      cost_m = pointcloud.PointCloud(
          x_s[i], x_s[j], cost_fn=c_fn_pair(i, j)
      ).cost_matrix
      axis = list(range(i)) + list(range(i + 1, j)) + list(range(j + 1, k))
      cost_t += jnp.expand_dims(cost_m, axis=axis)
  return cost_t


def remove_tensor_sum(c: jnp.ndarray, u: Tuple[jnp.ndarray, ...]):
  r"""Removes tensor sum of k vectors to tensor of dimension k.

  Args:
    c: :math:`n_1 \times \\cdots \\ n_k` tensor.
    u: Tuple of :math:`k` vectors, each of size :math:`n_i`.

  Return:
    `c` minus :math:`u[0] \\oplus u[1] \\oplus ... \\oplus u[n]`.
  """
  k = len(u)
  for i in range(k):
    c -= jnp.expand_dims(u[i], axis=list(range(i)) + list(range(i + 1, k)))
  return c


def tensor_marginals(coupling):
  return [tensor_marginal(coupling, l) for l in range(len(coupling.shape))]


def tensor_marginal(coupling, slice_index: int):
  n_s = coupling.shape
  k = len(n_s)
  axis = list(range(slice_index)) + list(range(slice_index + 1, k))
  return coupling.sum(axis=axis)


class MMSinkhorn(sinkhorn.Sinkhorn):
  r"""Multimarginal Sinkhorn solver, aligns :math:`k \,d`-dim point clouds.

  This solver implements the entropic multimarginal solver presented in
  :cite:`benamou:15` and described in :cite:`piran:24`, Algorithm 1. The current
  implementation follows largely the template of the
  :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver, from which it inherits
  the list of parameters. Here only hyperparameters controlling the number of
  iterations and convergence threshold are used, along with the application
  of the :cite:`danskin:67` theorem to instantiate the OT cost.

  To use the solver, one needs to call it on a tuple of :math:`k`
  :math:`d` dimensional point clouds stored in ``x_s``, along with :math:`k`
  probability vectors, stored in ``a_s``, as well as a
  :class:`~ott.geometry.costs.CostFn` instance (or :math:`k(k-1)/2` of them, one
  for each pair of point clouds ``x_s[i]`` and ``x_s[j]``, ``i<j``.)

  The solver uses ``epsilon`` as an input, with the default rule set to one
  twentieth of the mean of the cost tensor resulting from these inputs.
  """

  def __call__(
      self,
      x_s: Tuple[jnp.ndarray, ...],
      a_s: Optional[Tuple[jnp.ndarray, ...]] = None,
      cost_fns: Optional[Union[costs.CostFn, Tuple[costs.CostFn, ...]]] = None,
      epsilon: Optional[float] = None
  ):
    r"""Solves multimarginal OT for :math:`k` :math:`d`-dim point clouds.

    Takes :math:`k` weights :math:`d`-dim point clouds and computes their
    multimarginal optimal transport tensor

    Args:
      x_s: Tuple of :math:`k` point clouds, ``x_s[i]`` is a matrix of size
      :math:`n_i\times d` where :math:`d` is a dimension common to all
      point clouds.
      a_s: Tuple of :math:`k` probability vectors, each of size :math:`n_i`.
      cost_fns: Instance of :class:`~ott.solvers.geometry.costs.CostFn`, or
        Tuple of :math:`k(k-1)/2` such instances. Note that the solver currently
        assumes that these cost functions are symmetric. The cost function at
        index :math:`i(k-\tfrac{i+1}{2})+j-i-1` will be used to compare
        point cloud ``x_s[i]`` with point cloud ``x_s[j]``.
      epsilon: entropic regularization used to solve the multimarginal Sinkhorn
        problem.

    Returns:
      a :class:`~ott.experimental.mmsinkhorn.MMSinkhornOutput` object.
    """
    cost_fns = costs.SqEuclidean() if cost_fns is None else cost_fns
    n_s = [x.shape[0] for x in x_s]
    # Default to uniform probability weights for each point cloud.
    if a_s is None:
      a_s = [jnp.ones((n,)) / n for n in n_s]
    else:
      # Case in which user passes ``None`` weights within tuple.
      a_s = [(jnp.ones((n,)) / n if a is None else a) for a, n in zip(a_s, n_s)]

    cost_t = cost_tensor(x_s, cost_fns)
    state = self.init_state(n_s)
    epsilon = jnp.mean(cost_t) / 20 if epsilon is None else epsilon
    const = cost_t, a_s, epsilon
    out = run(const, self, state)
    return out.set(x_s=x_s, a_s=a_s, cost_fns=cost_fns, epsilon=epsilon)

  def init_state(self, n_s: Tuple[int]) -> MMSinkhornState:
    """Return the initial state of the loop."""
    errors = -jnp.ones((self.outer_iterations, 1))
    potentials = [jnp.zeros((n,)) for n in n_s]
    return MMSinkhornState(potentials=potentials, errors=errors)

  def tree_flatten(self):  # noqa: D102
    aux = vars(self).copy()
    aux.pop("threshold")
    return [self.threshold], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(**aux_data, threshold=children[0])


def run(const, solver, state):
  cost_t, a_s, epsilon = const

  def cond_fn(iteration: int, const: Any, state: MMSinkhornState) -> bool:
    return solver._continue(state, iteration)

  def body_fn(
      iteration: int, const: Tuple[jnp.ndarray, ...],
      state: Tuple[jnp.ndarray, ...], compute_error: bool
  ) -> Tuple[jnp.ndarray, float]:
    cost_t, a_s, epsilon = const
    k = len(a_s)

    def one_slice(potentials: Tuple[jnp.ndarray, ...], l: int, a: jnp.ndarray):
      k = len(potentials)
      pot = potentials[l]
      axis = list(range(l)) + list(range(l + 1, k))
      pot += epsilon * jnp.log(a) + mu.softmin(
          remove_tensor_sum(cost_t, potentials), epsilon, axis=axis
      )
      return potentials[:l] + [pot] + potentials[l + 1:]

    potentials = state.potentials
    for l in range(k):
      potentials = jax.jit(
          one_slice, static_argnames="l"
      )(potentials, l, a_s[l])

    state = state.set(potentials=potentials)
    err = jax.lax.cond(
        jnp.logical_or(
            iteration == solver.max_iterations - 1,
            jnp.logical_and(compute_error, iteration >= solver.min_iterations)
        ), lambda state, c, a, e: state.solution_error(c, a, e),
        lambda *_: jnp.inf, state, cost_t, a_s, epsilon
    )
    errors = state.errors.at[iteration // solver.inner_iterations, :].set(err)
    return state.set(errors=errors)

  fix_point = fixed_point_loop.fixpoint_iter_backprop
  const = (cost_t, a_s, epsilon)
  state = fix_point(
      cond_fn, body_fn, solver.min_iterations, solver.max_iterations,
      solver.inner_iterations, const, state
  )
  converged = jnp.logical_and(
      jnp.logical_not(jnp.any(jnp.isnan(state.errors))), state.errors[-1]
      < solver.threshold
  )[0]
  # Compute cost
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

  ent_reg_cost = jnp.sum(
      jnp.array([
          jnp.sum(potential * a) for potential, a in zip(potentials, a_s)
      ])
  )
  ent_reg_cost += epsilon * (
      1 - jnp.sum(coupling_tensor(potentials, cost_t, epsilon))
  )
  return out.set(ent_reg_cost=ent_reg_cost)


def coupling_tensor(potentials, cost_t, epsilon):
  return jnp.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)
