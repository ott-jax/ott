# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Jax implementation of the unbalanced low-rank GW algorithm."""
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ott import utils
from ott.geometry import geometry, low_rank
from ott.initializers.linear import initializers_lr
from ott.math import unbalanced_functions as uf
from ott.math import utils as mu
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import lr_utils, sinkhorn, sinkhorn_lr

__all__ = ["LRGromovWasserstein", "LRGWOutput"]

ProgressFunction = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "LRGWState"]], None]


class LRGWState(NamedTuple):
  """State of the low-rank GW algorithm."""
  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  gamma: float
  costs: jnp.ndarray
  errors: jnp.ndarray
  crossed_threshold: bool

  def compute_error(  # noqa: D102
      self, previous_state: "LRGWState"
  ) -> float:
    err_q = mu.gen_js(self.q, previous_state.q, c=1.0)
    err_r = mu.gen_js(self.r, previous_state.r, c=1.0)
    err_g = mu.gen_js(self.g, previous_state.g, c=1.0)

    return ((1.0 / self.gamma) ** 2) * (err_q + err_r + err_g)

  def reg_gw_cost(  # noqa: D102
    self,
    ot_prob: quadratic_problem.QuadraticProblem,
    *,
    epsilon: float,
    use_danskin: bool = False
  ) -> float:
    return compute_reg_gw_cost(
        self.q,
        self.r,
        self.g,
        ot_prob,
        epsilon=epsilon,
        use_danskin=use_danskin
    )

  def set(self, **kwargs: Any) -> "LRGWState":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)


def compute_reg_gw_cost(
    q: jnp.ndarray,
    r: jnp.ndarray,
    g: jnp.ndarray,
    ot_prob: quadratic_problem.QuadraticProblem,
    epsilon: float,
    use_danskin: bool = False
) -> float:
  """Compute the regularized OT cost, here the primal cost of the LR solution.

  Args:
    q: first factor of solution
    r: second factor of solution
    g: weights of solution
    ot_prob: linear problem
    epsilon: Entropic regularization.
    use_danskin: if True, use Danskin's theorem :cite:`danskin:67,bertsekas:71`
      to avoid computing the gradient of the cost function.

  Returns:
    regularized OT cost, the (primal) transport cost of the low-rank solution.
  """
  q = jax.lax.stop_gradient(q) if use_danskin else q
  r = jax.lax.stop_gradient(r) if use_danskin else r
  g = jax.lax.stop_gradient(g) if use_danskin else g

  out = LRGWOutput(
      q=q,
      r=r,
      g=g,
      ot_prob=ot_prob,
      costs=None,
      errors=None,
      epsilon=None,
      inner_iterations=None,
      converged=False,
  )

  cost = out.primal_cost - epsilon * (
      mu.gen_ent(q) + mu.gen_ent(r) + mu.gen_ent(g)
  )
  if ot_prob.tau_a != 1.0:
    rho_a = uf.rho(1.0, ot_prob.tau_a)
    cost += rho_a * mu.gen_kl(jnp.sum(q, axis=1), ot_prob.a)
  if ot_prob.tau_b != 1.0:
    rho_b = uf.rho(1.0, ot_prob.tau_b)
    cost += rho_b * mu.gen_kl(jnp.sum(r, axis=1), ot_prob.b)

  return cost


class LRGWOutput(NamedTuple):
  """Transport interface for a low-rank GW solution."""
  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  costs: jnp.ndarray
  # TODO(michalk8): must be called `errors`, because of `store_inner_errors`
  # in future, enforce via class hierarchy
  errors: jnp.ndarray
  ot_prob: quadratic_problem.QuadraticProblem
  epsilon: float
  inner_iterations: int
  converged: bool
  reg_gw_cost: Optional[float] = None

  def set(self, **kwargs: Any) -> "LRGWOutput":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def set_cost(  # noqa: D102
    self,
    ot_prob: quadratic_problem.QuadraticProblem,
    lse_mode: bool,
    use_danskin: bool = False
  ) -> "LRGWOutput":
    del lse_mode
    return self.set(reg_gw_cost=self.compute_reg_gw_cost(ot_prob, use_danskin))

  def compute_reg_gw_cost(  # noqa: D102
    self,
    ot_prob: quadratic_problem.QuadraticProblem,
    use_danskin: bool = False,
  ) -> float:
    return compute_reg_gw_cost(
        self.q,
        self.r,
        self.g,
        ot_prob,
        epsilon=self.epsilon,
        use_danskin=use_danskin
    )

  @property
  def geom(self) -> geometry.Geometry:  # noqa: D102
    """Linearized geometry."""
    return _linearized_geometry(self.ot_prob, q=self.q, r=self.r, g=self.g)

  @property
  def a(self) -> jnp.ndarray:  # noqa: D102
    return self.ot_prob.a

  @property
  def b(self) -> jnp.ndarray:  # noqa: D102
    return self.ot_prob.b

  @property
  def n_iters(self) -> int:  # noqa: D102
    return jnp.sum(self.errors != -1) * self.inner_iterations

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix if it can be instantiated."""
    return (self.q * self._inv_g) @ self.r.T

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply the transport to a array; axis=1 for its transpose."""
    q, r = (self.q, self.r) if axis == 1 else (self.r, self.q)
    # for `axis=0`: (batch, n), (n, r), (r,), (r, m)
    return ((inputs @ r) * self._inv_g) @ q.T

  def marginal(self, axis: int) -> jnp.ndarray:  # noqa: D102
    length = self.q.shape[0] if axis == 0 else self.r.shape[0]
    return self.apply(jnp.ones(length, dtype=self.q.dtype), axis=axis)

  def cost_at_geom(self, other_geom: geometry.Geometry) -> float:
    """Return OT cost for current solution, evaluated at any cost matrix."""
    return jnp.sum(self.q * other_geom.apply_cost(self.r, axis=1) * self._inv_g)

  def transport_cost_at_geom(self, other_geom: geometry.Geometry) -> float:
    """Return (by recomputing it) bare transport cost of current solution."""
    return self.cost_at_geom(other_geom)

  @property
  def primal_cost(self) -> float:
    """Return (by recomputing it) transport cost of current solution."""
    marginal_a = self.ot_prob.a if self.ot_prob.tau_a == 1.0 else self.q.sum(1)
    marginal_b = self.ot_prob.b if self.ot_prob.tau_b == 1.0 else self.r.sum(1)
    lin_a, lin_b = self.ot_prob._apply_linear_loss(marginal_a, marginal_b)

    quad_cost = 0.5 * self.transport_cost_at_geom(other_geom=self.geom)
    quad_cost += jnp.vdot(lin_a, marginal_a)
    quad_cost += jnp.vdot(lin_b, marginal_b)

    if not self.ot_prob.is_fused:
      return quad_cost

    alpha = self.ot_prob.fused_penalty / (self.ot_prob.fused_penalty + 1.0)
    norm_g = jnp.linalg.norm(self.g, ord=1)

    lin_cost = self.cost_at_geom(self.ot_prob.geom_xy)
    return alpha * norm_g * lin_cost + (1.0 - alpha) * quad_cost

  @property
  def transport_mass(self) -> float:
    """Sum of transport matrix."""
    return self.marginal(0).sum()

  @property
  def _inv_g(self) -> jnp.ndarray:
    return 1.0 / self.g


@jax.tree_util.register_pytree_node_class
class LRGromovWasserstein(sinkhorn.Sinkhorn):
  r"""Low-rank Gromov-Wasserstein solver :cite:`scetbon:23`.

  The algorithm minimizes a non-convex problem. It therefore requires special
  care to initialization and convergence. Convergence is evaluated on the change
  between successive iterates of the factors.

  Args:
    rank: Rank constraint on the coupling to minimize the quadratic OT problem.
    gamma: The (inverse of) gradient step size used by mirror descent.
    gamma_rescale: Whether to rescale :math:`\gamma` every iteration as
      described in :cite:`scetbon:22b`.
    epsilon: Entropic regularization added on top of low-rank problem.
    initializer: How to initialize the :math:`Q`, :math:`R` and :math:`g`
      factors. If :obj:`None`, use
      :class:`~ott.initializers.linear.initializers_lr.RandomInitializer`.
    lse_mode: Whether to run computations in LSE or kernel mode.
    inner_iterations: Number of inner iterations used by the algorithm before
      re-evaluating progress.
    min_iterations: The minimum number of low-rank Sinkhorn iterations carried
      out before the error is computed and monitored.
    max_iterations: The maximum number of low-rank Sinkhorn iterations.
    use_danskin: Use Danskin theorem to evaluate gradient of objective w.r.t.
      input parameters. In the balanced case, the gradient w.r.t. ``a`` and
      ``b`` is then zero; use ``use_danskin = False`` to obtain it.
    implicit_diff: Whether to use implicit differentiation. Currently, only
      ``implicit_diff = False`` is implemented.
    progress_fn: callback function which gets called during the GW
      iterations, so the user can display the error at each iteration,
      e.g., using a progress bar. See :func:`~ott.utils.default_progress_fn`
      for a basic implementation.
    kwargs_dys: Keyword arguments passed to :meth:`dykstra_update_lse`,
      :meth:`dykstra_update_kernel`,
      :func:`~ott.solvers.linear.lr_utils.unbalanced_dykstra_lse` or
      :func:`~ott.solvers.linear.lr_utils.unbalanced_dykstra_kernel`, depending
      on whether the problem is balanced and on the ``lse_mode``.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self,
      rank: int,
      gamma: float = 10.0,
      gamma_rescale: bool = True,
      epsilon: float = 0.0,
      initializer: Optional[initializers_lr.LRInitializer] = None,
      lse_mode: bool = True,
      use_danskin: bool = True,
      implicit_diff: bool = False,
      inner_iterations: int = 2_000,
      min_iterations: int = 10_000,
      max_iterations: int = 100_000,
      kwargs_dys: Optional[Mapping[str, Any]] = None,
      progress_fn: Optional[ProgressFunction] = None,
      **kwargs: Any,
  ):
    assert not implicit_diff, "Implicit diff. not yet implemented."
    super().__init__(
        lse_mode=lse_mode,
        inner_iterations=inner_iterations,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        use_danskin=use_danskin,
        implicit_diff=implicit_diff,
        **kwargs
    )
    self.rank = rank
    self.gamma = gamma
    self.gamma_rescale = gamma_rescale
    self.epsilon = epsilon
    self.initializer = initializers_lr.RandomInitializer(
        rank
    ) if initializer is None else initializer
    self.progress_fn = progress_fn
    # can be `None`
    self.kwargs_dys = {} if kwargs_dys is None else kwargs_dys

  def __call__(
      self,
      ot_prob: quadratic_problem.QuadraticProblem,
      init: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = None,
      rng: Optional[jax.Array] = None,
      **kwargs: Any,
  ) -> LRGWOutput:
    """Run the low-rank Gromov-Wasserstein solver.

    Args:
      ot_prob: Quadratic OT problem.
      init: Initial values of the low-rank factors:

        - :attr:`~ott.solvers.quadratic.gromov_wasserstein_lr.LRGWOutput.q`.
        - :attr:`~ott.solvers.quadratic.gromov_wasserstein_lr.LRGWOutput.r`.
        - :attr:`~ott.solvers.quadratic.gromov_wasserstein_lr.LRGWOutput.g`.

        If :obj:`None`, use the initializer.
      rng: Random key for seeding.
      kwargs: Keyword arguments for the initializer.

    Returns:
      The low-rank GW output.
    """
    assert ot_prob._loss_name == "sqeucl", \
        "Only the squared Euclidean loss is supported."

    rng = utils.default_prng_key(rng)
    rng_lrc, rng_init = jax.random.split(rng)

    if ot_prob._is_low_rank_convertible:
      ot_prob = ot_prob.to_low_rank(rng=rng_lrc)
    if init is None:
      init = self.initializer(ot_prob, rng=rng_init, **kwargs)

    return run(ot_prob, self, init)

  def _get_costs(
      self,
      ot_prob: quadratic_problem.QuadraticProblem,
      state: LRGWState,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    q, r, g = state.q, state.r, state.g
    log_q, log_r, log_g = mu.safe_log(q), mu.safe_log(r), mu.safe_log(g)
    inv_g = 1.0 / g[None, :]
    lin_geom = _linearized_geometry(ot_prob, q=q, r=r, g=g)
    if not ot_prob.is_balanced:
      lin_q, lin_r = ot_prob._apply_linear_loss(q.sum(1), r.sum(1))

    tmp = lin_geom.apply_cost(r, axis=1)
    grad_q = tmp * inv_g
    if ot_prob.tau_a != 1.0:  # unbalanced grad
      grad_q += 2.0 * lin_q[:, None]

    grad_r = lin_geom.apply_cost(q, axis=0) * inv_g
    if ot_prob.tau_b != 1.0:  # unbalanced grad
      grad_r += 2.0 * lin_r[:, None]

    omega_quad = jnp.sum(q * tmp, axis=0)
    grad_g = -omega_quad / (g ** 2)

    if ot_prob.is_fused:
      alpha = ot_prob.fused_penalty / (ot_prob.fused_penalty + 1.0)
      norm_g = jnp.linalg.norm(g, ord=1)

      tmp = ot_prob.geom_xy.apply_cost(r, axis=1)
      lin_grad_q = tmp * inv_g * norm_g
      lin_grad_r = ot_prob.geom_xy.apply_cost(q) * inv_g * norm_g

      omega_lin = jnp.sum(q * tmp, axis=0)
      lin_grad_g = -omega_lin / (g ** 2) * norm_g + jnp.sum(q * tmp * inv_g)

      grad_q = alpha * lin_grad_q + (1.0 - alpha) * grad_q
      grad_r = alpha * lin_grad_r + (1.0 - alpha) * grad_r
      grad_g = alpha * lin_grad_g + (1.0 - alpha) * grad_g

    grad_q += self.epsilon * log_q
    grad_r += self.epsilon * log_r
    grad_g += self.epsilon * log_g

    if self.gamma_rescale:
      supp_a, supp_b = ot_prob.a[:, None] > 0.0, ot_prob.b[:, None] > 0.0
      norm_q = jnp.max(jnp.abs(grad_q), where=supp_a, initial=0.0) ** 2
      norm_r = jnp.max(jnp.abs(grad_r), where=supp_b, initial=0.0) ** 2
      norm_g = jnp.max(jnp.abs(grad_g)) ** 2
      gamma = self.gamma / jnp.max(jnp.array([norm_q, norm_r, norm_g]))
    else:
      gamma = self.gamma

    eps_factor = 1.0 / (self.epsilon * gamma + 1.0)
    gamma *= eps_factor

    c_q = -gamma * grad_q + log_q
    c_r = -gamma * grad_r + log_r
    c_g = -gamma * grad_g + log_g

    return c_q, c_r, c_g, gamma

  dykstra_update_lse = sinkhorn_lr.LRSinkhorn.dykstra_update_lse
  dykstra_update_kernel = sinkhorn_lr.LRSinkhorn.dykstra_update_kernel

  def lse_step(
      self, ot_prob: quadratic_problem.QuadraticProblem, state: LRGWState,
      iteration: int
  ) -> LRGWState:
    """Low-rank GW LSE update."""
    c_q, c_r, c_g, gamma = self._get_costs(ot_prob, state)

    if ot_prob.is_balanced:
      c_q, c_r, h = c_q / -gamma, c_r / -gamma, c_g / gamma
      q, r, g = self.dykstra_update_lse(
          c_q, c_r, h, gamma, ot_prob, **self.kwargs_dys
      )
    else:
      q, r, g = lr_utils.unbalanced_dykstra_lse(
          c_q, c_r, c_g, gamma, ot_prob, **self.kwargs_dys
      )
    return state.set(q=q, g=g, r=r, gamma=gamma)  #, (c_q, c_r, c_g)

  def kernel_step(
      self, ot_prob: quadratic_problem.QuadraticProblem, state: LRGWState,
      iteration: int
  ) -> LRGWState:
    """Low-rank GW kernel update."""
    c_q, c_r, c_g, gamma = self._get_costs(ot_prob, state)
    c_q, c_r, c_g = jnp.exp(c_q), jnp.exp(c_r), jnp.exp(c_g)

    if ot_prob.is_balanced:
      q, r, g = self.dykstra_update_kernel(
          c_q, c_r, c_g, gamma, ot_prob, **self.kwargs_dys
      )
    else:
      q, r, g = lr_utils.unbalanced_dykstra_kernel(
          c_q, c_r, c_g, gamma, ot_prob, **self.kwargs_dys
      )
    return state.set(q=q, g=g, r=r, gamma=gamma)

  def one_iteration(
      self, ot_prob: quadratic_problem.QuadraticProblem, state: LRGWState,
      iteration: int, compute_error: bool
  ) -> LRGWState:
    """Carries out one low-rank GW iteration.

    Depending on lse_mode, these iterations can be either in:

      - log-space for numerical stability.
      - scaling space, using standard kernel-vector multiply operations.

    Args:
      ot_prob: the transport problem definition
      state: the current state.
      iteration: the current iteration of the GW outer loop.
      compute_error: flag to indicate this iteration computes/stores an error

    Returns:
      The updated state.
    """
    previous_state = state

    it = iteration // self.inner_iterations
    if self.lse_mode:  # In lse_mode, run additive updates.
      state = self.lse_step(ot_prob, state, iteration)
    else:
      state = self.kernel_step(ot_prob, state, iteration)

    # re-computes error if compute_error is True, else set it to inf.
    record_error = jnp.logical_and(
        compute_error, iteration + 1 >= self.min_iterations
    )
    cost = jax.lax.cond(
        record_error, lambda: state.reg_gw_cost(ot_prob, epsilon=self.epsilon),
        lambda: jnp.array(jnp.inf, dtype=state.costs.dtype)
    )
    error = jax.lax.cond(
        record_error, lambda: state.compute_error(previous_state),
        lambda: jnp.array(jnp.inf, dtype=state.errors.dtype)
    )
    crossed_threshold = jnp.logical_or(
        state.crossed_threshold,
        jnp.logical_and(
            state.errors[it - 1] >= self.threshold, error < self.threshold
        )
    )

    state = state.set(
        costs=state.costs.at[it].set(cost),
        errors=state.errors.at[it].set(error),
        crossed_threshold=crossed_threshold,
    )

    if self.progress_fn is not None:
      jax.debug.callback(
          self.progress_fn,
          (iteration, self.inner_iterations, self.max_iterations, state)
      )

    return state

  @property
  def norm_error(self) -> Tuple[int]:  # noqa: D102
    return self._norm_error,

  def init_state(
      self, ot_prob: quadratic_problem.QuadraticProblem,
      init: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
  ) -> LRGWState:
    """Return the initial state of the loop."""
    q, r, g = init
    return LRGWState(
        q=q,
        r=r,
        g=g,
        gamma=self.gamma,
        costs=-jnp.ones(self.outer_iterations, dtype=q.dtype),
        errors=-jnp.ones(self.outer_iterations, dtype=q.dtype),
        crossed_threshold=False,
    )

  def output_from_state(
      self, ot_prob: quadratic_problem.QuadraticProblem, state: LRGWState
  ) -> LRGWOutput:
    """Create an output from a loop state.

    Args:
      ot_prob: the transport problem.
      state: GW state.

    Returns:
      A LRGWOutput.
    """
    it = jnp.sum(state.errors != -1.0) * self.inner_iterations
    converged = self._converged(state, it)
    return LRGWOutput(
        q=state.q,
        r=state.r,
        g=state.g,
        ot_prob=ot_prob,
        costs=state.costs,
        errors=state.errors,
        epsilon=self.epsilon,
        inner_iterations=self.inner_iterations,
        converged=converged,
    )

  def _converged(self, state: LRGWState, iteration: int) -> bool:

    def conv_crossed(prev_err: float, curr_err: float) -> bool:
      return jnp.logical_and(
          prev_err < self.threshold, curr_err < self.threshold
      )

    def conv_not_crossed(prev_err: float, curr_err: float) -> bool:
      return jnp.logical_and(curr_err < prev_err, curr_err < self.threshold)

    # for convergence error, we consider 2 possibilities:
    # 1. we either crossed the convergence threshold; in this case we require
    #   that the previous error was also below the threshold
    # 2. we haven't crossed the threshold; in this case, we can be below or
    #   above the threshold:
    #     if we're above, we wait until we reach the convergence threshold and
    #     then, the above condition applies
    #     if we're below and we improved w.r.t. the previous iteration,
    #     we have converged; otherwise we continue, since we may be stuck
    #     in a local minimum (e.g., during the initial iterations)

    it = iteration // self.inner_iterations
    return jax.lax.cond(
        state.crossed_threshold, conv_crossed, conv_not_crossed,
        state.errors[it - 2], state.errors[it - 1]
    )

  def _diverged(self, state: LRGWState, iteration: int) -> bool:
    it = iteration // self.inner_iterations - 1
    is_not_finite = jnp.logical_and(
        jnp.logical_not(jnp.isfinite(state.errors[it])),
        jnp.logical_not(jnp.isfinite(state.costs[it]))
    )
    # `jnp.inf` is used if `iteration < self.min_iterations`
    return jnp.logical_and(iteration >= self.min_iterations, is_not_finite)


def run(
    ot_prob: quadratic_problem.QuadraticProblem,
    solver: LRGromovWasserstein,
    init: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> LRGWOutput:
  """Run loop of the solver, outputting a state upgraded to an output."""
  out = sinkhorn.iterations(ot_prob, solver, init)
  out = out.set_cost(
      ot_prob, lse_mode=solver.lse_mode, use_danskin=solver.use_danskin
  )
  return out.set(ot_prob=ot_prob)


def _linearized_geometry(
    prob: quadratic_problem.QuadraticProblem,
    *,
    q: jnp.ndarray,
    r: jnp.ndarray,
    g: jnp.ndarray,
) -> low_rank.LRCGeometry:
  inv_sqrt_g = 1.0 / jnp.sqrt(g[None, :])
  h1, h2 = prob.quad_loss

  tmp1 = quadratic_problem.apply_cost(prob.geom_xx, q, axis=1, fn=h1)
  tmp2 = quadratic_problem.apply_cost(prob.geom_yy, r, axis=1, fn=h2)
  return low_rank.LRCGeometry(-2.0 * tmp1 * inv_sqrt_g, tmp2 * inv_sqrt_g)
