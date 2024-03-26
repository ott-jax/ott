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
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ott.geometry import geometry
from ott.initializers.linear import initializers_lr
from ott.math import fixed_point_loop
from ott.math import utils as mu
from ott.problems.linear import linear_problem
from ott.solvers.linear import lr_utils, sinkhorn

__all__ = ["LRSinkhorn", "LRSinkhornOutput"]

ProgressCallbackFn_t = Callable[
    [Tuple[np.ndarray, np.ndarray, np.ndarray, "LRSinkhornState"]], None]


class LRSinkhornState(NamedTuple):
  """State of the Low Rank Sinkhorn algorithm."""
  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  gamma: float
  costs: jnp.ndarray
  errors: jnp.ndarray
  crossed_threshold: bool

  def compute_error(  # noqa: D102
      self, previous_state: "LRSinkhornState"
  ) -> float:
    err_q = mu.gen_js(self.q, previous_state.q, c=1.0)
    err_r = mu.gen_js(self.r, previous_state.r, c=1.0)
    err_g = mu.gen_js(self.g, previous_state.g, c=1.0)

    return ((1.0 / self.gamma) ** 2) * (err_q + err_r + err_g)

  def reg_ot_cost(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      *,
      epsilon: float,
      use_danskin: bool = False
  ) -> float:
    """For LR Sinkhorn, this defaults to the primal cost of LR solution."""
    return compute_reg_ot_cost(
        self.q,
        self.r,
        self.g,
        ot_prob,
        epsilon=epsilon,
        use_danskin=use_danskin
    )

  def solution_error(  # noqa: D102
      self, ot_prob: linear_problem.LinearProblem, norm_error: Tuple[int, ...]
  ) -> jnp.ndarray:
    return solution_error(self.q, self.r, ot_prob, norm_error)

  def set(self, **kwargs: Any) -> "LRSinkhornState":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)


def compute_reg_ot_cost(
    q: jnp.ndarray,
    r: jnp.ndarray,
    g: jnp.ndarray,
    ot_prob: linear_problem.LinearProblem,
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

  def ent(x: jnp.ndarray) -> float:
    # generalized entropy
    return jnp.sum(jsp.special.entr(x) + x)

  tau_a, tau_b = ot_prob.tau_a, ot_prob.tau_b

  q = jax.lax.stop_gradient(q) if use_danskin else q
  r = jax.lax.stop_gradient(r) if use_danskin else r
  g = jax.lax.stop_gradient(g) if use_danskin else g

  cost = jnp.sum(ot_prob.geom.apply_cost(r, axis=1) * q * (1.0 / g)[None, :])
  cost -= epsilon * (ent(q) + ent(r) + ent(g))
  if tau_a != 1.0:
    cost += tau_a / (1.0 - tau_a) * mu.gen_kl(jnp.sum(q, axis=1), ot_prob.a)
  if tau_b != 1.0:
    cost += tau_b / (1.0 - tau_b) * mu.gen_kl(jnp.sum(r, axis=1), ot_prob.b)

  return cost


def solution_error(
    q: jnp.ndarray, r: jnp.ndarray, ot_prob: linear_problem.LinearProblem,
    norm_error: Tuple[int, ...]
) -> jnp.ndarray:
  """Compute solution error.

  Since only balanced case is available for LR, this is marginal deviation.

  Args:
    q: first factor of solution.
    r: second factor of solution.
    ot_prob: linear problem.
    norm_error: int, p-norm used to compute error.

  Returns:
    one or possibly many numbers quantifying deviation to true marginals.
  """
  norm_error = jnp.array(norm_error)
  # Update the error
  err = jnp.sum(
      jnp.abs(jnp.sum(q, axis=1) - ot_prob.a) ** norm_error[:, None], axis=1
  ) ** (1.0 / norm_error)
  err += jnp.sum(
      jnp.abs(jnp.sum(r, axis=1) - ot_prob.b) ** norm_error[:, None], axis=1
  ) ** (1.0 / norm_error)
  err += jnp.sum(
      jnp.abs(jnp.sum(q, axis=0) - jnp.sum(r, axis=0)) ** norm_error[:, None],
      axis=1
  ) ** (1.0 / norm_error)

  return err


class LRSinkhornOutput(NamedTuple):
  """Transport interface for a low-rank Sinkhorn solution."""

  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  costs: jnp.ndarray
  # TODO(michalk8): must be called `errors`, because of `store_inner_errors`
  # in future, enforce via class hierarchy
  errors: jnp.ndarray
  ot_prob: linear_problem.LinearProblem
  epsilon: float
  inner_iterations: int
  # TODO(michalk8): Optional is an artifact of the current impl., refactor
  reg_ot_cost: Optional[float] = None

  def set(self, **kwargs: Any) -> "LRSinkhornOutput":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def set_cost(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      use_danskin: bool = False
  ) -> "LRSinkhornOutput":
    del lse_mode
    return self.set(reg_ot_cost=self.compute_reg_ot_cost(ot_prob, use_danskin))

  def compute_reg_ot_cost(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      use_danskin: bool = False,
  ) -> float:
    return compute_reg_ot_cost(
        self.q,
        self.r,
        self.g,
        ot_prob,
        epsilon=self.epsilon,
        use_danskin=use_danskin
    )

  @property
  def geom(self) -> geometry.Geometry:  # noqa: D102
    return self.ot_prob.geom

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
  def converged(self) -> bool:  # noqa: D102
    return jnp.logical_and(
        jnp.any(self.costs == -1), jnp.all(jnp.isfinite(self.costs))
    )

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix if it can be instantiated."""
    return (self.q * self._inv_g) @ self.r.T

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply the transport to a array; axis=1 for its transpose."""
    q, r = (self.q, self.r) if axis == 1 else (self.r, self.q)
    # for `axis=0`: (batch, m), (m, r), (r,), (r, n)
    return ((inputs @ r) * self._inv_g) @ q.T

  def marginal(self, axis: int) -> jnp.ndarray:  # noqa: D102
    length = self.q.shape[0] if axis == 0 else self.r.shape[0]
    return self.apply(jnp.ones(length,), axis=axis)

  def cost_at_geom(self, other_geom: geometry.Geometry) -> float:
    """Return OT cost for current solution, evaluated at any cost matrix."""
    return jnp.sum(self.q * other_geom.apply_cost(self.r, axis=1) * self._inv_g)

  def transport_cost_at_geom(self, other_geom: geometry.Geometry) -> float:
    """Return (by recomputing it) bare transport cost of current solution."""
    return self.cost_at_geom(other_geom)

  @property
  def primal_cost(self) -> float:
    """Return (by recomputing it) transport cost of current solution."""
    return self.transport_cost_at_geom(other_geom=self.geom)

  @property
  def transport_mass(self) -> float:
    """Sum of transport matrix."""
    return self.marginal(0).sum()

  @property
  def _inv_g(self) -> jnp.ndarray:
    return 1.0 / self.g


@jax.tree_util.register_pytree_node_class
class LRSinkhorn(sinkhorn.Sinkhorn):
  r"""Low-Rank Sinkhorn solver for linear reg-OT problems.

  The algorithm is described in :cite:`scetbon:21` and the implementation
  contained here is adapted from `LOT <https://github.com/meyerscetbon/LOT>`_.

  The algorithm minimizes a non-convex problem. It therefore requires special
  care to initialization and convergence. Convergence is evaluated on successive
  evaluations of the objective.

  Args:
    rank: Rank constraint on the coupling to minimize the linear OT problem
    gamma: The (inverse of) gradient step size used by mirror descent.
    gamma_rescale: Whether to rescale :math:`\gamma` every iteration as
      described in :cite:`scetbon:22b`.
    epsilon: Entropic regularization added on top of low-rank problem.
    initializer: How to initialize the :math:`Q`, :math:`R` and :math:`g`
      factors.
    lse_mode: Whether to run computations in LSE or kernel mode.
    inner_iterations: Number of inner iterations used by the algorithm before
      re-evaluating progress.
    use_danskin: Use Danskin theorem to evaluate gradient of objective w.r.t.
      input parameters. Only `True` handled at this moment.
    progress_fn: callback function which gets called during the Sinkhorn
      iterations, so the user can display the error at each iteration,
      e.g., using a progress bar. See :func:`~ott.utils.default_progress_fn`
      for a basic implementation.
    kwargs_dys: Keyword arguments passed to :meth:`dykstra_update_lse`,
      :meth:`dykstra_update_kernel` or one of the functions defined in
      :mod:`ott.solvers.linear`, depending on whether the problem
      is balanced and on the ``lse_mode``.
    kwargs_init: Keyword arguments for
      :class:`~ott.initializers.linear.initializers_lr.LRInitializer`.
    kwargs: Keyword arguments for
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self,
      rank: int,
      gamma: float = 10.0,
      gamma_rescale: bool = True,
      epsilon: float = 0.0,
      initializer: Union[Literal["random", "rank2", "k-means",
                                 "generalized-k-means"],
                         initializers_lr.LRInitializer] = "random",
      lse_mode: bool = True,
      inner_iterations: int = 10,
      use_danskin: bool = True,
      kwargs_dys: Optional[Mapping[str, Any]] = None,
      kwargs_init: Optional[Mapping[str, Any]] = None,
      progress_fn: Optional[ProgressCallbackFn_t] = None,
      **kwargs: Any,
  ):
    kwargs["implicit_diff"] = None  # not yet implemented
    super().__init__(
        lse_mode=lse_mode,
        inner_iterations=inner_iterations,
        use_danskin=use_danskin,
        **kwargs
    )
    self.rank = rank
    self.gamma = gamma
    self.gamma_rescale = gamma_rescale
    self.epsilon = epsilon
    self.initializer = initializer
    self.progress_fn = progress_fn
    # can be `None`
    self.kwargs_dys = {} if kwargs_dys is None else kwargs_dys
    self.kwargs_init = {} if kwargs_init is None else kwargs_init

  def __call__(
      self,
      ot_prob: linear_problem.LinearProblem,
      init: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray],
                  Optional[jnp.ndarray]] = (None, None, None),
      rng: Optional[jax.Array] = None,
      **kwargs: Any,
  ) -> LRSinkhornOutput:
    """Run low-rank Sinkhorn.

    Args:
      ot_prob: Linear OT problem.
      init: Initial values for the low-rank factors:

        - :attr:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornOutput.q`.
        - :attr:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornOutput.r`.
        - :attr:`~ott.solvers.linear.sinkhorn_lr.LRSinkhornOutput.g`.

        Any `None` values will be initialized using the initializer.
      rng: Random key for seeding.
      kwargs: Additional arguments when calling the initializer.

    Returns:
      The low-rank Sinkhorn output.
    """
    initializer = self.create_initializer(ot_prob)
    init = initializer(ot_prob, *init, rng=rng, **kwargs)
    return run(ot_prob, self, init)

  def _get_costs(
      self,
      ot_prob: linear_problem.LinearProblem,
      state: LRSinkhornState,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    log_q, log_r, log_g = (
        mu.safe_log(state.q), mu.safe_log(state.r), mu.safe_log(state.g)
    )

    inv_g = 1.0 / state.g[None, :]
    tmp = ot_prob.geom.apply_cost(state.r, axis=1)

    grad_q = tmp * inv_g
    grad_r = ot_prob.geom.apply_cost(state.q) * inv_g
    grad_g = -jnp.sum(state.q * tmp, axis=0) / (state.g ** 2)

    grad_q += self.epsilon * log_q
    grad_r += self.epsilon * log_r
    grad_g += self.epsilon * log_g

    if self.gamma_rescale:
      norm_q = jnp.max(jnp.abs(grad_q)) ** 2
      norm_r = jnp.max(jnp.abs(grad_r)) ** 2
      norm_g = jnp.max(jnp.abs(grad_g)) ** 2
      gamma = self.gamma / jnp.max(jnp.array([norm_q, norm_r, norm_g]))
    else:
      gamma = self.gamma

    eps_factor = 1.0 / (self.epsilon * gamma + 1.0)
    gamma *= eps_factor

    c_q = -gamma * grad_q + eps_factor * log_q
    c_r = -gamma * grad_r + eps_factor * log_r
    c_g = -gamma * grad_g + eps_factor * log_g

    return c_q, c_r, c_g, gamma

  # TODO(michalk8): move to `lr_utils` when refactoring this
  def dykstra_update_lse(
      self,
      c_q: jnp.ndarray,
      c_r: jnp.ndarray,
      h: jnp.ndarray,
      gamma: float,
      ot_prob: linear_problem.LinearProblem,
      min_entry_value: float = 1e-6,
      tolerance: float = 1e-3,
      min_iter: int = 0,
      inner_iter: int = 10,
      max_iter: int = 10000
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run Dykstra's algorithm."""
    # shortcuts for problem's definition.
    r = self.rank
    n, m = ot_prob.geom.shape
    loga, logb = jnp.log(ot_prob.a), jnp.log(ot_prob.b)

    h_old = h
    g1_old, g2_old = jnp.zeros(r), jnp.zeros(r)
    f1, f2 = jnp.zeros(n), jnp.zeros(m)

    w_gi, w_gp = jnp.zeros(r), jnp.zeros(r)
    w_q, w_r = jnp.zeros(r), jnp.zeros(r)
    err = jnp.inf
    state_inner = f1, f2, g1_old, g2_old, h_old, w_gi, w_gp, w_q, w_r, err
    constants = c_q, c_r, loga, logb

    def cond_fn(
        iteration: int, constants: Tuple[jnp.ndarray, ...],
        state_inner: Tuple[jnp.ndarray, ...]
    ) -> bool:
      del iteration, constants
      *_, err = state_inner
      return err > tolerance

    def _softm(
        f: jnp.ndarray, g: jnp.ndarray, c: jnp.ndarray, axis: int
    ) -> jnp.ndarray:
      return jsp.special.logsumexp(
          gamma * (f[:, None] + g[None, :] - c), axis=axis
      )

    def body_fn(
        iteration: int, constants: Tuple[jnp.ndarray, ...],
        state_inner: Tuple[jnp.ndarray, ...], compute_error: bool
    ) -> Tuple[jnp.ndarray, ...]:
      # TODO(michalk8): in the future, use `NamedTuple`
      f1, f2, g1_old, g2_old, h_old, w_gi, w_gp, w_q, w_r, err = state_inner
      c_q, c_r, loga, logb = constants

      # First Projection
      f1 = jnp.where(
          jnp.isfinite(loga),
          (loga - _softm(f1, g1_old, c_q, axis=1)) / gamma + f1, loga
      )
      f2 = jnp.where(
          jnp.isfinite(logb),
          (logb - _softm(f2, g2_old, c_r, axis=1)) / gamma + f2, logb
      )

      h = h_old + w_gi
      h = jnp.maximum(jnp.log(min_entry_value) / gamma, h)
      w_gi += h_old - h
      h_old = h

      # Update couplings
      g_q = _softm(f1, g1_old, c_q, axis=0)
      g_r = _softm(f2, g2_old, c_r, axis=0)

      # Second Projection
      h = (1.0 / 3.0) * (h_old + w_gp + w_q + w_r)
      h += g_q / (3.0 * gamma)
      h += g_r / (3.0 * gamma)
      g1 = h + g1_old - g_q / gamma
      g2 = h + g2_old - g_r / gamma

      w_q = w_q + g1_old - g1
      w_r = w_r + g2_old - g2
      w_gp = h_old + w_gp - h

      q, r, _ = recompute_couplings(f1, g1, c_q, f2, g2, c_r, h, gamma)

      g1_old = g1
      g2_old = g2
      h_old = h

      err = jax.lax.cond(
          jnp.logical_and(compute_error, iteration >= min_iter),
          lambda: solution_error(q, r, ot_prob, self.norm_error)[0], lambda: err
      )

      return f1, f2, g1_old, g2_old, h_old, w_gi, w_gp, w_q, w_r, err

    def recompute_couplings(
        f1: jnp.ndarray,
        g1: jnp.ndarray,
        c_q: jnp.ndarray,
        f2: jnp.ndarray,
        g2: jnp.ndarray,
        c_r: jnp.ndarray,
        h: jnp.ndarray,
        gamma: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      q = jnp.exp(gamma * (f1[:, None] + g1[None, :] - c_q))
      r = jnp.exp(gamma * (f2[:, None] + g2[None, :] - c_r))
      g = jnp.exp(gamma * h)
      return q, r, g

    state_inner = fixed_point_loop.fixpoint_iter_backprop(
        cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, state_inner
    )

    f1, f2, g1_old, g2_old, h_old, _, _, _, _, _ = state_inner
    return recompute_couplings(f1, g1_old, c_q, f2, g2_old, c_r, h_old, gamma)

  def dykstra_update_kernel(
      self,
      k_q: jnp.ndarray,
      k_r: jnp.ndarray,
      k_g: jnp.ndarray,
      gamma: float,
      ot_prob: linear_problem.LinearProblem,
      min_entry_value: float = 1e-6,
      tolerance: float = 1e-3,
      min_iter: int = 0,
      inner_iter: int = 10,
      max_iter: int = 10000
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run Dykstra's algorithm."""
    # shortcuts for problem's definition.
    rank = self.rank
    n, m = ot_prob.geom.shape
    a, b = ot_prob.a, ot_prob.b
    supp_a, supp_b = a > 0, b > 0

    g_old = k_g
    v1_old, v2_old = jnp.ones(rank), jnp.ones(rank)
    u1, u2 = jnp.ones(n), jnp.ones(m)

    q_gi, q_gp = jnp.ones(rank), jnp.ones(rank)
    q_q, q_r = jnp.ones(rank), jnp.ones(rank)
    err = jnp.inf
    state_inner = u1, u2, v1_old, v2_old, g_old, q_gi, q_gp, q_q, q_r, err
    constants = k_q, k_r, k_g, a, b

    def cond_fn(
        iteration: int, constants: Tuple[jnp.ndarray, ...],
        state_inner: Tuple[jnp.ndarray, ...]
    ) -> bool:
      del iteration, constants
      *_, err = state_inner
      return err > tolerance

    def body_fn(
        iteration: int, constants: Tuple[jnp.ndarray, ...],
        state_inner: Tuple[jnp.ndarray, ...], compute_error: bool
    ) -> Tuple[jnp.ndarray, ...]:
      # TODO(michalk8): in the future, use `NamedTuple`
      u1, u2, v1_old, v2_old, g_old, q_gi, q_gp, q_q, q_r, err = state_inner
      k_q, k_r, k_g, a, b = constants

      # First Projection
      u1 = jnp.where(supp_a, a / jnp.dot(k_q, v1_old), 0.0)
      u2 = jnp.where(supp_b, b / jnp.dot(k_r, v2_old), 0.0)
      g = jnp.maximum(min_entry_value, g_old * q_gi)
      q_gi = (g_old * q_gi) / g
      g_old = g

      # Second Projection
      v1_trans = jnp.dot(k_q.T, u1)
      v2_trans = jnp.dot(k_r.T, u2)
      g = (g_old * q_gp * v1_old * q_q * v1_trans * v2_old * q_r *
           v2_trans) ** (1 / 3)
      v1 = g / v1_trans
      v2 = g / v2_trans
      q_gp = (g_old * q_gp) / g
      q_q = (q_q * v1_old) / v1
      q_r = (q_r * v2_old) / v2
      v1_old = v1
      v2_old = v2
      g_old = g

      # Compute Couplings
      q, r, _ = recompute_couplings(u1, v1, k_q, u2, v2, k_r, g)

      err = jax.lax.cond(
          jnp.logical_and(compute_error, iteration >= min_iter),
          lambda: solution_error(q, r, ot_prob, self.norm_error)[0], lambda: err
      )

      return u1, u2, v1_old, v2_old, g_old, q_gi, q_gp, q_q, q_r, err

    def recompute_couplings(
        u1: jnp.ndarray,
        v1: jnp.ndarray,
        k_q: jnp.ndarray,
        u2: jnp.ndarray,
        v2: jnp.ndarray,
        k_r: jnp.ndarray,
        g: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      q = u1.reshape((-1, 1)) * k_q * v1.reshape((1, -1))
      r = u2.reshape((-1, 1)) * k_r * v2.reshape((1, -1))
      return q, r, g

    state_inner = fixed_point_loop.fixpoint_iter_backprop(
        cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, state_inner
    )

    u1, u2, v1_old, v2_old, g_old, _, _, _, _, _ = state_inner
    return recompute_couplings(u1, v1_old, k_q, u2, v2_old, k_r, g_old)

  def lse_step(
      self, ot_prob: linear_problem.LinearProblem, state: LRSinkhornState,
      iteration: int
  ) -> LRSinkhornState:
    """LR Sinkhorn LSE update."""
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
    return state.set(q=q, g=g, r=r, gamma=gamma)

  def kernel_step(
      self, ot_prob: linear_problem.LinearProblem, state: LRSinkhornState,
      iteration: int
  ) -> LRSinkhornState:
    """LR Sinkhorn Kernel update."""
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
      self, ot_prob: linear_problem.LinearProblem, state: LRSinkhornState,
      iteration: int, compute_error: bool
  ) -> LRSinkhornState:
    """Carries out one low-rank Sinkhorn iteration.

    Depending on lse_mode, these iterations can be either in:

      - log-space for numerical stability.
      - scaling space, using standard kernel-vector multiply operations.

    Args:
      ot_prob: the transport problem definition
      state: LRSinkhornState named tuple.
      iteration: the current iteration of the Sinkhorn outer loop.
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
    cost = jax.lax.cond(
        jnp.logical_and(compute_error, iteration >= self.min_iterations),
        lambda: state.reg_ot_cost(ot_prob, epsilon=self.epsilon),
        lambda: jnp.inf
    )
    error = state.compute_error(previous_state)
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

  def create_initializer(
      self, prob: linear_problem.LinearProblem
  ) -> initializers_lr.LRInitializer:
    """Create a low-rank Sinkhorn initializer.

    Args:
      prob: Linear OT problem used to determine the initializer.

    Returns:
      Low-rank initializer.
    """
    if isinstance(self.initializer, initializers_lr.LRInitializer):
      assert self.initializer.rank == self.rank, \
        f"Expected initializer's rank to be `{self.rank}`," \
        f"found `{self.initializer.rank}`."
      return self.initializer
    return initializers_lr.LRInitializer.from_solver(
        self, kind=self.initializer, **self.kwargs_init
    )

  def init_state(
      self, ot_prob: linear_problem.LinearProblem,
      init: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
  ) -> LRSinkhornState:
    """Return the initial state of the loop."""
    q, r, g = init
    return LRSinkhornState(
        q=q,
        r=r,
        g=g,
        gamma=self.gamma,
        costs=-jnp.ones(self.outer_iterations),
        errors=-jnp.ones(self.outer_iterations),
        crossed_threshold=False,
    )

  def output_from_state(
      self, ot_prob: linear_problem.LinearProblem, state: LRSinkhornState
  ) -> LRSinkhornOutput:
    """Create an output from a loop state.

    Args:
      ot_prob: the transport problem.
      state: a LRSinkhornState.

    Returns:
      A LRSinkhornOutput.
    """
    return LRSinkhornOutput(
        q=state.q,
        r=state.r,
        g=state.g,
        ot_prob=ot_prob,
        costs=state.costs,
        errors=state.errors,
        epsilon=self.epsilon,
        inner_iterations=self.inner_iterations,
    )

  def _converged(self, state: LRSinkhornState, iteration: int) -> bool:

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

  def _diverged(self, state: LRSinkhornState, iteration: int) -> bool:
    it = iteration // self.inner_iterations
    return jnp.logical_and(
        jnp.logical_not(jnp.isfinite(state.errors[it - 1])),
        jnp.logical_not(jnp.isfinite(state.costs[it - 1]))
    )


def run(
    ot_prob: linear_problem.LinearProblem,
    solver: LRSinkhorn,
    init: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray],
                Optional[jnp.ndarray]],
) -> LRSinkhornOutput:
  """Run loop of the solver, outputting a state upgraded to an output."""
  out = sinkhorn.iterations(ot_prob, solver, init)
  out = out.set_cost(
      ot_prob, lse_mode=solver.lse_mode, use_danskin=solver.use_danskin
  )
  return out.set(ot_prob=ot_prob)
