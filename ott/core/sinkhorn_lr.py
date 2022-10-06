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
"""A Jax implementation of the Low-Rank Sinkhorn algorithm."""
from typing import Any, Mapping, NamedTuple, NoReturn, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing_extensions import Literal

from ott.core import _math_utils as mu
from ott.core import fixed_point_loop
from ott.core import initializers_lr as init_lib
from ott.core import linear_problems, sinkhorn
from ott.geometry import geometry, low_rank, pointcloud


class LRSinkhornState(NamedTuple):
  """State of the Low Rank Sinkhorn algorithm."""

  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  gamma: float
  costs: jnp.ndarray
  errors: jnp.ndarray
  crossed_threshold: bool

  def compute_error(self, previous_state: "LRSinkhornState") -> float:
    err_1 = mu.js(self.q, previous_state.q, c=1.)
    err_2 = mu.js(self.r, previous_state.r, c=1.)
    err_3 = mu.js(self.g, previous_state.g, c=1.)

    return ((1. / self.gamma) ** 2) * (err_1 + err_2 + err_3)

  def reg_ot_cost(
      self,
      ot_prob: linear_problems.LinearProblem,
      use_danskin: bool = False
  ) -> float:
    return compute_reg_ot_cost(self.q, self.r, self.g, ot_prob, use_danskin)

  def solution_error(
      self, ot_prob: linear_problems.LinearProblem, norm_error: Tuple[int, ...],
      lse_mode: bool
  ) -> jnp.ndarray:
    return solution_error(self.q, self.r, ot_prob, norm_error, lse_mode)

  def set(self, **kwargs: Any) -> 'LRSinkhornState':
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)


def compute_reg_ot_cost(
    q: jnp.ndarray,
    r: jnp.ndarray,
    g: jnp.ndarray,
    ot_prob: linear_problems.LinearProblem,
    use_danskin: bool = False
) -> float:
  q = jax.lax.stop_gradient(q) if use_danskin else q
  r = jax.lax.stop_gradient(r) if use_danskin else r
  g = jax.lax.stop_gradient(g) if use_danskin else g
  return jnp.sum(ot_prob.geom.apply_cost(r, axis=1) * q * (1. / g)[None, :])


def solution_error(
    q: jnp.ndarray, r: jnp.ndarray, ot_prob: linear_problems.LinearProblem,
    norm_error: Tuple[int, ...], lse_mode: bool
) -> jnp.ndarray:
  """Compute solution error.

  Since only balanced case is available for LR, this is marginal deviation.

  Args:
    q: first factor of solution
    r: second factor of solution
    ot_prob: linear problem
    norm_error: int, p-norm used to compute error.
    lse_mode: True if log-sum-exp operations, False if kernel vector products.

  Returns:
    one or possibly many numbers quantifying deviation to true marginals.
  """
  del lse_mode
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
  """Implement the problems.Transport interface, for a LR Sinkhorn solution."""

  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  costs: jnp.ndarray
  # TODO(michalk8): must be called `errors`, because of `store_inner_errors`
  # in future, enforce via class hierarchy
  errors: jnp.ndarray
  ot_prob: linear_problems.LinearProblem
  # TODO(michalk8): Optional is an artifact of the current impl., refactor
  reg_ot_cost: Optional[float] = None

  def set(self, **kwargs: Any) -> 'LRSinkhornOutput':
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def set_cost(
      self,
      ot_prob: linear_problems.LinearProblem,
      lse_mode: bool,
      use_danskin: bool = False
  ) -> 'LRSinkhornOutput':
    del lse_mode
    return self.set(reg_ot_cost=self.compute_reg_ot_cost(ot_prob, use_danskin))

  def compute_reg_ot_cost(
      self,
      ot_prob: linear_problems.LinearProblem,
      use_danskin: bool = False,
  ) -> float:
    return compute_reg_ot_cost(self.q, self.r, self.g, ot_prob, use_danskin)

  @property
  def linear(self) -> bool:
    return isinstance(self.ot_prob, linear_problems.LinearProblem)

  @property
  def geom(self) -> geometry.Geometry:
    return self.ot_prob.geom

  @property
  def a(self) -> jnp.ndarray:
    return self.ot_prob.a

  @property
  def b(self) -> jnp.ndarray:
    return self.ot_prob.b

  @property
  def linear_output(self) -> bool:
    return True

  @property
  def converged(self) -> bool:
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

  def marginal(self, axis: int) -> jnp.ndarray:
    length = self.q.shape[0] if axis == 0 else self.r.shape[0]
    return self.apply(jnp.ones(length,), axis=axis)

  def cost_at_geom(self, other_geom: geometry.Geometry) -> float:
    """Return OT cost for matrix, evaluated at other cost matrix."""
    return jnp.sum(self.q * other_geom.apply_cost(self.r, axis=1) * self._inv_g)

  # TODO(michalk8): when refactoring the API, use a property
  def transport_mass(self) -> float:
    """Sum of transport matrix."""
    return self.marginal(0).sum()

  @property
  def _inv_g(self) -> jnp.ndarray:
    return 1. / self.g


@jax.tree_util.register_pytree_node_class
class LRSinkhorn(sinkhorn.Sinkhorn):
  r"""A Low-Rank Sinkhorn solver for linear reg-OT problems.

  The algorithm is described in :cite:`scetbon:21` and the implementation
  contained here is adapted from `LOT <https://github.com/meyerscetbon/LOT>`_.

  The algorithm minimizes a non-convex problem. It therefore requires special
  care to initialization and convergence. Convergence is evaluated on successive
  evaluations of the objective. The algorithm is only provided for the balanced
  case.

  Args:
    rank: the rank constraint on the coupling to minimize the linear OT problem
    gamma: the (inverse of) gradient step size used by mirror descent.
    gamma_rescale: Whether to rescale :math:`\gamma` every iteration as
      described in :cite:`scetbon:22b`.
    epsilon: entropic regularization added on top of low-rank problem.
    initializer: How to initialize the :math:`Q`, :math:`R` and :math:`g`
      factors. Valid options are:

        - `'random'` - :class:`~ott.core.initializers_lr.RandomInitializer`.
        - `'rank2'` - :class:`~ott.core.initializers_lr.Rank2Initializer`.
        - `'k-means'` - :class:`~ott.core.initializers_lr.KMeansInitializer`.
        - `'generalized-k-means'` -
          :class:`~ott.core.initializers_lr.GeneralizedKMeansInitializer`.

      If `None`, :class:`~ott.core.initializers_lr.KMeansInitializer`
      is used when the linear problem's geometry is
      :class:`~ott.geometry.pointcloud.PointCloud` or
      :class:`~ott.geometry.low_rank.LRCGeometry`.
      Otherwise, use :class:`~ott.core.initializers_lr.RandomInitializer`.

    lse_mode: whether to run computations in lse or kernel mode. At the moment,
      only ``lse_mode = True`` is implemented.
    inner_iterations: number of inner iterations used by the algorithm before
      re-evaluating progress.
    use_danskin: use Danskin theorem to evaluate gradient of objective w.r.t.
      input parameters. Only `True` handled at this moment.
    implicit_diff: Whether to use implicit differentiation. Currently, only
      ``implicit_diff = False`` is implemented.
    kwargs_dys: keyword arguments passed to :meth:`dykstra_update`.
    kwargs_init: keyword arguments for
      :class:`~ott.core.initializers_lr.LRInitializer`.
    kwargs: Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self,
      rank: int,
      gamma: float = 10.,
      gamma_rescale: bool = True,
      epsilon: float = 0.,
      initializer: Optional[Union[Literal["random", "rank2", "k-means",
                                          "generalized-k-means"],
                                  init_lib.LRInitializer]] = "random",
      lse_mode: bool = True,
      inner_iterations: int = 10,
      use_danskin: bool = True,
      implicit_diff: bool = False,
      kwargs_dys: Optional[Mapping[str, Any]] = None,
      kwargs_init: Optional[Mapping[str, Any]] = None,
      **kwargs: Any,
  ):
    assert lse_mode, "Kernel mode not yet implemented for LRSinkhorn."
    assert not implicit_diff, "Implicit diff. not yet implemented for LRSink."
    super().__init__(
        lse_mode=lse_mode,
        inner_iterations=inner_iterations,
        use_danskin=use_danskin,
        implicit_diff=implicit_diff,
        **kwargs
    )
    self.rank = rank
    self.gamma = gamma
    self.gamma_rescale = gamma_rescale
    self.epsilon = epsilon
    self.initializer = initializer
    # can be `None`
    self.kwargs_dys = {} if kwargs_dys is None else kwargs_dys
    self.kwargs_init = {} if kwargs_init is None else kwargs_init

  def __call__(
      self,
      ot_prob: linear_problems.LinearProblem,
      init: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray],
                  Optional[jnp.ndarray]] = (None, None, None),
      key: Optional[jnp.ndarray] = None,
      **kwargs: Any,
  ) -> LRSinkhornOutput:
    """Run low-rank Sinkhorn.

    Args:
      ot_prob: Linear OT problem.
      init: Initial values for the low-rank factors:

        - :attr:`~ott.core.sinkhorn_lr.LRSinkhornOutput.q`.
        - :attr:`~ott.core.sinkhorn_lr.LRSinkhornOutput.r`.
        - :attr:`~ott.core.sinkhorn_lr.LRSinkhornOutput.g`.

        Any `None` values will be initialized using the initializer.
      key: Random key for seeding.
      kwargs: Additional arguments when calling the initializer.

    Returns:
      The low-rank Sinkhorn output.
    """
    assert ot_prob.is_balanced, "Unbalanced case is not implemented."
    initializer = self.create_initializer(ot_prob)
    init = initializer(ot_prob, *init, key=key, **kwargs)
    run_fn = jax.jit(run) if self.jit else run
    return run_fn(ot_prob, self, init)

  def _lr_costs(
      self,
      ot_prob: linear_problems.LinearProblem,
      state: LRSinkhornState,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    log_q, log_r, log_g = (
        mu.safe_log(state.q), mu.safe_log(state.r), mu.safe_log(state.g)
    )

    grad_q = ot_prob.geom.apply_cost(state.r, axis=1) / state.g[None, :]
    grad_r = ot_prob.geom.apply_cost(state.q) / state.g[None, :]
    diag_qcr = jnp.sum(
        state.q * ot_prob.geom.apply_cost(state.r, axis=1), axis=0
    )
    grad_g = -diag_qcr / (state.g ** 2)
    if self.is_entropic:
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

    c_q = grad_q - (1. / gamma) * log_q
    c_r = grad_r - (1. / gamma) * log_r
    h = -grad_g + (1. / gamma) * log_g
    return c_q, c_r, h, gamma

  def dykstra_update(
      self,
      c_q: jnp.ndarray,
      c_r: jnp.ndarray,
      h: jnp.ndarray,
      gamma: float,
      ot_prob: linear_problems.LinearProblem,
      min_entry_value: float = 1e-6,
      tolerance: float = 1e-3,
      min_iter: int = 0,
      inner_iter: int = 10,
      max_iter: int = 10000
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
      h = (1. / 3.) * (h_old + w_gp + w_q + w_r)
      h += g_q / (3. * gamma)
      h += g_r / (3. * gamma)
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
          lambda: solution_error(q, r, ot_prob, self.norm_error, self.lse_mode)[
              0], lambda: err
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

  def lse_step(
      self, ot_prob: linear_problems.LinearProblem, state: LRSinkhornState,
      iteration: int
  ) -> LRSinkhornState:
    """LR Sinkhorn LSE update."""
    c_q, c_r, h, gamma = self._lr_costs(ot_prob, state)
    q, r, g = self.dykstra_update(
        c_q, c_r, h, gamma, ot_prob, **self.kwargs_dys
    )
    return state.set(q=q, g=g, r=r, gamma=gamma)

  def kernel_step(
      self, ot_prob: linear_problems.LinearProblem, state: LRSinkhornState,
      iteration: int
  ) -> NoReturn:
    """Not implemented."""
    # TODO(cuturi): kernel step not implemented.
    raise NotImplementedError("Not implemented.")

  def one_iteration(
      self, ot_prob: linear_problems.LinearProblem, state: LRSinkhornState,
      iteration: int, compute_error: bool
  ) -> LRSinkhornState:
    """Carries out one LR sinkhorn iteration.

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
        lambda: state.reg_ot_cost(ot_prob), lambda: jnp.inf
    )
    error = state.compute_error(previous_state)
    crossed_threshold = jnp.logical_or(
        state.crossed_threshold,
        jnp.logical_and(
            state.errors[it - 1] >= self.threshold, error < self.threshold
        )
    )

    return state.set(
        costs=state.costs.at[it].set(cost),
        errors=state.errors.at[it].set(error),
        crossed_threshold=crossed_threshold,
    )

  @property
  def norm_error(self) -> Tuple[int]:
    return self._norm_error,

  @property
  def is_entropic(self) -> bool:
    """Whether entropy regularization is used."""
    return self.epsilon > 0.

  def create_initializer(
      self, prob: linear_problems.LinearProblem
  ) -> init_lib.LRInitializer:
    """Create a low-rank Sinkhorn initializer.

    Args:
      prob: Linear OT problem used to determine the initializer.

    Returns:
      Low-rank initializer.
    """
    if isinstance(self.initializer, init_lib.LRInitializer):
      initializer = self.initializer
    elif self.initializer is None:
      kind = "k-means" if isinstance(
          prob.geom, (pointcloud.PointCloud, low_rank.LRCGeometry)
      ) else "random"
      initializer = init_lib.LRInitializer.from_solver(
          self, kind=kind, **self.kwargs_init
      )
    else:
      initializer = init_lib.LRInitializer.from_solver(
          self, kind=self.initializer, **self.kwargs_init
      )

    assert initializer.rank == self.rank, \
        f"Expected initializer of rank `{self.rank}`, " \
        f"found `{initializer.rank}`."
    return initializer

  def init_state(
      self, ot_prob: linear_problems.LinearProblem,
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
      self, ot_prob: linear_problems.LinearProblem, state: LRSinkhornState
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
    ot_prob: linear_problems.LinearProblem,
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


def make(
    rank: int,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
    initializer: Literal['random', 'rank2', 'k-means'] = 'k-means',
    lse_mode: bool = True,
    threshold: float = 1e-3,
    norm_error: int = 10,
    inner_iterations: int = 1,
    min_iterations: int = 0,
    max_iterations: int = 2000,
    use_danskin: bool = True,
    implicit_diff: bool = False,
    jit: bool = True,
    kwargs_dys: Optional[Mapping[str, Any]] = None
) -> LRSinkhorn:
  return LRSinkhorn(
      rank=rank,
      gamma=gamma,
      epsilon=epsilon,
      initializer=initializer,
      lse_mode=lse_mode,
      threshold=threshold,
      norm_error=norm_error,
      inner_iterations=inner_iterations,
      min_iterations=min_iterations,
      max_iterations=max_iterations,
      use_danskin=use_danskin,
      implicit_diff=implicit_diff,
      jit=jit,
      kwargs_dys=kwargs_dys
  )
