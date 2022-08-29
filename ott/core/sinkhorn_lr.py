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
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from typing_extensions import Literal

from ott.core import fixed_point_loop
from ott.core import initializers_lr as init_lib
from ott.core import linear_problems, sinkhorn
from ott.geometry import geometry


class LRSinkhornState(NamedTuple):
  """State of the Low Rank Sinkhorn algorithm."""

  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  gamma: float
  costs: jnp.ndarray
  criterions: jnp.ndarray
  count_escape: int

  def set(self, **kwargs: Any) -> 'LRSinkhornState':
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def compute_criterion(self, previous_state: "LRSinkhornState") -> float:
    err_1 = ((1. / self.gamma) ** 2) * (
        kl(self.q, previous_state.q) + kl(previous_state.q, self.q)
    )
    err_2 = ((1. / self.gamma) ** 2) * (
        kl(self.r, previous_state.r) + kl(previous_state.r, self.r)
    )
    err_3 = ((1. / self.gamma) ** 2) * (
        kl(self.g, previous_state.g) + kl(previous_state.g, self.g)
    )
    return err_1 + err_2 + err_3

  def reg_ot_cost(
      self,
      ot_prob: linear_problems.LinearProblem,
      use_danskin: bool = False
  ) -> float:
    return compute_reg_ot_cost(self.q, self.r, self.g, ot_prob, use_danskin)

  def solution_error(
      self, ot_prob: linear_problems.LinearProblem, norm_error: jnp.ndarray,
      lse_mode: bool
  ) -> jnp.ndarray:
    return solution_error(self.q, self.r, ot_prob, norm_error, lse_mode)


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
  return jnp.sum(ot_prob.geom.apply_cost(r, axis=1) * q * (1.0 / g)[None, :])


def solution_error(
    q: jnp.ndarray, r: jnp.ndarray, ot_prob: linear_problems.LinearProblem,
    norm_error: jnp.ndarray, lse_mode: bool
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
      jnp.abs(jnp.sum(q, axis=1) - ot_prob.a) ** norm_error[:, jnp.newaxis],
      axis=1
  ) ** (1.0 / norm_error)
  err += jnp.sum(
      jnp.abs(jnp.sum(r, axis=1) - ot_prob.b) ** norm_error[:, jnp.newaxis],
      axis=1
  ) ** (1.0 / norm_error)
  err += jnp.sum(
      jnp.abs(jnp.sum(q, axis=0) -
              jnp.sum(r, axis=0)) ** norm_error[:, jnp.newaxis],
      axis=1
  ) ** (1.0 / norm_error)

  return err


def kl(q1: jnp.ndarray, q2: jnp.ndarray, clipping_value: float = 1e-8) -> float:
  res_1 = -jax.scipy.special.entr(q1)
  res_2 = q1 * jnp.log(jnp.clip(q2, clipping_value))
  res = res_1 - res_2
  return jnp.sum(res)


class LRSinkhornOutput(NamedTuple):
  """Implement the problems.Transport interface, for a LR Sinkhorn solution."""

  q: jnp.ndarray
  r: jnp.ndarray
  g: jnp.ndarray
  costs: jnp.ndarray
  criterions: jnp.ndarray
  ot_prob: linear_problems.LinearProblem
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
    if self.costs is None:
      return False
    return jnp.logical_and(
        jnp.sum(self.costs == -1) > 0,
        jnp.sum(jnp.isnan(self.costs)) == 0
    )

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix if it can be instantiated."""
    return jnp.matmul(self.q * (1 / self.g)[None, :], self.r.T)

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply the transport to a array; axis=1 for its transpose."""
    q, r = (self.q, self.r) if axis == 1 else (self.r, self.q)
    if inputs.ndim == 1:
      inputs = inputs.reshape((1, -1))
    return jnp.dot(q, jnp.dot(inputs, r).T / self.g.reshape(-1, 1)).T.squeeze()

  def marginal(self, axis: int) -> jnp.ndarray:
    length = self.q.shape[0] if axis == 0 else self.r.shape[0]
    return self.apply(jnp.ones(length,), axis=axis)

  def cost_at_geom(self, other_geom: geometry.Geometry) -> float:
    """Return OT cost for matrix, evaluated at other cost matrix."""
    return jnp.sum(
        self.q * other_geom.apply_cost(self.r, axis=1) / self.g[None, :]
    )

  def transport_mass(self) -> float:
    """Sum of transport matrix."""
    return self.marginal(0).sum()


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

      - `'k-means'` - :class:`~ott.core.initializers_lr.KMeansInitializer`.
      - `'rank_2'` - :class:`~ott.core.initializers_lr.Rank2Initializer`.
      - `'random'` - :class:`~ott.core.initializers_lr.RandomInitializer`.

    lse_mode: whether to run computations in lse or kernel mode. At this moment,
      only ``lse_mode = True`` is implemented.
    inner_iterations: number of inner iterations used by the algorithm before
      re-evaluating progress.
    use_danskin: use Danskin theorem to evaluate gradient of objective w.r.t.
      input parameters. Only `True` handled at this moment.
    kwargs_dys: keyword arguments passed to :meth:`dysktra_update`.
    kwargs_init: keyword arguments for
      :class:`~ott.core.initializers_lr.LRSinkhornInitializer`.
    kwargs: Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn`.
  """

  def __init__(
      self,
      rank: int = 10,
      gamma: float = 10.,
      gamma_rescale: bool = True,
      epsilon: float = 0.,
      initializer: Union[Literal["random", "rank_2", "k-means"],
                         init_lib.LRSinkhornInitializer] = "k-means",
      lse_mode: bool = True,
      inner_iterations: int = 1,
      use_danskin: bool = True,
      implicit_diff: bool = False,
      kwargs_dys: Mapping[str, Any] = MappingProxyType({}),
      kwargs_init: Mapping[str, Any] = MappingProxyType({}),
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
    self._initializer = initializer
    self.kwargs_dys = kwargs_dys
    self.kwargs_init = kwargs_init

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
      init: Initial values for low-rank factors:

        - :attr:`~ott.core.sinkhorn_lr.LRSinkhornOutput.q`.
        - :attr:`~ott.core.sinkhorn_lr.LRSinkhornOutput.r`.
        - :attr:`~ott.core.sinkhorn_lr.LRSinkhornOutput.g`.

        Any `None` values will be initialized using the :attr:`initializer`.
      key: Random key for seeding.
      kwargs: Additional arguments when calling :attr:`initializer`.

    Returns:
      The low-rank Sinkhorn output.
    """
    init = self.initializer(ot_prob, *init, key=key, **kwargs)
    run_fn = jax.jit(run) if self.jit else run
    return run_fn(ot_prob, self, init)

  def lr_costs(
      self,
      ot_prob: linear_problems.LinearProblem,
      state: LRSinkhornState,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    log_q, log_r, log_g = jnp.log(state.q), jnp.log(state.r), jnp.log(state.g)

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
      self.gamma /= jnp.max(jnp.array([norm_q, norm_r, norm_g]))

    c_q = grad_q - (1. / self.gamma) * log_q
    c_r = grad_r - (1. / self.gamma) * log_r
    h = -grad_g + (1. / self.gamma) * log_g
    return c_q, c_r, h

  def dysktra_update(
      self,
      c_q: jnp.ndarray,
      c_r: jnp.ndarray,
      h: jnp.ndarray,
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
        iteration: int, constants: Tuple[jnp.ndarray],
        state_inner: Tuple[jnp.ndarray, ...]
    ) -> bool:
      del iteration, constants
      err = state_inner[-1]
      return err > tolerance

    def _softm(
        f: jnp.ndarray, g: jnp.ndarray, c: jnp.ndarray, axis: int
    ) -> jnp.ndarray:
      return jax.scipy.special.logsumexp(
          self.gamma * (f[:, None] + g[None, :] - c), axis=axis
      )

    def body_fn(
        iteration: int, constants: Tuple[jnp.ndarray],
        state_inner: Tuple[jnp.ndarray], compute_error: bool
    ) -> Tuple[jnp.ndarray, ...]:
      # TODO(michalk8): in the future, use `NamedTuple`
      f1, f2, g1_old, g2_old, h_old, w_gi, w_gp, w_q, w_r, err = state_inner
      c_q, c_r, loga, logb = constants

      # First Projection
      f1 = jnp.where(
          jnp.isfinite(loga),
          (loga - _softm(f1, g1_old, c_q, axis=1)) / self.gamma + f1, loga
      )
      f2 = jnp.where(
          jnp.isfinite(logb),
          (logb - _softm(f2, g2_old, c_r, axis=1)) / self.gamma + f2, logb
      )

      h = h_old + w_gi
      h = jnp.maximum(jnp.log(min_entry_value) / self.gamma, h)
      w_gi += h_old - h
      h_old = h

      # Update couplings
      g_q = _softm(f1, g1_old, c_q, axis=0)
      g_r = _softm(f2, g2_old, c_r, axis=0)

      # Second Projection
      h = (1. / 3) * (h_old + w_gp + w_q + w_r)
      h += g_q / (3. * self.gamma)
      h += g_r / (3. * self.gamma)
      g1 = h + g1_old - g_q / self.gamma
      g2 = h + g2_old - g_r / self.gamma

      w_q = w_q + g1_old - g1
      w_r = w_r + g2_old - g2
      w_gp = h_old + w_gp - h

      q, r, _ = self.recompute_couplings(f1, g1, c_q, f2, g2, c_r, h)

      g1_old = g1
      g2_old = g2
      h_old = h

      err = jnp.where(
          jnp.logical_and(compute_error, iteration >= min_iter),
          solution_error(q, r, ot_prob, self.norm_error, self.lse_mode), err
      )[0]

      return f1, f2, g1_old, g2_old, h_old, w_gi, w_gp, w_q, w_r, err

    state_inner = fixed_point_loop.fixpoint_iter_backprop(
        cond_fn, body_fn, min_iter, max_iter, inner_iter, constants, state_inner
    )

    f1, f2, g1_old, g2_old, h_old, _, _, _, _, _ = state_inner

    q, r, g = self.recompute_couplings(f1, g1_old, c_q, f2, g2_old, c_r, h_old)
    return q, r, g

  def recompute_couplings(
      self, f1: jnp.ndarray, g1: jnp.ndarray, c_q: jnp.ndarray, f2: jnp.ndarray,
      g2: jnp.ndarray, c_r: jnp.ndarray, h: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q = jnp.exp(self.gamma * (f1[:, None] + g1[None, :] - c_q))
    r = jnp.exp(self.gamma * (f2[:, None] + g2[None, :] - c_r))
    g = jnp.exp(self.gamma * h)
    return q, r, g

  def lse_step(
      self, ot_prob: linear_problems.LinearProblem, state: LRSinkhornState,
      iteration: int
  ) -> LRSinkhornState:
    """LR Sinkhorn LSE update."""
    c_q, c_r, h = self.lr_costs(ot_prob, state)
    gamma = self.gamma
    q, r, g = self.dysktra_update(c_q, c_r, h, ot_prob, **self.kwargs_dys)
    return state.set(q=q, g=g, r=r, gamma=gamma)

  def kernel_step(
      self, ot_prob: linear_problems.LinearProblem, state: LRSinkhornState,
      iteration: int
  ) -> LRSinkhornState:
    """LR Sinkhorn multiplicative update."""
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
    outer_it = iteration // self.inner_iterations
    if self.lse_mode:  # In lse_mode, run additive updates.
      state = self.lse_step(ot_prob, state, iteration)
    else:
      state = self.kernel_step(ot_prob, state, iteration)

    # re-computes error if compute_error is True, else set it to inf.
    cost = jnp.where(
        jnp.logical_and(compute_error, iteration >= self.min_iterations),
        state.reg_ot_cost(ot_prob), jnp.inf
    )
    costs = state.costs.at[outer_it].set(cost)
    # compute the criterion
    criterion = state.compute_criterion(previous_state)
    criterions = state.criterions.at[outer_it].set(criterion)

    # compute count_escape
    count_escape = state.count_escape + jnp.logical_and(
        iteration >= 2, criterion <= self.threshold * 10.
    )

    return state.set(
        costs=costs, criterions=criterions, count_escape=count_escape
    )

  @property
  def norm_error(self) -> Tuple[int]:
    return self._norm_error,

  @property
  def is_entropic(self) -> bool:
    return self.epsilon != 0.0

  @property
  def initializer(self) -> init_lib.LRSinkhornInitializer:
    """Low-rank Sinkhorn initializer."""
    if isinstance(self._initializer, init_lib.LRSinkhornInitializer):
      assert self._initializer.rank == self.rank
      return self._initializer
    if self._initializer == "k-means":
      return init_lib.KMeansInitializer(
          self.rank,
          sinkhorn_kwargs={
              "norm_error": self._norm_error,
              "lse_mode": self.lse_mode,
              "jit": self.jit,
              "implicit_diff": self.implicit_diff,
              "use_danskin": self.use_danskin
          },
          **self.kwargs_init,
      )
    if self._initializer == "rank_2":
      return init_lib.Rank2Initializer(self.rank, **self.kwargs_init)
    if self._initializer == "random":
      return init_lib.RandomInitializer(self.rank, **self.kwargs_init)
    raise NotImplementedError(
        f"Initializer `{self._initializer}` is not implemented."
    )

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
        criterions=-jnp.ones(self.outer_iterations),
        count_escape=1,
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
        criterions=state.criterions,
    )

  def _converged(self, state: LRSinkhornState, iteration: int) -> bool:
    criterions, count_escape, i, tol = state.criterions, state.count_escape, iteration, self.threshold
    criterion = criterions[i - 1]
    cond_1 = jnp.logical_and(
        jnp.logical_not(i < 2), jnp.logical_not(criterion <= tol / 1e-1)
    )
    cond_2 = jnp.logical_and(
        jnp.logical_and(
            jnp.logical_not(i < 2), jnp.logical_not(criterion > tol / 1e-1)
        ), jnp.logical_not(count_escape == iteration)
    )
    err = jnp.where(jnp.logical_or(cond_1, cond_2), criterion, jnp.inf)
    return jnp.logical_and(i >= 2, err < tol)

  def _diverged(self, state: LRSinkhornState, iteration: int) -> bool:
    return jnp.logical_or(
        jnp.logical_not(jnp.isfinite(state.criterions[iteration - 1])),
        jnp.logical_not(jnp.isfinite(state.costs[iteration - 1]))
    )

  def _continue(self, state: LRSinkhornState, iteration: int) -> bool:
    """Continue while not(converged) and not(diverged)."""
    return jnp.logical_or(
        iteration <= 2,
        jnp.logical_and(
            jnp.logical_not(self._diverged(state, iteration)),
            jnp.logical_not(self._converged(state, iteration))
        )
    )

  def tree_flatten(self):
    children, aux_data = super().tree_flatten()
    aux_data["initializer"] = aux_data.pop("_initializer")
    return children, aux_data


# TODO(michalk8): refactor
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
    rank: int = 10,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
    initializer: Literal['random', 'rank_2', 'k-means'] = 'k-means',
    lse_mode: bool = True,
    threshold: float = 1e-3,
    norm_error: int = 1,
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
