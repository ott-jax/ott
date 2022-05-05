# coding=utf-8
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
from typing import Optional, NamedTuple, Tuple, Any

import jax
import jax.numpy as jnp
from ott.core import fixed_point_loop
from ott.core import problems
from ott.core import sinkhorn
from ott.geometry import geometry


class LRSinkhornState(NamedTuple):
  """Holds the state of the Low Rank Sinkhorn algorithm."""
  q: Optional[jnp.ndarray] = None
  r: Optional[jnp.ndarray] = None
  g: Optional[jnp.ndarray] = None
  costs: Optional[jnp.ndarray] = None

  def set(self, **kwargs) -> 'LRSinkhornState':
    """Returns a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def reg_ot_cost(self, ot_prob, use_danskin=False):
    return compute_reg_ot_cost(self.q, self.r, self.g, ot_prob, use_danskin)

  def solution_error(self, ot_prob, norm_error: jnp.ndarray, lse_mode: bool
                     ) -> jnp.ndarray:
    return solution_error(self.q, self.r, ot_prob, norm_error, lse_mode)


def compute_reg_ot_cost(q, r, g, ot_prob, use_danskin=False):
  q = jax.lax.stop_gradient(q) if use_danskin else q
  r = jax.lax.stop_gradient(r) if use_danskin else r
  g = jax.lax.stop_gradient(g) if use_danskin else g
  return jnp.sum(ot_prob.geom.apply_cost(r, axis=1) * q * (1.0 / g)[None, :])


def solution_error(q, r, ot_prob,
                   norm_error: jnp.ndarray, lse_mode: bool) -> jnp.ndarray:

  """Computes solution error.

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
      jnp.abs(jnp.sum(q, axis=1) - ot_prob.a) **
      norm_error[:, jnp.newaxis], axis=1) ** (1.0 / norm_error)
  err += jnp.sum(
      jnp.abs(jnp.sum(r, axis=1) - ot_prob.b) **
      norm_error[:, jnp.newaxis], axis=1) ** (1.0 / norm_error)
  err += jnp.sum(
      jnp.abs(jnp.sum(q, axis=0) - jnp.sum(r, axis=0)) **
      norm_error[:, jnp.newaxis], axis=1) ** (1.0 / norm_error)

  return err


class LRSinkhornOutput(NamedTuple):
  """Implements the problems.Transport interface, for a LR Sinkhorn solution."""
  q: Optional[jnp.ndarray] = None
  r: Optional[jnp.ndarray] = None
  g: Optional[jnp.ndarray] = None
  costs: Optional[jnp.ndarray] = None
  reg_ot_cost: Optional[jnp.ndarray] = None
  ot_prob: Optional[problems.LinearProblem] = None

  def set(self, **kwargs) -> 'LRSinkhornOutput':
    """Returns a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  def set_cost(self, ot_prob, lse_mode, use_danskin) -> 'LRSinkhornOutput':
    del lse_mode
    return self.set(reg_ot_cost=self.compute_reg_ot_cost(
        ot_prob, use_danskin))

  def compute_reg_ot_cost(self, ot_prob, use_danskin):
    return compute_reg_ot_cost(self.q, self.r, self.g,
                               ot_prob, use_danskin)

  @property
  def linear(self):
    return isinstance(self.ot_prob, problems.LinearProblem)

  @property
  def geom(self):
    return self.ot_prob.geom

  @property
  def a(self):
    return self.ot_prob.a

  @property
  def b(self):
    return self.ot_prob.b

  @property
  def linear_output(self):
    return True

  @property
  def converged(self):
    if self.costs is None:
      return False
    return jnp.logical_and(jnp.sum(self.costs == -1) > 0,
                           jnp.sum(jnp.isnan(self.costs)) == 0)

  @property
  def matrix(self) -> jnp.ndarray:
    """Transport matrix if it can be instantiated."""
    return jnp.matmul(self.q * (1/self.g)[None, :], self.r.T)

  def apply(self, inputs: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Applies the transport to a ndarray; axis=1 for its transpose."""
    q, r = (self.q, self.r) if axis == 1 else (self.r, self.q)
    inputs = inputs.reshape(-1, r.shape[0])  # (batch, ...)
    return jnp.dot(q, jnp.dot(inputs, r).T / self.g.reshape(-1, 1)).T.squeeze()

  def marginal(self, axis: int) -> jnp.ndarray:
    length = self.q.shape[0] if axis == 0 else self.r.shape[0]
    return self.apply(jnp.ones(length,), axis=axis)

  def cost_at_geom(self, other_geom: geometry.Geometry):
    """Returns OT cost for matrix, evaluated at other cost matrix."""
    return jnp.sum(
        self.q * other_geom.apply_cost(self.r, axis=1) / self.g[None, :])

  def transport_mass(self) -> float:
    """Sum of transport matrix."""
    return self.marginal(0).sum()


@jax.tree_util.register_pytree_node_class
class LRSinkhorn(sinkhorn.Sinkhorn):
  r"""A Low-Rank Sinkhorn solver for linear reg-OT problems.

  A Low-Rank Sinkhorn solver takes a linear OT problem as an input, to return
  a LRSinkhornOutput object.

  The algorithm is described in:
  Low-Rank Sinkhorn Factorization, Scetbon-Cuturi-Peyre, ICML'21.
  http://proceedings.mlr.press/v139/scetbon21a/scetbon21a.pdf

  and the implementation contained here is adapted from that of:
  https://github.com/meyerscetbon/LOT

  The algorithm minimizes a non-convex problem. It therefore requires special
  care to initialization and convergence. Initialization is random by default,
  and convergence evaluated on successive evaluations of the objective. The
  algorithm is only provided for the balanced case.

  Attributes:
    rank: the rank constraint on the coupling to minimize the linear OT problem
    gamma: the (inverse of) gradient stepsize used by mirror descent.
    epsilon: entropic regularization added on top of low-rank problem.
    lse_mode: whether to run computations in lse or kernel mode. At this moment,
      only ``lse_mode=True`` is implemented.
    threshold: convergence threshold, used to quantify whether two successive
      evaluations of the objective are (relatively) close enough to terminate.
    norm_error: norm used to quantify feasibility (deviation to marginals).
    inner_iterations: number of inner iterations used by the algorithm before
      reevaluating progress.
    min_iterations: min number of iterations before evaluating objective.
    max_iterations: max number of iterations allowed.
    use_danskin: use Danskin theorem to evaluate gradient of objective w.r.t.
      input parameters. Only ``True`` handled at this moment.
    implicit_diff: whether to use implicit differentiation. Not implemented
      at this moment.
    jit: jit by default iterations loop.
    rng_key: seed of random numer generator to initialize the LR factors.
    kwargs_dys : keyword arguments passed onto dysktra_update.
  """

  def __init__(self,
               rank: int = 10,
               gamma: float = 1.0,
               epsilon: float = 1e-4,
               init_type: str = 'random',
               lse_mode: bool = True,
               threshold: float = 1e-3,
               norm_error: int = 1,
               inner_iterations: int = 1,
               min_iterations: int = 0,
               max_iterations: int = 2000,
               use_danskin: bool = True,
               implicit_diff: bool = False,
               jit: bool = True,
               rng_key: int = 0,
               kwargs_dys: Any = None):
    self.rank = rank
    self.gamma = gamma
    self.epsilon = epsilon
    self.init_type = init_type
    self.lse_mode = lse_mode
    assert lse_mode, "Kernel mode not yet implemented for LRSinkhorn."
    self.threshold = threshold
    self.inner_iterations = inner_iterations
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self._norm_error = norm_error
    self.jit = jit
    self.use_danskin = use_danskin
    self.implicit_diff = implicit_diff
    assert not implicit_diff, "Implicit diff. not yet implemented for LRSink."
    self.rng_key = rng_key
    self.kwargs_dys = {} if kwargs_dys is None else kwargs_dys

  def __call__(self,
               ot_prob: problems.LinearProblem,
               init: Optional[Tuple[Optional[jnp.ndarray], ...]] = None
               ) -> LRSinkhornOutput:
    """Main interface to run LR sinkhorn."""
    init_q, init_r, init_g = (init if init is not None else (None, None, None))
    # Random initialization for q, r, g using rng_key
    rng = jax.random.split(jax.random.PRNGKey(self.rng_key), 3)
    a, b = ot_prob.a, ot_prob.b
    if self.init_type == 'random':
      if init_g is None:
        init_g = jnp.abs(jax.random.uniform(rng[0], (self.rank,))) + 1
        init_g = init_g / jnp.sum(init_g)
      if init_q is None:
        init_q = jnp.abs(jax.random.normal(rng[1], (a.shape[0], self.rank)))
        init_q = init_q * (a / jnp.sum(init_q, axis=1))[:, None]
      if init_r is None:
        init_r = jnp.abs(jax.random.normal(rng[2], (b.shape[0], self.rank)))
        init_r = init_r * (b / jnp.sum(init_r, axis=1))[:, None]
    elif self.init_type == 'rank_2':
      if init_g is None:
        init_g = jnp.ones((self.rank,)) / self.rank
        lambda_1 = min(jnp.min(a), jnp.min(init_g), jnp.min(b)) / 2
        a1 = jnp.arange(1, a.shape[0] + 1)
        a1 = a1 / jnp.sum(a1)
        a2 = (a - lambda_1 * a1) / (1 - lambda_1)
        b1 = jnp.arange(1, b.shape[0] + 1)
        b1 = b1 / jnp.sum(b1)
        b2 = (b - lambda_1 * b1) / (1 - lambda_1)
        g1 = jnp.arange(1, self.rank + 1)
        g1 = g1 / jnp.sum(g1)
        g2 = (init_g - lambda_1 * g1) / (1 - lambda_1)
      if init_q is None:
        init_q = lambda_1 * jnp.dot(a1[:, None], g1.reshape(1, -1))
        init_q += (1 - lambda_1) * jnp.dot(a2[:, None], g2.reshape(1, -1))
      if init_r is None:
        init_r = lambda_1 * jnp.dot(b1[:, None], g1.reshape(1, -1))
        init_r += (1 - lambda_1) * jnp.dot(b2[:, None], g2.reshape(1, -1))
    else:
      raise NotImplementedError
    run_fn = run if not self.jit else jax.jit(run)
    return run_fn(ot_prob, self, (init_q, init_r, init_g))

  @property
  def norm_error(self):
    return (self._norm_error,)

  def _converged(self, state, iteration):
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_and(
        i >= 2,
        jnp.isclose(costs[i - 2], costs[i - 1], rtol=tol))

  def _diverged(self, state, iteration):
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_not(jnp.isfinite(costs[i - 1]))

  def _continue(self, state, iteration):
    """ continue while not(converged) and not(diverged)"""
    costs, i, tol = state.costs, iteration, self.threshold
    return jnp.logical_or(
        i <= 2,
        jnp.logical_and(
            jnp.logical_not(self._diverged(state, iteration)),
            jnp.logical_not(self._converged(state, iteration))))

  def lr_costs(self, ot_prob, state, iteration):
    c_q = ot_prob.geom.apply_cost(state.r, axis=1) / state.g[None, :]
    c_q += (self.epsilon - 1 / self.gamma) * jnp.log(state.q)
    c_r = ot_prob.geom.apply_cost(state.q) / state.g[None, :]
    c_r += (self.epsilon - 1 / self.gamma) *  jnp.log(state.r)
    diag_qcr = jnp.sum(state.q * ot_prob.geom.apply_cost(state.r, axis=1),
                       axis=0)
    h = diag_qcr / state.g ** 2 - (
      self.epsilon - 1 / self.gamma) * jnp.log(state.g)
    return c_q, c_r, h

  def dysktra_update(self, c_q, c_r, h, ot_prob, state, iteration,
                     min_entry_value=1e-6, tolerance=1e-4,
                     min_iter=0, inner_iter=10, max_iter=200):

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

    def cond_fn(iteration, constants, state_inner):
      del iteration, constants
      err = state_inner[-1]
      return err > tolerance

    def _softm(f, g, c, axis):
      return jax.scipy.special.logsumexp(
          self.gamma * (f[:, None] + g[None, :] - c), axis=axis)

    def body_fn(iteration, constants, state_inner, compute_error):
      f1, f2, g1_old, g2_old, h_old, w_gi, w_gp, w_q, w_r, err = state_inner
      c_q, c_r, loga, logb = constants

      # First Projection
      f1 = jnp.where(
        jnp.isfinite(loga),
        (loga - _softm(f1, g1_old, c_q, 1)) / self.gamma + f1, loga)
      f2 = jnp.where(
        jnp.isfinite(logb),
        (logb - _softm(f2, g2_old, c_r, 1)) / self.gamma + f2, logb)

      h = h_old + w_gi
      h = jnp.maximum(jnp.log(min_entry_value) / self.gamma, h)
      w_gi += h_old - h
      h_old = h

      # Update couplings
      g_q = _softm(f1, g1_old, c_q, 0)
      g_r = _softm(f2, g2_old, c_r, 0)

      # Second Projection
      h = (1 / 3) * (h_old + w_gp + w_q + w_r)
      h += g_q / (3 * self.gamma)
      h += g_r / (3 * self.gamma)
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
          solution_error(q, r, ot_prob, self.norm_error, self.lse_mode), err)[0]

      return f1, f2, g1_old, g2_old, h_old, w_gi, w_gp, w_q, w_r, err

    state_inner = fixed_point_loop.fixpoint_iter_backprop(
        cond_fn, body_fn, min_iter, max_iter,
        inner_iter, constants, state_inner)

    f1, f2, g1_old, g2_old, h_old, _, _, _, _, _ = state_inner

    q, r, g = self.recompute_couplings(f1, g1_old, c_q, f2, g2_old, c_r, h_old)
    return q, r, g

  def recompute_couplings(self, f1, g1, c_q, f2, g2, c_r, h):
    q = jnp.exp(self.gamma * (f1[:, None] + g1[None, :] - c_q))
    r = jnp.exp(self.gamma * (f2[:, None] + g2[None, :] - c_r))
    g = jnp.exp(self.gamma * h)
    return q, r, g

  def lse_step(self, ot_prob, state, iteration) -> LRSinkhornState:
    """LR Sinkhorn LSE update."""
    c_q, c_r, h = self.lr_costs(ot_prob, state, iteration)
    q, r, g = self.dysktra_update(
        c_q, c_r, h, ot_prob, state, iteration, **self.kwargs_dys)
    return state.set(q=q, g=g, r=r)

  def kernel_step(self, ot_prob, state, iteration) -> LRSinkhornState:
    """LR Sinkhorn multiplicative update."""
    # TODO(cuturi): kernel step not implemented.
    return state

  def one_iteration(self, ot_prob, state, iteration, compute_error):
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
    if self.lse_mode:  # In lse_mode, run additive updates.
      state = self.lse_step(ot_prob, state, iteration)
    else:
      state = self.kernel_step(ot_prob, state, iteration)

    # re-computes error if compute_error is True, else set it to inf.
    cost = jnp.where(
        jnp.logical_and(compute_error, iteration >= self.min_iterations),
        state.reg_ot_cost(ot_prob), jnp.inf)
    costs = state.costs.at[iteration // self.inner_iterations].set(cost)
    return state.set(costs=costs)

  def init_state(self, ot_prob, init):
    """Returns the initial state of the loop."""
    q, r, g = init
    costs = -jnp.ones(self.outer_iterations)
    return LRSinkhornState(q=q, r=r, g=g, costs=costs)

  def output_from_state(self, ot_prob, state):
    """Creates an output from a loop state.

    Args:
      ot_prob: the transport problem.
      state: a LRSinkhornState.

    Returns:
      A LRSinkhornOutput.
    """
    return LRSinkhornOutput(
        q=state.q, r=state.r, g=state.g, ot_prob=ot_prob, costs=state.costs)


def run(ot_prob, solver, init) -> LRSinkhornOutput:
  """Run loop of the solver, outputting a state upgraded to an output."""
  out = sinkhorn.iterations(ot_prob, solver, init)
  out = out.set_cost(ot_prob, solver.lse_mode, solver.use_danskin)
  return out.set(ot_prob=ot_prob)


def make(
    rank: int = 10,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
    init_type: str = 'random',
    lse_mode: bool = True,
    threshold: float = 1e-3,
    norm_error: int = 1,
    inner_iterations: int = 1,
    min_iterations: int = 0,
    max_iterations: int = 2000,
    use_danskin: bool = True,
    implicit_diff: bool = False,
    jit: bool = True,
    rng_key: int = 0,
    kwargs_dys: Any = None) -> LRSinkhorn:

  return LRSinkhorn(
      rank=rank,
      gamma=gamma,
      epsilon=epsilon,
      init_type=init_type,
      lse_mode=lse_mode,
      threshold=threshold,
      norm_error=norm_error,
      inner_iterations=inner_iterations,
      min_iterations=min_iterations,
      max_iterations=max_iterations,
      use_danskin=use_danskin,
      implicit_diff=implicit_diff,
      jit=jit,
      rng_key=rng_key,
      kwargs_dys=kwargs_dys)
