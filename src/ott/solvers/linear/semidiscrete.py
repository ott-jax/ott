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
import dataclasses
import functools
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu

import optax

from ott.geometry import pointcloud
from ott.math import fixed_point_loop
from ott.math import utils as math_utils
from ott.problems.linear import linear_problem
from ott.problems.linear import semidiscrete_linear_problem as sdlp
from ott.solvers.linear import sinkhorn

__all__ = [
    "SemidiscreteState",
    "HardAssignmentOutput",
    "SemidiscreteOutput",
    "SemidiscreteSolver",
]


@jtu.register_dataclass
@dataclasses.dataclass(frozen=True)
class SemidiscreteState:
  """State of the :class:`SemidiscreteSolver`.

  Args:
    it: Iteration number:
    g: Dual potential.
    g_ema: EMA of the dual potential.
    opt_state: State of the optimizer.
    losses: Dual losses.
    grad_norms: Norms of the gradients.
    errors: Marginal deviation errors.
  """
  it: jax.Array
  g: jax.Array
  g_ema: jax.Array
  opt_state: Any
  losses: jax.Array
  grad_norms: jax.Array
  errors: jax.Array


@jtu.register_dataclass
@dataclasses.dataclass(frozen=True)
class HardAssignmentOutput:
  """Unregularized linear OT solution.

  Args:
    ot_prob: Linear OT problem.
    matrix: Transport matrix.
  """
  ot_prob: linear_problem.LinearProblem
  matrix: jesp.BCOO

  @property
  def primal_cost(self) -> jax.Array:
    """Transport cost of the linear OT solution."""
    geom = self.ot_prob.geom
    assert isinstance(geom, pointcloud.PointCloud), type(geom)
    weights = self.matrix.data  #
    row_ixs = self.matrix.indices[:, 0]
    col_ixs = self.matrix.indices[:, 1]
    x, y = geom.x[row_ixs], geom.y[col_ixs]
    return jnp.sum(weights * jax.vmap(geom.cost_fn, in_axes=[0, 0])(x, y))


@jtu.register_dataclass
@dataclasses.dataclass(frozen=True)
class SemidiscreteOutput:
  """Output of the :class:`SemidiscreteSolver`.

  Args:
    g: Dual potential.
    prob: Semi-discrete OT problem.
    it: Final iteration number.
    losses: Dual losses.
    errors: Marginal deviation errors.
    converged: Whether the solver converged.
  """
  g: jax.Array
  prob: sdlp.SemidiscreteLinearProblem
  it: Optional[int] = None
  losses: Optional[jax.Array] = None
  errors: Optional[jax.Array] = None
  converged: Optional[bool] = None

  def sample(
      self, rng: jax.Array, num_samples: int
  ) -> Union[sinkhorn.SinkhornOutput, HardAssignmentOutput]:
    """Sample a point cloud and compute the OT solution.

    Args:
      rng: Random key used for seeding.
      num_samples: Number of samples.

    Returns:
      The sampled output.
    """
    prob = self.prob.sample(rng, num_samples)
    return self._output_from_problem(prob)

  def marginal_chi2_error(
      self,
      rng: jax.Array,
      *,
      num_iters: int,
      batch_size: int,
  ) -> jax.Array:
    """Compute the marginal chi-squared error.

    Args:
      rng: Random key used for seeding.
      num_iters: Number of iterations used to estimate the error.
      batch_size: Number of points to sample from the source distribution
        at each iteration.

    Returns:
      The marginal chi-squared error.
    """
    return _marginal_chi2_error(
        rng,
        self.g,
        self.prob,
        num_iters=num_iters,
        batch_size=batch_size,
    )

  def _output_from_samples(
      self, x: jax.Array
  ) -> Union[sinkhorn.SinkhornOutput, HardAssignmentOutput]:
    epsilon = self.prob.geom.epsilon
    geom = self.prob.geom._from_samples(x, epsilon)
    prob = linear_problem.LinearProblem(
        geom, a=None, b=self.prob.b, tau_a=1.0, tau_b=self.prob.tau_b
    )
    return self._output_from_problem(prob)

  def _output_from_problem(
      self, prob: linear_problem.LinearProblem
  ) -> Union[sinkhorn.SinkhornOutput, HardAssignmentOutput]:
    num_samples, _ = prob.geom.shape
    if not self.prob.geom.is_entropy_regularized:
      z = self.g[None, :] - prob.geom.cost_matrix
      row_ixs = jnp.arange(num_samples)
      col_ixs = jnp.argmax(z, axis=-1)
      matrix = jesp.BCOO(
          (jnp.ones(num_samples, dtype=z.dtype), jnp.c_[row_ixs, col_ixs]),
          shape=prob.geom.shape,
      )
      return HardAssignmentOutput(prob, matrix)

    epsilon = self.prob.geom.epsilon
    f, _ = _soft_c_transform(self.g, prob)
    # SinkhornOutput's potentials must contain
    # probability weight normalization
    f_tilde = f + epsilon * jnp.log(1.0 / num_samples)
    g_tilde = self.g + epsilon * jnp.log(self.prob.b)

    return sinkhorn.SinkhornOutput(
        potentials=(f_tilde, g_tilde),
        ot_prob=prob,
    )


@jtu.register_static
@dataclasses.dataclass(frozen=True)
class SemidiscreteSolver:
  """Semi-discrete optimal transport solver.

  Args:
    batch_size: Number of points to sample at each iteration.
    min_iterations: Minimum number of iterations.
    max_iterations: Maximum number of iterations.
    optimizer: Optimizer.
    inner_iterations: Number of iterations to run between the error computation.
    error_batch_size: Batch size to use when computing
      the marginal chi-squared error. If :obj:`None`, use ``batch_size``.
    error_iterations: Number of iterations used to estimate
      the marginal chi-squared error.
    threshold: Convergence threshold for the marginal chi-squared error.
    potential_ema: Exponential moving average of the dual potential.
    callback: Callback with a signature ``(state) -> None`` that is called
      at every iteration.
  """
  batch_size: int
  min_iterations: int
  max_iterations: int
  optimizer: optax.GradientTransformation
  inner_iterations: int = 1000
  error_batch_size: Optional[int] = None
  error_iterations: int = 1000
  threshold: float = 1e-3
  potential_ema: float = 0.99
  callback: Optional[Callable[[SemidiscreteState], None]] = None

  def __call__(
      self,
      rng: jax.Array,
      prob: sdlp.SemidiscreteLinearProblem,
      g_init: Optional[jax.Array] = None,
  ) -> SemidiscreteOutput:
    """Run the semi-discrete solver.

    Args:
      rng: Random key used for seeding.
      prob: Semi-discrete problem.
      g_init: Initial potential value of shape ``[m,]``.

    Returns:
      The semi-discrete output.
    """

    def cond_fn(
        it: int,
        prob: sdlp.SemidiscreteLinearProblem,
        state: SemidiscreteState,
    ) -> bool:
      del prob
      loss = state.losses[it - 1]
      err = jnp.abs(state.errors[it // self.inner_iterations - 1])

      not_converged = err > self.threshold
      not_diverged = jnp.isfinite(loss)
      # continue if not converged and not diverged
      return jnp.logical_or(
          it == 0, jnp.logical_and(not_converged, not_diverged)
      )

    def body_fn(
        it: int,
        prob: sdlp.SemidiscreteLinearProblem,
        state: SemidiscreteState,
        compute_error: bool,
    ) -> SemidiscreteState:
      rng_it = jr.fold_in(rng, it)

      lin_prob = prob.sample(rng_it, self.batch_size)
      g_old = state.g

      loss, grads = jax.value_and_grad(_semidiscrete_loss)(
          g_old, lin_prob, prob.geom.is_entropy_regularized
      )
      grad_norm = jnp.linalg.norm(grads)
      losses = state.losses.at[it].set(loss)
      grad_norms = state.grad_norms.at[it].set(grad_norm)

      updates, opt_state = self.optimizer.update(grads, state.opt_state, g_old)
      g_new = optax.apply_updates(g_old, updates)
      g_ema = optax.incremental_update(g_new, state.g_ema, self.potential_ema)

      # fmt: off
      error = jax.lax.cond(
          compute_error,
          lambda: _marginal_chi2_error(
              # use same rng to evaluate the errors
              rng_chi2, g_ema, prob,
              num_iters=self.error_iterations,
              batch_size=self.error_batch_size or self.batch_size,
          ),
          lambda: jnp.array(jnp.inf, dtype=dtype),
      )
      # fmt: on
      errors = state.errors.at[it // self.inner_iterations].set(error)

      state = SemidiscreteState(
          it=it,
          g=g_new,
          g_ema=g_ema,
          opt_state=opt_state,
          losses=losses,
          grad_norms=grad_norms,
          errors=errors,
      )

      if self.callback is not None:
        jax.debug.callback(self.callback, state)

      return state

    _, m = prob.geom.shape
    dtype = prob.geom.dtype

    if g_init is None:
      g_init = jnp.zeros(m, dtype=dtype)
    else:
      assert g_init.shape == (m,), (g_init.shape, (m,))

    state = SemidiscreteState(
        it=jnp.array(0),
        g=g_init,
        g_ema=g_init,
        opt_state=self.optimizer.init(g_init),
        losses=jnp.full((self.max_iterations,), fill_value=jnp.inf,
                        dtype=dtype),
        grad_norms=jnp.full((self.max_iterations,),
                            fill_value=jnp.inf,
                            dtype=dtype),
        errors=jnp.full((self.max_iterations // self.inner_iterations),
                        fill_value=jnp.inf,
                        dtype=dtype),
    )

    rng, rng_chi2 = jr.split(rng, 2)
    state: SemidiscreteState = fixed_point_loop.fixpoint_iter(
        cond_fn,
        body_fn,
        min_iterations=self.min_iterations,
        max_iterations=self.max_iterations,
        inner_iterations=self.inner_iterations,
        constants=prob,
        state=state,
    )

    return self._to_output(state, prob)

  def _to_output(
      self, state: SemidiscreteState, prob: sdlp.SemidiscreteLinearProblem
  ) -> SemidiscreteOutput:
    it = state.it
    below_thr = state.errors[it // self.inner_iterations] <= self.threshold
    finite_loss = jnp.isfinite(state.losses[it])
    return SemidiscreteOutput(
        g=state.g_ema,
        prob=prob,
        it=it,
        losses=state.losses,
        errors=state.errors,
        converged=jnp.logical_and(below_thr, finite_loss),
    )


def _soft_c_transform(
    g: jax.Array, prob: linear_problem.LinearProblem
) -> Tuple[jax.Array, jax.Array]:
  cost = prob.geom.cost_matrix
  epsilon = prob.geom.epsilon
  z = (g[None, :] - cost) / epsilon
  return -epsilon * math_utils.logsumexp(z, b=prob.b, axis=-1), z


def _hard_c_transform(
    g: jax.Array, prob: linear_problem.LinearProblem
) -> Tuple[jax.Array, jax.Array]:
  cost = prob.geom.cost_matrix
  z = g[None, :] - cost
  return -jnp.max(z, axis=-1), z


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _semidiscrete_loss(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
    is_soft: bool,
) -> jax.Array:
  f, _ = _soft_c_transform(g, prob) if is_soft else _hard_c_transform(g, prob)
  return -jnp.mean(f) - jnp.dot(g, prob.b)


def _semidiscrete_loss_fwd(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
    is_soft: bool,
) -> Tuple[jax.Array, Tuple[jax.Array, linear_problem.LinearProblem]]:
  f, z = _soft_c_transform(g, prob) if is_soft else _hard_c_transform(g, prob)
  return -jnp.mean(f) - jnp.dot(g, prob.b), (z, prob)


def _semidiscrete_loss_bwd(
    is_soft: bool,
    res: jax.Array,
    g: jax.Array,
) -> Tuple[jax.Array, None]:
  z, prob = res
  n, m = prob.geom.shape
  if is_soft:
    grad = jsp.special.softmax(z, axis=-1).sum(0)
  else:
    ixs = jnp.argmax(z, axis=-1)
    grad = jax.ops.segment_sum(
        jnp.ones(n, dtype=z.dtype), segment_ids=ixs, num_segments=m
    )
  assert grad.shape == (m,), (grad.shape, (m,))
  # TODO(michalk8): double-check
  grad = grad * (1.0 / n) - prob.b
  grad = jnp.where(prob.b > 0.0, grad, 0.0)
  return g * grad, None


_semidiscrete_loss.defvjp(_semidiscrete_loss_fwd, _semidiscrete_loss_bwd)


def _marginal_chi2_error(
    rng: jax.Array,
    g: jax.Array,
    prob: sdlp.SemidiscreteLinearProblem,
    *,
    num_iters: int,
    batch_size: int,
) -> jax.Array:

  def compute_chi2(matrix: Union[jax.Array, jesp.BCOO]) -> jax.Array:
    # we assume each rows sums to one
    # also convert to flow to avoid integer overflows
    matrix = batch_size * matrix
    p2 = m * (matrix @ matrix.T)
    if isinstance(p2, jesp.BCOO):
      # no trace impl. for BCOO, densify
      p2 = p2.todense()
    return (p2.sum() - p2.trace()) / (batch_size * (batch_size - 1.0)) - 1.0

  def body(chi2_err_avg: jax.Array, it: jax.Array) -> Tuple[jax.Array, None]:
    rng_it = jr.fold_in(rng, it)
    matrix = out.sample(rng_it, batch_size).matrix
    chi2 = compute_chi2(matrix)
    chi2_err_avg = chi2_err_avg + chi2 / num_iters
    return chi2_err_avg, None

  out = SemidiscreteOutput(g=g, prob=prob)
  _, m = prob.geom.shape

  chi2_err = jnp.zeros((), dtype=g.dtype)
  chi2_err, _ = jax.lax.scan(body, init=chi2_err, xs=jnp.arange(num_iters))
  return chi2_err
