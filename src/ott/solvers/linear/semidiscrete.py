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
import math
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu

import optax

from ott import utils
from ott.geometry import pointcloud
from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem, potentials
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
    it: Iteration number.
    g: Dual potential.
    g_ema: Exponential moving average of the dual potential.
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
    f: The first dual potential.
    g: The second dual potential.
    ot_prob: Linear OT problem.
    matrix: Transport matrix.
  """
  f: jax.Array
  g: jax.Array
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

  @property
  def dual_cost(self) -> jax.Array:
    """Dual transport cost."""
    return jnp.dot(self.ot_prob.a, self.f) + jnp.dot(self.ot_prob.b, self.g)

  @property
  def geom(self) -> pointcloud.PointCloud:  # noqa: D102
    """Geometry."""
    return self.ot_prob.geom


@jtu.register_dataclass
@dataclasses.dataclass(frozen=True)
class SemidiscreteOutput:
  """Output of the :class:`SemidiscreteSolver`.

  Args:
    g: Dual potential.
    prob: Semidiscrete OT problem.
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

  def to_dual_potentials(
      self, epsilon: Optional[float] = None
  ) -> potentials.DualPotentials:
    """TODO.

    Args:
      epsilon: TODO.

    Returns:
      TODO.
    """
    f_fn = self.prob.potential_fn_from_dual_vec(self.g, epsilon=epsilon)
    cost_fn = self.prob.geom.cost_fn
    return potentials.DualPotentials(f=f_fn, g=None, cost_fn=cost_fn)

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

    if self.prob.geom.is_entropy_regularized:
      b, epsilon = self.prob.b, self.prob.geom.epsilon
      f, _ = prob.c_transform(self.g, axis=1)
      # SinkhornOutput's potentials must contain
      # probability weight normalization
      f_tilde = f + epsilon * jnp.log(1.0 / num_samples)
      g_tilde = self.g + epsilon * jnp.where(b > 0.0, jnp.log(b), 0.0)
      return sinkhorn.SinkhornOutput(
          potentials=(f_tilde, g_tilde),
          ot_prob=prob,
      )

    f, _ = prob.c_transform(self.g, axis=1)
    z = self.g[None, :] - prob.geom.cost_matrix
    data = jnp.full((num_samples,), fill_value=1.0 / num_samples, dtype=z.dtype)
    row_ixs = jnp.arange(num_samples)
    col_ixs = jnp.argmax(z, axis=-1)
    matrix = jesp.BCOO(
        (data, jnp.c_[row_ixs, col_ixs]),
        shape=prob.geom.shape,
    )
    return HardAssignmentOutput(
        f=f,
        g=self.g,
        ot_prob=prob,
        matrix=matrix,
    )


@jtu.register_static
@dataclasses.dataclass(frozen=True, kw_only=True)
class SemidiscreteSolver:
  """Semidiscrete optimal transport solver.

  Args:
    num_iterations: Number of iterations.
    batch_size: Number of points to sample at each iteration.
    optimizer: Optimizer.
    error_eval_every: Compute the chi-squared error every ``error_eval_every``
      iterations.
    error_batch_size: Batch size to use when computing
      the marginal chi-squared error. If :obj:`None`, use ``batch_size``.
    error_num_iterations: Number of iterations used to estimate
      the marginal chi-squared error.
    threshold: Convergence threshold for the marginal chi-squared error.
    potential_ema: Exponential moving average of the dual potential.
    callback: Callback with a signature ``(state) -> None`` that is called
      at every iteration.
  """
  num_iterations: int
  batch_size: int
  optimizer: optax.GradientTransformation
  error_eval_every: int = 1000
  error_batch_size: Optional[int] = None
  error_num_iterations: int = 1000
  threshold: float = 1e-3
  potential_ema: float = 0.99
  callback: Optional[Callable[[SemidiscreteState], None]] = None

  def __call__(
      self,
      rng: jax.Array,
      prob: sdlp.SemidiscreteLinearProblem,
      g_init: Optional[jax.Array] = None,
  ) -> SemidiscreteOutput:
    """Run the semidiscrete solver.

    Args:
      rng: Random key used for seeding.
      prob: Semidiscrete OT problem.
      g_init: Initial potential value of shape ``[m,]``. If :obj:`None`,
        use an array of 0s.

    Returns:
      The semidiscrete output.
    """

    def cond_fn(
        it: int,
        prob: sdlp.SemidiscreteLinearProblem,
        state: SemidiscreteState,
    ) -> bool:
      del prob
      loss = state.losses[it - 1]
      err = jnp.abs(state.errors[it // self.error_eval_every - 1])

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
      return self.step(
          jr.fold_in(rng, it),
          state=state,
          prob=prob,
          compute_error=compute_error,
          # we evaluate the error using the same samples
          rng_error=rng_error,
      )

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
        losses=jnp.full((self.num_iterations,), fill_value=jnp.inf,
                        dtype=dtype),
        grad_norms=jnp.full((self.num_iterations,),
                            fill_value=jnp.inf,
                            dtype=dtype),
        errors=jnp.full(
            math.ceil(self.num_iterations / self.error_eval_every),
            fill_value=jnp.inf,
            dtype=dtype
        ),
    )

    rng, rng_error = jr.split(rng, 2)
    state: SemidiscreteState = fixed_point_loop.fixpoint_iter(
        cond_fn,
        body_fn,
        min_iterations=0,
        max_iterations=self.num_iterations,
        inner_iterations=self.error_eval_every,
        constants=prob,
        state=state,
    )

    return self._to_output(state, prob)

  def step(
      self,
      rng: jax.Array,
      state: SemidiscreteState,
      prob: sdlp.SemidiscreteLinearProblem,
      *,
      compute_error: bool = False,
      rng_error: Optional[jax.Array] = None,
  ) -> SemidiscreteState:
    """Perform one optimization step.

    Args:
      rng: Random seed used for sampling.
      state: Semidiscrete state.
      prob: Semidiscrete linear problem.
      compute_error: Whether to compute the marginal chi-squared error.
      rng_error: Random seed when computing the chi-squared error.

    Returns:
      The updated state.
    """
    it = state.it
    rng_error = utils.default_prng_key(rng_error)

    g_old = state.g
    lin_prob = prob.sample(rng, self.batch_size)

    loss, grads = jax.value_and_grad(_semidiscrete_loss)(g_old, lin_prob)
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
        rng_error, g_ema, prob,
        num_iters=self.error_num_iterations,
        batch_size=self.error_batch_size or self.batch_size,
      ),
      lambda: jnp.array(jnp.inf, dtype=state.errors.dtype),
    )
    # fmt: on
    errors = state.errors.at[it // self.error_eval_every].set(error)

    state = SemidiscreteState(
        it=it + 1,
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

  def _to_output(
      self, state: SemidiscreteState, prob: sdlp.SemidiscreteLinearProblem
  ) -> SemidiscreteOutput:
    it = state.it
    leq_thr = state.errors[it // self.error_eval_every] <= self.threshold
    finite_loss = jnp.isfinite(state.losses[it])
    return SemidiscreteOutput(
        g=state.g_ema,
        prob=prob,
        it=it,
        losses=state.losses,
        errors=state.errors,
        converged=jnp.logical_and(leq_thr, finite_loss),
    )


@jax.custom_vjp
def _semidiscrete_loss(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
) -> jax.Array:
  f, _ = prob.c_transform(g, axis=1)
  # we assume uniform weights for `prob.a`
  return -(jnp.mean(f) + jnp.dot(g, prob.b))


def _semidiscrete_loss_fwd(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
) -> Tuple[jax.Array, Tuple[jax.Array, linear_problem.LinearProblem]]:
  f, z = prob.c_transform(g, axis=1)
  # we assume uniform weights for `prob.a`
  return -(jnp.mean(f) + jnp.dot(g, prob.b)), (z, prob)


def _semidiscrete_loss_bwd(
    res: jax.Array,
    g: jax.Array,
) -> Tuple[jax.Array, None]:

  def soft_grad(z: jax.Array) -> jax.Array:
    if prob._b is None:  # uniform weights
      return jsp.special.softmax(z, axis=-1).sum(0)
    return _weighted_softmax(z, b=prob.b, axis=-1).sum(0)

  def hard_grad(z: jax.Array) -> jax.Array:
    ixs = jnp.argmax(z, axis=-1)
    return jax.ops.segment_sum(
        jnp.ones(n, dtype=z.dtype), segment_ids=ixs, num_segments=m
    )

  z, prob = res
  is_soft = prob.geom.epsilon > 0.0
  n, m = prob.geom.shape

  grad = jax.lax.cond(is_soft, soft_grad, hard_grad, z)
  assert grad.shape == (m,), (grad.shape, (m,))

  grad = grad * (1.0 / n) - prob.b
  return g * grad, None


def _weighted_softmax(x: jax.Array, b: jax.Array, axis: int = -1) -> jax.Array:
  where = b > 0.0
  x_max = jnp.max(x, axis=axis, keepdims=True, where=where, initial=-jnp.inf)
  unnormalized = b * jnp.exp(x - x_max)
  softmax = unnormalized / unnormalized.sum(
      axis=axis, keepdims=True, where=where
  )
  return jnp.where(where, softmax, 0.0)


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
