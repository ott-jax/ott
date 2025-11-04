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
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu

import optax

from ott import utils
from ott.geometry import geometry, pointcloud
from ott.geometry import semidiscrete_pointcloud as sdpc
from ott.math import fixed_point_loop
from ott.problems.linear import linear_problem, potentials
from ott.problems.linear import semidiscrete_linear_problem as sdlp
from ott.solvers.linear import sinkhorn

if TYPE_CHECKING:
  from ott.neural.data import semidiscrete_dataloader

__all__ = [
    "SemidiscreteState",
    "HardAssignmentOutput",
    "SemidiscreteOutput",
    "SemidiscreteSolver",
    "constant_epsilon_scheduler",
]


@jtu.register_dataclass
@dataclasses.dataclass(frozen=True)
class SemidiscreteState:
  """State of the :class:`SemidiscreteSolver`.

  Args:
    it: Iteration number.
    epsilon: Epsilon value at the current iteration.
    g: Dual potential.
    g_ema: Exponential moving average of the dual potential.
    opt_state: State of the optimizer.
    losses: Dual losses.
    grad_norms: Norms of the gradients.
    errors: Marginal deviation errors.
  """
  it: jax.Array
  epsilon: jax.Array
  g: jax.Array
  g_ema: jax.Array
  opt_state: Any
  losses: jax.Array
  grad_norms: jax.Array
  errors: jax.Array


@jtu.register_dataclass
@dataclasses.dataclass(frozen=True)
class HardAssignmentOutput:
  r"""Unregularized linear OT solution.

  Args:
    ot_prob: Linear OT problem.
    paired_indices: Array of shape ``[2, n]``, of :math:`n` pairs
      of indices, for which the optimal transport assigns mass. Namely, for each
      index :math:`0 \leq k < n`, if one has
      :math:`i := \text{paired_indices}[0, k]` and
      :math:`j := \text{paired_indices}[1, k]`, then point :math:`i` in
      the first geometry sends mass to point :math:`j` in the second.
    f: The first dual potential.
    g: The second dual potential.
  """
  ot_prob: linear_problem.LinearProblem
  paired_indices: jax.Array
  f: Optional[jax.Array] = None
  g: Optional[jax.Array] = None

  @property
  def matrix(self) -> jesp.BCOO:
    """Transport matrix of shape ``[n, m]`` with ``n`` non-zero entries."""
    n, m = self.geom.shape
    unit_mass = jnp.full(n, fill_value=1.0 / n, dtype=self.geom.dtype)
    indices = self.paired_indices.T
    return jesp.BCOO((unit_mass, indices), shape=(n, m))

  @property
  def primal_cost(self) -> jax.Array:
    """Transport cost of the linear OT solution."""
    geom = self.geom
    weights = self.matrix.data
    i, j = self.paired_indices[0], self.paired_indices[1]
    if isinstance(geom, pointcloud.PointCloud):
      cost = jax.vmap(geom.cost_fn)(geom.x[i], geom.y[j])
    else:
      cost = geom.cost_matrix[i, j]
    return jnp.sum(weights * cost)

  @property
  def dual_cost(self) -> jax.Array:
    """Dual transport cost."""
    assert self.f is not None, "Dual potential `f` is not computed."
    assert self.g is not None, "Dual potential `g` is not computed."
    return jnp.dot(self.ot_prob.a, self.f) + jnp.dot(self.ot_prob.b, self.g)

  @property
  def geom(self) -> geometry.Geometry:  # noqa: D102
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
      self,
      rng: jax.Array,
      num_samples: int,
      *,
      epsilon: Optional[float] = None,
  ) -> Union[sinkhorn.SinkhornOutput, HardAssignmentOutput]:
    """Sample a point cloud and compute the OT solution.

    Args:
      rng: Random key used for seeding.
      num_samples: Number of samples.
      epsilon: Epsilon regularization. If :obj:`None`, use the one stored
        in the :attr:`geometry <geom>`.

    Returns:
      The sampled output.
    """
    prob = self.prob.sample(rng, num_samples, epsilon=epsilon)
    is_entreg = self.geom.is_entropy_regularized if epsilon is None else (
        epsilon > 0.0
    )

    if is_entreg:
      f, _ = prob._c_transform(self.g, axis=1)
      # SinkhornOutput's potentials must contain prob. weight normalization
      f_tilde = f + prob.epsilon * jnp.log(1.0 / num_samples)
      g_tilde = self.g + prob.epsilon * jnp.log(prob.b)

      return sinkhorn.SinkhornOutput(
          potentials=(f_tilde, g_tilde),
          ot_prob=prob,
      )

    f, _ = prob._c_transform(self.g, axis=1)
    z = self.g[None, :] - prob.geom.cost_matrix
    row_ixs = jnp.arange(num_samples)
    col_ixs = jnp.argmax(jnp.where(prob.b[None, :], z, -jnp.inf), axis=-1)

    return HardAssignmentOutput(
        prob,
        paired_indices=jnp.stack([row_ixs, col_ixs]),
        f=f,
        g=self.g,
    )

  def to_dual_potentials(
      self, epsilon: Optional[float] = None
  ) -> potentials.DualPotentials:
    """Compute the dual potential function :math:`f`.

    Args:
      epsilon: Epsilon regularization. If :obj:`None`, use the one stored
        in the :attr:`geometry <geom>`.

    Returns:
      The dual potential :math:`f`.
    """
    f_fn = self.prob.potential_fn_from_dual_vec(self.g, epsilon=epsilon)
    cost_fn = self.geom.cost_fn
    return potentials.DualPotentials(f=f_fn, g=None, cost_fn=cost_fn)

  def to_dataloader(
      self, rng: jax.Array, batch_size: int, **kwargs: Any
  ) -> "semidiscrete_dataloader.SemidiscreteDataloader":
    """Create a semidiscrete dataloader.

    Args:
      rng: Random number seed used for sampling from the source distribution.
      batch_size: Batch size used in the dataloader to sample from source.
      kwargs: Keyword arguments for
        :class:`~ott.neural.data.semidiscrete_dataloader.SemidiscreteDataloader`.

    Returns:
      The semidiscrete dataloader.
    """  # noqa: E501
    from ott.neural.data import semidiscrete_dataloader

    return semidiscrete_dataloader.SemidiscreteDataloader(
        rng,
        sd_out=self,
        batch_size=batch_size,
        **kwargs,
    )

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

  @property
  def geom(self) -> sdpc.SemidiscretePointCloud:
    """Semidiscrete geometry."""
    return self.prob.geom


def constant_epsilon_scheduler(
    step: jax.Array, target_epsilon: jax.Array
) -> jax.Array:
  """Constant epsilon scheduler.

  Args:
    step: Current step (ignored).
    target_epsilon: Epsilon at the last iteration.

  Returns:
    The target epsilon.
  """
  del step
  return target_epsilon


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
    error_num_repeats: Number of repeats used to estimate
      the marginal chi-squared error, set to sixteen by default.
    threshold: Convergence threshold for the marginal chi-squared error.
    potential_ema: Exponential moving average of the dual potential.
    epsilon_scheduler: Epsilon scheduler along the iterations with a signature
      ``(step, target_epsilon) -> epsilon``.
      By default, :func:`constant_epsilon_scheduler` is used.
    callback: Callback with a signature ``(state) -> None`` that is called
      at every iteration.
  """
  num_iterations: int
  batch_size: int
  optimizer: optax.GradientTransformation
  error_eval_every: int = 1000
  error_batch_size: Optional[int] = None
  error_num_repeats: int = 16
  threshold: float = 1e-3
  potential_ema: float = 0.99
  epsilon_scheduler: Callable[[jax.Array, jax.Array],
                              jax.Array] = constant_epsilon_scheduler
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
      loss = state.losses[it - 1]
      err = jnp.abs(state.errors[it // self.error_eval_every - 1])
      not_converged = err > self.threshold
      not_diverged = jnp.isfinite(loss)
      target_eps_not_reached = ~jnp.isclose(state.epsilon, prob.epsilon)
      # cont. if not converged and not diverged or not reached target epsilon
      return jnp.logical_or(
          it == 0,
          jnp.logical_or(
              jnp.logical_and(not_converged, not_diverged),
              target_eps_not_reached
          )
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
        epsilon=jnp.nan,
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
    epsilon = self.epsilon_scheduler(it, prob.epsilon)
    lin_prob = prob.sample(rng, self.batch_size, epsilon=epsilon)

    loss, grads = jax.value_and_grad(_semidiscrete_loss)(g_old, lin_prob)
    grad_norm = jnp.linalg.norm(grads)
    losses = state.losses.at[it].set(loss)
    grad_norms = state.grad_norms.at[it].set(grad_norm)

    updates, opt_state = self.optimizer.update(
        grads, state.opt_state, g_old, value=loss
    )
    g_new = optax.apply_updates(g_old, updates)
    g_ema = optax.incremental_update(g_new, state.g_ema, self.potential_ema)

    # fmt: off
    error = jax.lax.cond(
      compute_error,
      lambda: _marginal_chi2_error(
        rng_error, g_ema, prob,
        num_iters=self.error_num_repeats,
        batch_size=self.error_batch_size or self.batch_size,
      ),
      lambda: jnp.array(jnp.inf, dtype=state.errors.dtype),
    )
    # fmt: on
    errors = state.errors.at[it // self.error_eval_every].set(error)

    state = SemidiscreteState(
        it=it + 1,
        epsilon=epsilon,
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
  f, _ = prob._c_transform(g, axis=1)
  # we assume uniform weights for `prob.a`
  return -(jnp.mean(f) + jnp.dot(g, prob.b))


def _semidiscrete_loss_fwd(
    g: jax.Array,
    prob: linear_problem.LinearProblem,
) -> Tuple[jax.Array, Tuple[jax.Array, linear_problem.LinearProblem]]:
  f, z = prob._c_transform(g, axis=1)
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
    pos_weights = prob.b[None, :] > 0.0
    ixs = jnp.argmax(jnp.where(pos_weights, z, -jnp.inf), axis=-1)
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
    """Compute chi2 metric.

    Implements Eq. 3.5  in https://arxiv.org/pdf/2509.25519v1,
    from reference :cite:`mousavi:25`.

    Args:
      matrix: coupling matrix, either dense or sparse.

    Returns:
      Chi2 metric estimator, lowerbounded by -1.0
    """
    # normalize coupling matrix to have columns sum to 1.
    matrix = batch_size * matrix

    if isinstance(matrix, jesp.BCOO):
      out = jesp.bcoo_reduce_sum(matrix, axes=(0,)).todense() ** 2
      out -= jesp.bcoo_reduce_sum(matrix ** 2, axes=(0,)).todense()
    else:
      out = jnp.sum(matrix, axis=0) ** 2 - jnp.sum(matrix ** 2, axis=0)

    out = jnp.sum(out / prob.b) / (batch_size * (batch_size - 1.0))
    return out - 1.0

  def body(chi2_err_avg: jax.Array, it: jax.Array) -> Tuple[jax.Array, None]:
    rng_it = jr.fold_in(rng, it)
    matrix = out.sample(rng_it, batch_size).matrix
    chi2 = compute_chi2(matrix)
    chi2_err_avg = chi2_err_avg + chi2 / num_iters
    return chi2_err_avg, None

  out = SemidiscreteOutput(g=g, prob=prob)

  chi2_err = jnp.zeros((), dtype=g.dtype)
  chi2_err, _ = jax.lax.scan(body, init=chi2_err, xs=jnp.arange(num_iters))
  return chi2_err
