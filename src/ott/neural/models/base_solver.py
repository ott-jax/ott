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
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import optax
from flax.training import train_state

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

__all__ = ["ResampleMixin", "UnbalancednessHandler"]


def _get_sinkhorn_match_fn(
    ot_solver: Any,
    epsilon: float = 1e-2,
    cost_fn: Optional[costs.CostFn] = None,
    scale_cost: Union[bool, int, float, Literal["mean", "max_norm", "max_bound",
                                                "max_cost", "median"]] = "mean",
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    *,
    filter_input: bool = False,
) -> Callable:

  @jax.jit
  def match_pairs(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    geom = pointcloud.PointCloud(
        x, y, epsilon=epsilon, scale_cost=scale_cost, cost_fn=cost_fn
    )
    return ot_solver(
        linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    )

  @jax.jit
  def match_pairs_filtered(
      x_lin: jnp.ndarray, x_quad: jnp.ndarray, y_lin: jnp.ndarray,
      y_quad: jnp.ndarray
  ) -> jnp.ndarray:
    geom = pointcloud.PointCloud(
        x_lin, y_lin, epsilon=epsilon, scale_cost=scale_cost, cost_fn=cost_fn
    )
    return ot_solver(
        linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    )

  return match_pairs_filtered if filter_input else match_pairs


def _get_gromov_match_fn(
    ot_solver: Any,
    cost_fn: Union[Any, Mapping[str, Any]],
    scale_cost: Union[Union[bool, int, float,
                            Literal["mean", "max_norm", "max_bound", "max_cost",
                                    "median"]],
                      Dict[str, Union[bool, int, float,
                                      Literal["mean", "max_norm", "max_bound",
                                              "max_cost", "median"]]]],
    tau_a: float,
    tau_b: float,
    fused_penalty: float,
) -> Callable:
  if isinstance(cost_fn, Mapping):
    assert "cost_fn_xx" in cost_fn
    assert "cost_fn_yy" in cost_fn
    cost_fn_xx = cost_fn["cost_fn_xx"]
    cost_fn_yy = cost_fn["cost_fn_yy"]
    if fused_penalty > 0:
      assert "cost_fn_xy" in cost_fn_xx
      cost_fn_xy = cost_fn["cost_fn_xy"]
  else:
    cost_fn_xx = cost_fn_yy = cost_fn_xy = cost_fn

  if isinstance(scale_cost, Mapping):
    assert "scale_cost_xx" in scale_cost
    assert "scale_cost_yy" in scale_cost
    scale_cost_xx = scale_cost["scale_cost_xx"]
    scale_cost_yy = scale_cost["scale_cost_yy"]
    if fused_penalty > 0:
      assert "scale_cost_xy" in scale_cost
      scale_cost_xy = cost_fn["scale_cost_xy"]
  else:
    scale_cost_xx = scale_cost_yy = scale_cost_xy = scale_cost

  @jax.jit
  def match_pairs(
      x_lin: Optional[jnp.ndarray],
      x_quad: Tuple[jnp.ndarray, jnp.ndarray],
      y_lin: Optional[jnp.ndarray],
      y_quad: Tuple[jnp.ndarray, jnp.ndarray],
  ) -> jnp.ndarray:
    geom_xx = pointcloud.PointCloud(
        x=x_quad, y=x_quad, cost_fn=cost_fn_xx, scale_cost=scale_cost_xx
    )
    geom_yy = pointcloud.PointCloud(
        x=y_quad, y=y_quad, cost_fn=cost_fn_yy, scale_cost=scale_cost_yy
    )
    if fused_penalty > 0:
      geom_xy = pointcloud.PointCloud(
          x=x_lin, y=y_lin, cost_fn=cost_fn_xy, scale_cost=scale_cost_xy
      )
    else:
      geom_xy = None
    prob = quadratic_problem.QuadraticProblem(
        geom_xx,
        geom_yy,
        geom_xy,
        fused_penalty=fused_penalty,
        tau_a=tau_a,
        tau_b=tau_b
    )
    return ot_solver(prob)

  return match_pairs


class ResampleMixin:
  """Mixin class for mini-batch OT in neural optimal transport solvers."""

  def _resample_data(
      self,
      rng: jax.Array,
      tmat: jnp.ndarray,
      source_arrays: Tuple[jnp.ndarray, ...],
      target_arrays: Tuple[jnp.ndarray, ...],
  ) -> Tuple[jnp.ndarray, ...]:
    """Resample a batch according to coupling `tmat`."""
    tmat_flattened = tmat.flatten()
    indices = jax.random.choice(rng, len(tmat_flattened), shape=[tmat.shape[0]])
    indices_source = indices // tmat.shape[1]
    indices_target = indices % tmat.shape[1]
    return tuple(
        b[indices_source] if b is not None else None for b in source_arrays
    ), tuple(
        b[indices_target] if b is not None else None for b in target_arrays
    )

  def _sample_conditional_indices_from_tmap(
      self,
      rng: jax.Array,
      tmat: jnp.ndarray,
      k_samples_per_x: Union[int, jnp.ndarray],
      source_arrays: Tuple[jnp.ndarray, ...],
      target_arrays: Tuple[jnp.ndarray, ...],
      *,
      source_is_balanced: bool,
  ) -> Tuple[jnp.array, jnp.array]:
    batch_size = tmat.shape[0]
    left_marginals = tmat.sum(axis=1)
    if not source_is_balanced:
      rng, rng_2 = jax.random.split(rng, 2)
      indices = jax.random.choice(
          key=rng_2,
          a=jnp.arange(len(left_marginals)),
          p=left_marginals,
          shape=(len(left_marginals),)
      )
    else:
      indices = jnp.arange(batch_size)
    tmat_adapted = tmat[indices]
    indices_per_row = jax.vmap(
        lambda row: jax.random.choice(
            key=rng, a=jnp.arange(batch_size), p=row, shape=(k_samples_per_x,)
        ),
        in_axes=0,
        out_axes=0,
    )(
        tmat_adapted
    )

    indices_source = jnp.repeat(indices, k_samples_per_x)
    indices_target = jnp.reshape(
        indices_per_row % tmat.shape[1], (batch_size * k_samples_per_x,)
    )
    return tuple(
        jnp.reshape(b[indices_source], (k_samples_per_x, batch_size,
                                        -1)) if b is not None else None
        for b in source_arrays
    ), tuple(
        jnp.reshape(b[indices_target], (k_samples_per_x, batch_size,
                                        -1)) if b is not None else None
        for b in target_arrays
    )

  def _get_sinkhorn_match_fn(self, *args, **kwargs) -> jnp.ndarray:
    fn = _get_sinkhorn_match_fn(*args, **kwargs)

    @jax.jit
    def match_pairs(*args, **kwargs):
      return fn(*args, **kwargs).matrix

    return match_pairs

  def _get_gromov_match_fn(self, *args, **kwargs) -> jnp.ndarray:
    fn = _get_gromov_match_fn(*args, **kwargs)

    @jax.jit
    def match_pairs(*args, **kwargs):
      return fn(*args, **kwargs).matrix

    return match_pairs


class UnbalancednessHandler:
  """Class to incorporate unbalancedness into neural OT models.

  This class implements the concepts introduced in :cite:`eyring:23`
  in the Monge Map scenario and :cite:`klein:23` for the entropic OT case
  for linear and quadratic cases.

  Args:
    rng: Random number generator.
    source_dim: Dimension of the source domain.
    target_dim: Dimension of the target domain.
    cond_dim: Dimension of the conditioning variable.
    If :obj:`None`, no conditioning is used.
    tau_a: Unbalancedness parameter for the source distribution.
    tau_b: Unbalancedness parameter for the target distribution.
    rescaling_a: Rescaling function for the source distribution.
    If :obj:`None`, the left rescaling factor is not learnt.
    rescaling_b: Rescaling function for the target distribution.
    If :obj:`None`, the right rescaling factor is not learnt.
    opt_eta: Optimizer for the left rescaling function.
    opt_xi: Optimzier for the right rescaling function.
    resample_epsilon: Epsilon for resampling.
    scale_cost: Scaling of the cost matrix for estimating the rescaling factors.
    ot_solver: Solver to compute unbalanced marginals. If `ot_solver` is `None`,
    the method
    :meth:`ott.neural.models.base_solver.UnbalancednessHandler.compute_unbalanced_marginals`
    is not available, and hence the unbalanced marginals must be computed by the neural solver.
    kwargs: Additional keyword arguments.

  """

  def __init__(
      self,
      rng: jax.Array,
      source_dim: int,
      target_dim: int,
      cond_dim: Optional[int],
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      rescaling_a: Optional[Callable[[jnp.ndarray, Optional[jnp.ndarray]],
                                     jnp.ndarray]] = None,
      rescaling_b: Optional[Callable[[jnp.ndarray, Optional[jnp.ndarray]],
                                     jnp.ndarray]] = None,
      opt_eta: Optional[optax.GradientTransformation] = None,
      opt_xi: Optional[optax.GradientTransformation] = None,
      resample_epsilon: float = 1e-2,
      scale_cost: Union[bool, int, float, Literal["mean", "max_cost",
                                                  "median"]] = "mean",
      ot_solver: Optional[was_solver.WassersteinSolver] = None,
      **kwargs: Mapping[str, Any],
  ):
    self.rng_unbalanced = rng
    self.source_dim = source_dim
    self.target_dim = target_dim
    self.cond_dim = cond_dim
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.rescaling_a = rescaling_a
    self.rescaling_b = rescaling_b
    self.opt_eta = opt_eta
    self.opt_xi = opt_xi
    self.resample_epsilon = resample_epsilon
    self.scale_cost = scale_cost
    self.ot_solver = ot_solver

    if isinstance(ot_solver, sinkhorn.Sinkhorn):
      self.compute_unbalanced_marginals = (
          self._get_compute_unbalanced_marginals_lin(
              tau_a=tau_a,
              tau_b=tau_b,
              resample_epsilon=resample_epsilon,
              scale_cost=scale_cost,
              **kwargs
          )
      )
    elif isinstance(ot_solver, gromov_wasserstein.GromovWasserstein):
      self.compute_unbalanced_marginals = self._get_compute_unbalanced_marginals_quad
    self.setup(source_dim=source_dim, target_dim=target_dim, cond_dim=cond_dim)

  def _get_compute_unbalanced_marginals_lin(
      self, *args: Any, **kwargs: Mapping[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the unbalanced source and target marginals for a batch."""
    fn = _get_sinkhorn_match_fn(*args, **kwargs)

    @jax.jit
    def compute_unbalanced_marginals_lin(*args, **kwargs):
      out = fn(*args, **kwargs)
      return out.marginals(axis=1), out.marginals(axis=0)

    return compute_unbalanced_marginals_lin

  def _get_compute_unbalanced_marginals_quad(
      self, *args: Any, **kwargs: Mapping[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the unbalanced source and target marginals for a batch."""
    fn = _get_sinkhorn_match_fn(*args, **kwargs)

    @jax.jit
    def compute_unbalanced_marginals_quad(*args, **kwargs):
      out = fn(*args, **kwargs)
      return out.marginals(axis=1), out.marginals(axis=0)

    return compute_unbalanced_marginals_quad

  @jax.jit
  def _resample_unbalanced(
      self,
      rng: jax.Array,
      batch: Tuple[jnp.ndarray, ...],
      marginals: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, ...]:
    """Resample a batch based on marginals."""
    indices = jax.random.choice(
        rng, a=len(marginals), p=jnp.squeeze(marginals), shape=[len(marginals)]
    )
    return tuple(b[indices] if b is not None else None for b in batch)

  def setup(self, source_dim: int, target_dim: int, cond_dim: int):
    """Setup the model.

    Args:
      source_dim: Dimension of the source domain.
      target_dim: Dimension of the target domain.
      cond_dim: Dimension of the conditioning variable.
      If :obj:`None`, no conditioning is used.
    """
    self.rng_unbalanced, rng_eta, rng_xi = jax.random.split(
        self.rng_unbalanced, 3
    )
    self.step_fn = self._get_rescaling_step_fn()
    if self.rescaling_a is not None:
      self.opt_eta = (
          self.opt_eta if self.opt_eta is not None else
          optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
      )
      self.state_eta = self.rescaling_a.create_train_state(
          rng_eta, self.opt_eta, source_dim
      )
    if self.rescaling_b is not None:
      self.opt_xi = (
          self.opt_xi if self.opt_xi is not None else
          optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
      )
      self.state_xi = self.rescaling_b.create_train_state(
          rng_xi, self.opt_xi, target_dim
      )

  def _get_rescaling_step_fn(self) -> Callable:  # type:ignore[type-arg]

    def loss_a_fn(
        params_eta: Optional[jnp.ndarray],
        apply_fn_eta: Callable[[Dict[str, jnp.ndarray], jnp.ndarray],
                               jnp.ndarray],
        x: jnp.ndarray,
        condition: Optional[jnp.ndarray],
        a: jnp.ndarray,
        expectation_reweighting: float,
    ) -> Tuple[float, jnp.ndarray]:
      eta_predictions = apply_fn_eta({"params": params_eta}, x, condition)
      return (
          optax.l2_loss(eta_predictions[:, 0], a).mean() +
          optax.l2_loss(jnp.mean(eta_predictions) - expectation_reweighting),
          eta_predictions,
      )

    def loss_b_fn(
        params_xi: Optional[jnp.ndarray],
        apply_fn_xi: Callable[[Dict[str, jnp.ndarray], jnp.ndarray],
                              jnp.ndarray],
        x: jnp.ndarray,
        condition: Optional[jnp.ndarray],
        b: jnp.ndarray,
        expectation_reweighting: float,
    ) -> Tuple[float, jnp.ndarray]:
      xi_predictions = apply_fn_xi({"params": params_xi}, x, condition)
      return (
          optax.l2_loss(xi_predictions[:, 0], b).mean() +
          optax.l2_loss(jnp.mean(xi_predictions) - expectation_reweighting),
          xi_predictions,
      )

    @jax.jit
    def step_fn(
        source: jnp.ndarray,
        target: jnp.ndarray,
        condition: Optional[jnp.ndarray],
        a: jnp.ndarray,
        b: jnp.ndarray,
        state_eta: Optional[train_state.TrainState] = None,
        state_xi: Optional[train_state.TrainState] = None,
        *,
        is_training: bool = True,
    ):
      if state_eta is not None:
        grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
        (loss_a, eta_predictions), grads_eta = grad_a_fn(
            state_eta.params,
            state_eta.apply_fn,
            source,
            condition,
            a * len(a),
            jnp.sum(b),
        )
        new_state_eta = state_eta.apply_gradients(
            grads=grads_eta
        ) if is_training else None

      else:
        new_state_eta = eta_predictions = loss_a = None
      if state_xi is not None:
        grad_b_fn = jax.value_and_grad(loss_b_fn, argnums=0, has_aux=True)
        (loss_b, xi_predictions), grads_xi = grad_b_fn(
            state_xi.params,
            state_xi.apply_fn,
            target,
            condition,
            b * len(b),
            jnp.sum(a),
        )
        new_state_xi = state_xi.apply_gradients(
            grads=grads_xi
        ) if is_training else None
      else:
        new_state_xi = xi_predictions = loss_b = None

      return (
          new_state_eta, new_state_xi, eta_predictions, xi_predictions, loss_a,
          loss_b
      )

    return step_fn

  def evaluate_eta(
      self,
      source: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None
  ) -> jnp.ndarray:
    """Evaluate the left learnt rescaling factor.

    Args:
      source: Samples from the source distribution to evaluate rescaling
        function on.
      condition: Condition belonging to the samples in the source distribution.

    Returns:
      Learnt left rescaling factors.
    """
    if self.state_eta is None:
      raise ValueError("The left rescaling factor was not parameterized.")
    return self.state_eta.apply_fn({"params": self.state_eta.params},
                                   x=source,
                                   condition=condition)

  def evaluate_xi(
      self,
      target: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None
  ) -> jnp.ndarray:
    """Evaluate the right learnt rescaling factor.

    Args:
      target: Samples from the target distribution to evaluate the rescaling
        function on.
      condition: Condition belonging to the samples in the target distribution.

    Returns:
      Learnt right rescaling factors.
    """
    if self.state_xi is None:
      raise ValueError("The right rescaling factor was not parameterized.")
    return self.state_xi.apply_fn({"params": self.state_xi.params},
                                  x=target,
                                  condition=condition)
