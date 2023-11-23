from abc import ABC, abstractmethod
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random

from ott.geometry import pointcloud
from ott.geometry.pointcloud import PointCloud
from ott.neural.models import models
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn


class BaseNeuralSolver(ABC):
  """Base class for neural solvers.

  Args:
    iterations: Number of iterations to train for.
    valid_freq: Frequency at which to run validation.
  """

  def __init__(self, iterations: int, valid_freq: int, **_: Any) -> None:
    self.iterations = iterations
    self.valid_freq = valid_freq

  @abstractmethod
  def setup(self, *args: Any, **kwargs: Any) -> None:
    pass

  @abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> None:
    """Train the model."""
    pass

  @abstractmethod
  def transport(self, *args: Any, forward: bool, **kwargs: Any) -> Any:
    """Transport."""
    pass

  @abstractmethod
  def save(self, path: Path):
    """Save the model."""
    pass

  @abstractmethod
  def load(self, path: Path):
    """Load the model."""
    pass

  @property
  @abstractmethod
  def training_logs(self) -> Dict[str, Any]:
    """Return the training logs."""
    pass


class ResampleMixin:

  def __init__(*args, **kwargs):
    pass

  def _resample_data(
      self,
      key: jax.random.KeyArray,
      tmat: jnp.ndarray,
      source_arrays: Tuple[jnp.ndarray, ...],
      target_arrays: Tuple[jnp.ndarray, ...],
  ) -> Tuple[jnp.ndarray, ...]:
    """Resample a batch according to coupling `tmat`."""
    tmat_flattened = tmat.flatten()
    indices = random.choice(
        key, len(tmat_flattened), shape=[len(tmat_flattened)]
    )
    indices_source = indices // tmat.shape[1]
    indices_target = indices % tmat.shape[1]
    return tuple(
        b[indices_source, :] if b is not None else None for b in source_arrays
    ), tuple(
        b[indices_target, :] if b is not None else None for b in target_arrays
    )

  def sample_conditional_indices_from_tmap(
      key: jax.random.PRNGKeyArray,
      tmat: jnp.ndarray,
      k_samples_per_x: Union[int, jnp.ndarray],
      source_arrays: Tuple[jnp.ndarray, ...],
      target_arrays: Tuple[jnp.ndarray, ...],
      *,
      is_balanced: bool,
  ) -> Tuple[jnp.array, jnp.array]:
    left_marginals = tmat.sum(axis=1)
    if not is_balanced:
      key, key2 = jax.random.split(key, 2)
      indices = jax.random.choice(
          key=key2,
          a=jnp.arange(len(left_marginals)),
          p=left_marginals,
          shape=(len(left_marginals),)
      )
    else:
      indices = jnp.arange(tmat.shape[0])
    tmat_adapted = tmat[indices]
    indices_per_row = jax.vmap(
        lambda tmat_adapted: jax.random.choice(
            key=key,
            a=jnp.arange(tmat.shape[1]),
            p=tmat_adapted,
            shape=(k_samples_per_x,)
        ),
        in_axes=0,
        out_axes=0,
    )(
        tmat_adapted
    )

    indices_source = jnp.repeat(indices, k_samples_per_x)
    indices_target = indices_per_row % tmat.shape[1]
    return tuple(
        b[indices_source, :] if b is not None else None for b in source_arrays
    ), tuple(
        b[indices_target, :] if b is not None else None for b in target_arrays
    )

  def _get_sinkhorn_match_fn(
      self,
      ot_solver: Any,
      epsilon: float,
      cost_fn: str,
      scale_cost: Any,
      tau_a: float,
      tau_b: float,
  ) -> Callable:

    def match_pairs(
        x: jax.Array, y: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      geom = pointcloud.PointCloud(
          x, y, epsilon=epsilon, scale_cost=scale_cost, cost_fn=cost_fn
      )
      return ot_solver(
          linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
      ).matrix

    return match_pairs

  def _get_gromov_match_fn(
      self,
      ot_solver: Any,
      cost_fn: Union[Any, Mapping[str, Any]],
      scale_cost: Union[Any, Mapping[str, Any]],
      tau_a: float,
      tau_b: float,
      fused_penalty: float,
  ) -> Callable:
    if isinstance(cost_fn, Mapping):
      assert "x_cost_fn" in cost_fn
      assert "y_cost_fn" in cost_fn
      x_cost_fn = cost_fn["x_cost_fn"]
      y_cost_fn = cost_fn["y_cost_fn"]
      if fused_penalty > 0:
        assert "xy_cost_fn" in x_cost_fn
        xy_cost_fn = cost_fn["xy_cost_fn"]
    else:
      x_cost_fn = y_cost_fn = xy_cost_fn = cost_fn

    if isinstance(scale_cost, Mapping):
      assert "x_scale_cost" in scale_cost
      assert "y_scale_cost" in scale_cost
      x_scale_cost = scale_cost["x_scale_cost"]
      y_scale_cost = scale_cost["y_scale_cost"]
      if fused_penalty > 0:
        assert "xy_scale_cost" in scale_cost
        xy_scale_cost = cost_fn["xy_scale_cost"]
    else:
      x_scale_cost = y_scale_cost = xy_scale_cost = scale_cost

    def match_pairs(
        x_quad: Tuple[jnp.ndarray, jnp.ndarray],
        y_quad: Tuple[jnp.ndarray, jnp.ndarray],
        x_lin: Optional[jax.Array],
        y_lin: Optional[jax.Array],
    ) -> Tuple[jnp.array, jnp.array]:
      geom_xx = pointcloud.PointCloud(
          x=x_quad, y=x_quad, cost_fn=x_cost_fn, scale_cost=x_scale_cost
      )
      geom_yy = pointcloud.PointCloud(
          x=y_quad, y=y_quad, cost_fn=y_cost_fn, scale_cost=y_scale_cost
      )
      if fused_penalty > 0:
        geom_xy = pointcloud.PointCloud(
            x=x_lin, y=y_lin, cost_fn=xy_cost_fn, scale_cost=xy_scale_cost
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
      out = ot_solver(prob)
      return out.matrix

    return match_pairs


class UnbalancednessMixin:

  def __init__(
      self,
      source_dim: int,
      target_dim: int,
      cond_dim: Optional[int],
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      mlp_eta: Optional[models.BaseRescalingNet] = None,
      mlp_xi: Optional[models.BaseRescalingNet] = None,
      seed: Optional[int] = None,
      opt_eta: Optional[optax.GradientTransformation] = None,
      opt_xi: Optional[optax.GradientTransformation] = None,
      resample_epsilon: float = 1e-2,
      scale_cost: Union[bool, int, float, Literal["mean", "max_cost",
                                                  "median"]] = "mean",
      sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
      **_: Any,
  ) -> None:
    self.source_dim = source_dim
    self.target_dim = target_dim
    self.cond_dim = cond_dim
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.mlp_eta = mlp_eta
    self.mlp_xi = mlp_xi
    self.seed = seed
    self.opt_eta = opt_eta
    self.opt_xi = opt_xi
    self.resample_epsilon = resample_epsilon
    self.scale_cost = scale_cost

    self._compute_unbalanced_marginals = self._get_compute_unbalanced_marginals(
        tau_a=tau_a,
        tau_b=tau_b,
        resample_epsilon=resample_epsilon,
        scale_cost=scale_cost,
        sinkhorn_kwargs=sinkhorn_kwargs
    )
    self._setup(source_dim=source_dim, target_dim=target_dim, cond_dim=cond_dim)

  def _get_compute_unbalanced_marginals(
      self,
      tau_a: float,
      tau_b: float,
      resample_epsilon: float,
      scale_cost: Union[bool, int, float, Literal["mean", "max_cost",
                                                  "median"]] = "mean",
      sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the unbalanced source and target marginals for a batch."""

    @jax.jit
    def compute_unbalanced_marginals(
        batch_source: jnp.ndarray, batch_target: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
      geom = PointCloud(
          batch_source,
          batch_target,
          epsilon=resample_epsilon,
          scale_cost=scale_cost
      )
      out = sinkhorn.Sinkhorn(**sinkhorn_kwargs)(
          linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
      )
      return out.matrix.sum(axis=1), out.matrix.sum(axis=0)

    return compute_unbalanced_marginals

  @jax.jit
  def _resample_unbalanced(
      self,
      key: jax.random.KeyArray,
      batch: Tuple[jnp.ndarray, ...],
      marginals: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, ...]:
    """Resample a batch based upon marginals."""
    indices = jax.random.choice(
        key, a=len(marginals), p=jnp.squeeze(marginals), shape=[len(marginals)]
    )
    return tuple(b[indices] if b is not None else None for b in batch)

  def _setup(self, source_dim: int, target_dim: int, cond_dim: int):
    self.unbalancedness_step_fn = self._get_step_fn()
    if self.mlp_eta is not None:
      self.opt_eta = (
          self.opt_eta if self.opt_eta is not None else
          optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
      )
      self.state_eta = self.mlp_eta.create_train_state(
          self._key, self.opt_eta, source_dim + cond_dim
      )
    if self.mlp_xi is not None:
      self.opt_xi = (
          self.opt_xi if self.opt_xi is not None else
          optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
      )
      self.state_xi = self.mlp_xi.create_train_state(
          self._key, self.opt_xi, target_dim + cond_dim
      )

  def _get_step_fn(self) -> Callable:  # type:ignore[type-arg]

    def loss_a_fn(
        params_eta: Optional[jnp.ndarray],
        apply_fn_eta: Callable[[Dict[str, jnp.ndarray], jnp.ndarray],
                               jnp.ndarray],
        x: jnp.ndarray,
        a: jnp.ndarray,
        expectation_reweighting: float,
    ) -> Tuple[float, jnp.ndarray]:
      eta_predictions = apply_fn_eta({"params": params_eta}, x)
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
        b: jnp.ndarray,
        expectation_reweighting: float,
    ) -> Tuple[float, jnp.ndarray]:
      xi_predictions = apply_fn_xi({"params": params_xi}, x)
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
      if condition is None:
        input_source = source
        input_target = target
      else:
        input_source = jnp.concatenate([source, condition], axis=-1)
        input_target = jnp.concatenate([target, condition], axis=-1)
      if state_eta is not None:
        grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
        (loss_a, eta_predictions), grads_eta = grad_a_fn(
            state_eta.params,
            state_eta.apply_fn,
            input_source,
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
            input_target,
            b * len(b),
            jnp.sum(a),
        )
        new_state_xi = state_xi.apply_gradients(
            grads=grads_xi
        ) if is_training else None
      else:
        new_state_xi = xi_predictions = loss_b = None

      return new_state_eta, new_state_xi, eta_predictions, xi_predictions, loss_a, loss_b

    return step_fn
