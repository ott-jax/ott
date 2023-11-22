from abc import ABC, abstractmethod
from pathlib import Path
from types import Mapping, MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import optax
from flax import train_state
from jax import random

from ott.geometry.pointcloud import PointCloud
from ott.neural.solvers import models
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


class BaseNeuralSolver(ABC):
  """Base class for neural solvers.

  Args:
    iterations: Number of iterations to train for.
    valid_freq: Frequency at which to run validation.
  """

  def __init__(self, iterations: int, valid_freq: int, **_: Any) -> Any:
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
  def save(self, path: Path):
    """Save the model."""
    pass

  @abstractmethod
  @property
  def is_balanced(self) -> Dict[str, Any]:
    """Return the training logs."""
    pass

  @abstractmethod
  @property
  def training_logs(self) -> Dict[str, Any]:
    """Return the training logs."""
    pass


class ResampleMixin:

  def _resample_data(
      self,
      key: jax.random.KeyArray,
      tmat: jnp.ndarray,
      source_arrays: Tuple[jnp.ndarray, ...],
      target_arrays: Tuple[jnp.ndarray, ...],
  ) -> Tuple[jnp.ndarray, ...]:
    """Resample a batch according to coupling `tmat`."""
    transition_matrix = tmat.flatten()
    indices = random.choice(
        key, transition_matrix.flatten(), shape=[len(transition_matrix) ** 2]
    )
    indices_source = indices // self.batch_size
    indices_target = indices % self.batch_size
    return tuple(
        b[indices_source] if b is not None else None for b in source_arrays
    ), tuple(
        b[indices_target] if b is not None else None for b in target_arrays
    )

  def _resample_data_conditionally(
      self,
      *args: Any,
      **kwargs: Any,
  ):
    raise NotImplementedError


class UnbalancednessMixin:

  def __init__(
      self,
      source_dim: int,
      target_dim: int,
      cond_dim: Optional[int],
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      mlp_eta: Optional[models.ModelBase] = None,
      mlp_xi: Optional[models.ModelBase] = None,
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
