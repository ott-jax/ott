from typing import Any, Callable, Dict, Optional, Type

import jax.numpy as jnp
import orbax as obx

from ott.geometry import costs
from ott.neural.models.models import BaseNeuralVectorField
from ott.neural.solver.base_solver import BaseNeuralSolver, UnbalancednessMixin
from ott.solvers import was_solver


class FlowMatching(BaseNeuralSolver, UnbalancednessMixin):

  def __init__(
      self,
      neural_vector_field: Type[BaseNeuralVectorField],
      input_dim: int,
      iterations: int,
      valid_freq: int,
      ot_solver: Type[was_solver.WassersteinSolver],
      optimizer: Optional[Any] = None,
      checkpoint_manager: Type[obx.CheckpointManager] = None,
      epsilon: float = 1e-2,
      cost_fn: Type[costs.CostFn] = costs.SqEuclidean(),
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      mlp_eta: Callable[[jnp.ndarray], float] = None,
      mlp_xi: Callable[[jnp.ndarray], float] = None,
      unbalanced_kwargs: Dict[str, Any] = {},
      callback_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                     Any]] = None,
      seed: int = 0,
      **kwargs: Any,
  ) -> None:

    super().__init__(iterations=iterations, valid_freq=valid_freq)
    super(UnbalancednessMixin, self).__init__(
        mlp_eta=mlp_eta,
        mlp_xi=mlp_xi,
        tau_a=tau_a,
        tau_b=tau_b,
        **unbalanced_kwargs
    )
    self.neural_vector_field = neural_vector_field
    self.input_dim = input_dim
    self.ot_solver = ot_solver
    self.optimizer = optimizer
    self.epsilon = epsilon
    self.cost_fn = cost_fn
    self.callback_fn = callback_fn
    self.checkpoint_manager = checkpoint_manager
    self.seed = seed

  def setup(self, **kwargs: Any) -> None:
    self.state_neural_vector_field = self.neural_vector_field.create_train_state(
        self.rng, self.optimizer, self.output_dim
    )

    self.step_fn = self._get_step_fn()

    self.match_fn = self._get_match_fn(
        self.ot_solver,
        epsilon=self.epsilon,
        cost_fn=self.cost_fn,
        tau_a=self.tau_a,
        tau_b=self.tau_b,
        scale_cost=self.scale_cost,
    )

  def _get_match_fn(self):
    pass

  def __call__(self, train_loader, valid_loader) -> None:
    for iter in range(self.iterations):
      batch = next(train_loader)
      batch, a, b = self.match_fn(batch)
      self.state_neural_vector_field, logs = self.step_fn(
          self.state_neural_vector_field, batch
      )
      if not self.is_balanced:
        self.unbalancedness_step_fn(batch, a, b)
      if iter % self.valid_freq == 0:
        self._valid_step(valid_loader, iter)
        if self.checkpoint_manager is not None:
          states_to_save = {
              "state_neural_vector_field": self.state_neural_vector_field
          }
          if self.state_mlp is not None:
            states_to_save["state_eta"] = self.state_mlp
          if self.state_xi is not None:
            states_to_save["state_xi"] = self.state_xi
          self.checkpoint_manager.save(iter, states_to_save)

  def _valid_step(self, valid_loader, iter) -> None:
    batch = next(valid_loader)
    batch, a, b = self.match_fn(batch)
    if not self.is_balanced:
      self.unbalancedness_step_fn(batch, a, b)
    if self.callback_fn is not None:
      self.callback_fn(batch, a, b)
