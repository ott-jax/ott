import functools
import types
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

import diffrax
import jax
import jax.numpy as jnp
import optax
from jax import random
from orbax import checkpoint

from ott.geometry import costs, pointcloud
from ott.neural.models.models import BaseNeuralVectorField
from ott.neural.solvers.base_solver import (
    BaseNeuralSolver,
    ResampleMixin,
    UnbalancednessMixin,
)
from ott.neural.solvers.flows import (
    BaseFlow,
)
from ott.problems.linear import linear_problem
from ott.solvers import was_solver


class FlowMatching(UnbalancednessMixin, ResampleMixin, BaseNeuralSolver):

  def __init__(
      self,
      neural_vector_field: Type[BaseNeuralVectorField],
      input_dim: int,
      cond_dim: int,
      iterations: int,
      valid_freq: int,
      ot_solver: Type[was_solver.WassersteinSolver],
      flow: Type[BaseFlow],
      optimizer: Type[optax.GradientTransformation],
      checkpoint_manager: Type[checkpoint.CheckpointManager] = None,
      epsilon: float = 1e-2,
      cost_fn: Type[costs.CostFn] = costs.SqEuclidean(),
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      mlp_eta: Callable[[jnp.ndarray], float] = None,
      mlp_xi: Callable[[jnp.ndarray], float] = None,
      unbalanced_kwargs: Dict[str, Any] = {},
      callback_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                     Any]] = None,
      rng: random.PRNGKeyArray = random.PRNGKey(0),
      **kwargs: Any,
  ) -> None:
    BaseNeuralSolver.__init__(
        self, iterations=iterations, valid_freq=valid_freq
    )
    ResampleMixin.__init__(self)
    UnbalancednessMixin.__init__(
        self,
        source_dim=input_dim,
        target_dim=input_dim,
        cond_dim=cond_dim,
        tau_a=tau_a,
        tau_b=tau_b,
        mlp_eta=mlp_eta,
        mlp_xi=mlp_xi,
        unbalanced_kwargs=unbalanced_kwargs,
    )

    self.neural_vector_field = neural_vector_field
    self.input_dim = input_dim
    self.ot_solver = ot_solver
    self.flow = flow
    self.optimizer = optimizer
    self.epsilon = epsilon
    self.cost_fn = cost_fn
    self.callback_fn = callback_fn
    self.checkpoint_manager = checkpoint_manager
    self.rng = rng

    self.setup()

  def setup(self) -> None:
    self.state_neural_vector_field = self.neural_vector_field.create_train_state(
        self.rng, self.optimizer, self.input_dim
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

  def _get_step_fn(self) -> Callable:

    def step_fn(
        key: random.PRNGKeyArray,
        state_neural_vector_field: Any,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[Any, Any]:

      def loss_fn(
          params: jax.Array, t: jax.Array, noise: jax.Array,
          batch: Dict[str, jnp.ndarray], keys_model: random.PRNGKeyArray
      ) -> jnp.ndarray:

        x_t = self.flow.compute_xt(noise, t, batch["source"], batch["target"])
        apply_fn = functools.partial(
            state_neural_vector_field.apply_fn, {"params": params}
        )
        v_t = jax.vmap(apply_fn)(
            t=t, x=x_t, condition=batch["condition"], keys_model=keys_model
        )
        u_t = self.flow.compute_ut(t, batch["source"], batch["target"])
        return jnp.mean((v_t - u_t) ** 2)

      batch_size = len(batch["source"])
      key_noise, key_t, key_model = random.split(key, 3)
      keys_model = random.split(key_model, batch_size)
      t = self.sample_t(key_t, batch_size)
      noise = self.sample_noise(key_noise, batch_size)
      loss_grad = jax.value_and_grad(loss_fn)
      loss, grads = loss_grad(
          state_neural_vector_field.params, t, noise, batch, keys_model
      )
      return state_neural_vector_field.apply_gradients(grads=grads), loss

    return step_fn

  def _get_match_fn(
      self,
      ot_solver: Any,
      epsilon: float,
      cost_fn: str,
      tau_a: float,
      tau_b: float,
      scale_cost: Any,
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

  def __call__(self, train_loader, valid_loader) -> None:
    batch: Mapping[str, jnp.ndarray] = {}
    for iter in range(self.iterations):
      rng_resample, rng_step_fn, self.rng = random.split(self.rng, 3)
      batch["source"], batch["target"], batch["condition"] = next(train_loader)
      tmat = self.match_fn(batch["source"], batch["target"])
      (batch["source"],
       batch["condition"]), (batch["target"],) = self._resample_data(
           rng_resample, tmat, (batch["source"], batch["condition"]),
           (batch["target"],)
       )
      self.state_neural_vector_field, loss = self.step_fn(
          rng_step_fn, self.state_neural_vector_field, batch
      )
      if self.learn_rescaling:
        self.state_eta, self.state_xi, eta_predictions, xi_predictions, loss_a, loss_b = self.unbalancedness_step_fn(
            batch, tmat.sum(axis=1), tmat.sum(axis=0)
        )
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

  def transport(
      self,
      data: jnp.array,
      condition: Optional[jax.Array],
      rng: random.PRNGKey,
      forward: bool = True,
      diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({})
  ) -> diffrax.Solution:
    diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
    t0, t1 = (0, 1) if forward else (1, 0)
    return diffrax.diffeqsolve(
        diffrax.ODETerm(
            lambda t, y: self.state_neural_vector_field.
            apply({"params": self.state_neural_vector_field.params},
                  t=t,
                  x=y,
                  condition=condition)
        ),
        diffeqsolve_kwargs.pop("solver", diffrax.Tsit5()),
        t0=t0,
        t1=t1,
        dt0=diffeqsolve_kwargs.pop("dt0", None),
        y0=data,
        stepsize_controller=diffeqsolve_kwargs.pop(
            "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
        ),
        **diffeqsolve_kwargs,
    )

  def _valid_step(self, valid_loader, iter) -> None:
    next(valid_loader)
    # TODO: add callback and logging

  @property
  def learn_rescaling(self) -> bool:
    return self.mlp_eta is not None or self.mlp_xi is not None

  def save(self, path: str) -> None:
    raise NotImplementedError

  def training_logs(self) -> Dict[str, Any]:
    raise NotImplementedError

  def sample_t( #TODO: make more general
      self, key: random.PRNGKey, batch_size: int
  ) -> jnp.ndarray:  #TODO: make more general
    return random.uniform(key, [batch_size, 1])

  def sample_noise( #TODO: make more general
      self, key: random.PRNGKey, batch_size: int
  ) -> jnp.ndarray:  #TODO: make more general
    return random.normal(key, shape=(batch_size, self.input_dim))
