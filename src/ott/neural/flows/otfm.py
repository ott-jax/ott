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
import collections
import functools
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp

import diffrax
import optax
from flax.training import train_state

from ott import utils
from ott.geometry import costs
from ott.neural.flows import flows
from ott.neural.models import base_solver

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
  """(Optimal transport) flow matching class.

  Flow matching as introduced in :cite:`lipman:22`, with extension to OT-FM
  (:cite`tong:23`, :cite:`pooladian:23`).

  Args:
    velocity_field: Neural vector field parameterized by a neural network.
    input_dim: Dimension of the input data.
    cond_dim: Dimension of the conditioning variable.
    iterations: Number of iterations.
    valid_freq: Frequency of validation.
    flow: Flow between source and target distribution.
    time_sampler: Sampler for the time.
    optimizer: Optimizer for `velocity_field`.
    callback_fn: Callback function.
    num_eval_samples: Number of samples to evaluate on during evaluation.
    rng: Random number generator.
  """

  def __init__(
      self,
      velocity_field: Callable[[
          jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]
      ], jnp.ndarray],
      input_dim: int,
      cond_dim: int,
      iterations: int,
      flow: Type[flows.BaseFlow],
      time_sampler: Callable[[jax.Array, int], jnp.ndarray],
      optimizer: optax.GradientTransformation,
      ot_matcher: Optional[base_solver.OTMatcherLinear],
      unbalancedness_handler: base_solver.UnbalancednessHandler,
      epsilon: float = 1e-2,
      cost_fn: Optional[Type[costs.CostFn]] = None,
      scale_cost: Union[bool, int, float,
                        Literal["mean", "max_norm", "max_bound", "max_cost",
                                "median"]] = "mean",
      callback_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                     Any]] = None,
      logging_freq: int = 100,
      valid_freq: int = 5000,
      num_eval_samples: int = 1000,
      rng: Optional[jax.Array] = None,
  ):
    rng = utils.default_prng_key(rng)
    self.unbalancedness_handler = unbalancedness_handler
    self.iterations = iterations
    self.valid_freq = valid_freq
    self.velocity_field = velocity_field
    self.input_dim = input_dim
    self.ot_matcher = ot_matcher
    self.flow = flow
    self.time_sampler = time_sampler
    self.optimizer = optimizer
    self.epsilon = epsilon
    self.cost_fn = cost_fn
    self.scale_cost = scale_cost
    self.callback_fn = callback_fn
    self.rng = rng
    self.logging_freq = logging_freq
    self.num_eval_samples = num_eval_samples
    self._training_logs: Mapping[str, Any] = collections.defaultdict(list)

    self.setup()

  def setup(self) -> None:
    """Setup :class:`OTFlowMatching`."""
    self.state_velocity_field = (
        self.velocity_field.create_train_state(
            self.rng, self.optimizer, self.input_dim
        )
    )

    self.step_fn = self._get_step_fn()

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        rng: jax.Array,
        state_velocity_field: train_state.TrainState,
        source: jnp.ndarray,
        target: jnp.ndarray,
        source_conditions: Optional[jnp.ndarray],
    ) -> Tuple[Any, Any]:

      def loss_fn(
          params: jnp.ndarray, t: jnp.ndarray, source: jnp.ndarray,
          target: jnp.ndarray, source_conditions: Optional[jnp.ndarray],
          rng: jax.Array
      ) -> jnp.ndarray:

        x_t = self.flow.compute_xt(rng, t, source, target)
        apply_fn = functools.partial(
            state_velocity_field.apply_fn, {"params": params}
        )
        v_t = jax.vmap(apply_fn)(t=t, x=x_t, condition=source_conditions)
        u_t = self.flow.compute_ut(t, source, target)
        return jnp.mean((v_t - u_t) ** 2)

      batch_size = len(source)
      key_t, key_model = jax.random.split(rng, 2)
      t = self.time_sampler(key_t, batch_size)
      grad_fn = jax.value_and_grad(loss_fn)
      loss, grads = grad_fn(
          state_velocity_field.params, t, source, target, source_conditions,
          key_model
      )
      return state_velocity_field.apply_gradients(grads=grads), loss

    return step_fn

  def __call__(self, train_loader, valid_loader):
    """Train :class:`OTFlowMatching`.

    Args;
      train_loader: Dataloader for the training data.
      valid_loader: Dataloader for the validation data.
    """
    batch: Mapping[str, jnp.ndarray] = {}
    curr_loss = 0.0

    for iter in range(self.iterations):
      rng_resample, rng_step_fn, self.rng = jax.random.split(self.rng, 3)
      batch = next(train_loader)
      source, source_conditions, target = batch["source_lin"], batch[
          "source_conditions"], batch["target_lin"]
      if self.ot_matcher is not None:
        tmat = self.ot_matcher.match_fn(source, target)
        (source, source_conditions), (target,) = self.ot_matcher._resample_data(
            rng_resample, tmat, (source, source_conditions), (target,)
        )
      self.state_velocity_field, loss = self.step_fn(
          rng_step_fn, self.state_velocity_field, source, target,
          source_conditions
      )
      curr_loss += loss
      if iter % self.logging_freq == 0:
        self._training_logs["loss"].append(curr_loss / self.logging_freq)
        curr_loss = 0.0
      if self.learn_rescaling:
        (
            self.unbalancedness_handler.state_eta,
            self.unbalancedness_handler.state_xi, eta_predictions,
            xi_predictions, loss_a, loss_b
        ) = self.unbalancedness_handler.step_fn(
            source=source,
            target=target,
            condition=source_conditions,
            a=tmat.sum(axis=1),
            b=tmat.sum(axis=0),
            state_eta=self.unbalancedness_handler.state_eta,
            state_xi=self.unbalancedness_handler.state_xi,
        )
      if iter % self.valid_freq == 0:
        self._valid_step(valid_loader, iter)

  def transport(
      self,
      data: jnp.array,
      condition: Optional[jnp.ndarray] = None,
      forward: bool = True,
      t_0: float = 0.0,
      t_1: float = 1.0,
      **kwargs: Any,
  ) -> diffrax.Solution:
    """Transport data with the learnt map.

    This method pushes-forward the `source` by
    solving the neural ODE parameterized by the
    :attr:`~ott.neural.flows.OTFlowMatching.velocity_field`.

    Args:
      data: Initial condition of the ODE.
      condition: Condition of the input data.
      forward: If `True` integrates forward, otherwise backwards.
      t_0: Starting point of integration.
      t_1: End point of integration.
      kwargs: Keyword arguments for the ODE solver.

    Returns:
      The push-forward or pull-back distribution defined by the learnt
      transport plan.

    """
    t0, t1 = (t_0, t_1) if forward else (t_1, t_0)

    @jax.jit
    def solve_ode(input: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      return diffrax.diffeqsolve(
          diffrax.ODETerm(
              lambda t, x, args: self.state_velocity_field.
              apply_fn({"params": self.state_velocity_field.params},
                       t=t,
                       x=x,
                       condition=cond)
          ),
          kwargs.pop("solver", diffrax.Tsit5()),
          t0=t0,
          t1=t1,
          dt0=kwargs.pop("dt0", None),
          y0=input,
          stepsize_controller=kwargs.pop(
              "stepsize_controller",
              diffrax.PIDController(rtol=1e-5, atol=1e-5)
          ),
          **kwargs,
      ).ys[0]

    return jax.vmap(solve_ode)(data, condition)

  def _valid_step(self, valid_loader, iter):
    next(valid_loader)
    # TODO: add callback and logging

  @property
  def learn_rescaling(self) -> bool:
    """Whether to learn at least one rescaling factor."""
    return (
        self.unbalancedness_handler.rescaling_a is not None or
        self.unbalancedness_handler.rescaling_b is not None
    )

  def save(self, path: str):
    """Save the model.

    Args:
      path: Where to save the model to.
    """
    raise NotImplementedError

  def load(self, path: str) -> "OTFlowMatching":
    """Load a model.

    Args:
      path: Where to load the model from.

    Returns:
      An instance of :class:`ott.neural.solvers.OTFlowMatching`.
    """
    raise NotImplementedError

  @property
  def training_logs(self) -> Dict[str, Any]:
    """Logs of the training."""
    raise NotImplementedError
