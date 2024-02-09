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
import functools
import types
from typing import Any, Callable, Dict, Literal, Optional, Type, Union

import jax
import jax.numpy as jnp

import diffrax
import optax
from flax.training import train_state

from ott import utils
from ott.geometry import costs
from ott.neural.flows import flows, samplers
from ott.neural.models import base_solver
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

__all__ = ["GENOT"]


class GENOT:
  """The GENOT training class as introduced in :cite:`klein_uscidda:23`.

  Args:
    velocity_field: Neural vector field parameterized by a neural network.
    input_dim: Dimension of the data in the source distribution.
    output_dim: Dimension of the data in the target distribution.
    cond_dim: Dimension of the conditioning variable.
    iterations: Number of iterations.
    valid_freq: Frequency of validation.
    ot_solver: OT solver to match samples from the source and the target
      distribution.
    epsilon: Entropy regularization term of the OT problem solved by
      `ot_solver`.
    cost_fn: Cost function for the OT problem solved by the `ot_solver`.
      In the linear case, this is always expected to be of type `str`.
      If the problem is of quadratic type and `cost_fn` is a string,
      the `cost_fn` is used for all terms, i.e. both quadratic terms and,
      if applicable, the linear temr. If of type :class:`dict`, the keys
      are expected to be `cost_fn_xx`, `cost_fn_yy`, and if applicable,
      `cost_fn_xy`.
    scale_cost: How to scale the cost matrix for the OT problem solved by
      the `ot_solver`. In the linear case, this is always expected to be
      not a :class:`dict`. If the problem is of quadratic type and
      `scale_cost` is a string, the `scale_cost` argument is used for all
      terms, i.e. both quadratic terms and, if applicable, the linear temr.
      If of type :class:`dict`, the keys are expected to be `scale_cost_xx`,
      `scale_cost_yy`, and if applicable, `scale_cost_xy`.
    optimizer: Optimizer for `velocity_field`.
    flow: Flow between latent distribution and target distribution.
    time_sampler: Sampler for the time.
    unbalancedness_handler: Handler for unbalancedness.
    k_samples_per_x: Number of samples drawn from the conditional distribution
      of an input sample, see algorithm TODO.
    solver_latent_to_data: Linear OT solver to match the latent distribution
      with the conditional distribution.
    kwargs_solver_latent_to_data: Keyword arguments for `solver_latent_to_data`.
      #TODO: adapt
    fused_penalty: Fused penalty of the linear/fused term in the Fused
      Gromov-Wasserstein problem.
    callback_fn: Callback function.
    rng: Random number generator.
  """

  def __init__(
      self,
      velocity_field: Callable[[
          jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]
      ], jnp.ndarray],
      input_dim: int,
      output_dim: int,
      cond_dim: int,
      iterations: int,
      valid_freq: int,
      ot_solver: was_solver.WassersteinSolver,
      epsilon: float,
      cost_fn: Union[costs.CostFn, Dict[str, costs.CostFn]],
      scale_cost: Union[Union[bool, int, float,
                              Literal["mean", "max_norm", "max_bound",
                                      "max_cost", "median"]],
                        Dict[str, Union[bool, int, float,
                                        Literal["mean", "max_norm", "max_bound",
                                                "max_cost", "median"]]]],
      unbalancedness_handler: base_solver.UnbalancednessHandler,
      optimizer: optax.GradientTransformation,
      flow: Type[flows.BaseFlow] = flows.ConstantNoiseFlow(0.0),  # noqa: B008
      time_sampler: Callable[[jax.Array, int],
                             jnp.ndarray] = samplers.uniform_sampler,
      k_samples_per_x: int = 1,
      solver_latent_to_data: Optional[Type[was_solver.WassersteinSolver]
                                     ] = None,
      kwargs_solver_latent_to_data: Dict[str, Any] = types.MappingProxyType({}),
      fused_penalty: float = 0.0,
      callback_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                     Any]] = None,
      rng: Optional[jax.Array] = None,
  ):
    rng = utils.default_prng_key(rng)

    if isinstance(
        ot_solver, gromov_wasserstein.GromovWasserstein
    ) and epsilon is not None:
      raise ValueError(
          "If `ot_solver` is `GromovWasserstein`, `epsilon` must be `None`. " +
          "This check is performed to ensure that in the (fused) Gromov case " +
          "the `epsilon` parameter is passed via the `ot_solver`."
      )

    self.rng = utils.default_prng_key(rng)
    self.iterations = iterations
    self.valid_freq = valid_freq
    self.velocity_field = velocity_field
    self.state_velocity_field: Optional[train_state.TrainState] = None
    self.flow = flow
    self.time_sampler = time_sampler
    self.optimizer = optimizer
    self.latent_noise_fn = jax.tree_util.Partial(
        jax.random.multivariate_normal,
        mean=jnp.zeros((output_dim,)),
        cov=jnp.diag(jnp.ones((output_dim,)))
    )
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.cond_dim = cond_dim
    self.k_samples_per_x = k_samples_per_x

    # unbalancedness
    self.unbalancedness_handler = unbalancedness_handler

    # OT data-data matching parameters
    self.ot_solver = ot_solver
    self.epsilon = epsilon
    self.cost_fn = cost_fn
    self.scale_cost = scale_cost
    self.fused_penalty = fused_penalty

    # OT latent-data matching parameters
    self.solver_latent_to_data = solver_latent_to_data
    self.kwargs_solver_latent_to_data = kwargs_solver_latent_to_data

    # callback parameteres
    self.callback_fn = callback_fn
    self.setup()

  def setup(self) -> None:
    """Set up the model."""
    self.state_velocity_field = (
        self.velocity_field.create_train_state(
            self.rng, self.optimizer, self.output_dim
        )
    )
    self.step_fn = self._get_step_fn()
    if self.solver_latent_to_data is not None:
      self.match_latent_to_data_fn = self._get_sinkhorn_match_fn(
          ot_solver=self.solver_latent_to_data,
          **self.kwargs_solver_latent_to_data
      )
    else:
      self.match_latent_to_data_fn = lambda key, x, y, **_: (x, y)

    # TODO: add graph construction function
    if isinstance(self.ot_solver, sinkhorn.Sinkhorn):
      self.match_fn = self._get_sinkhorn_match_fn(
          ot_solver=self.ot_solver,
          epsilon=self.epsilon,
          cost_fn=self.cost_fn,
          scale_cost=self.scale_cost,
          tau_a=self.unbalancedness_handler.tau_a,
          tau_b=self.unbalancedness_handler.tau_b,
          filter_input=True
      )
    else:
      self.match_fn = self._get_gromov_match_fn(
          ot_solver=self.ot_solver,
          cost_fn=self.cost_fn,
          scale_cost=self.scale_cost,
          tau_a=self.unbalancedness_handler.tau_a,
          tau_b=self.unbalancedness_handler.tau_b,
          fused_penalty=self.fused_penalty
      )

  def __call__(self, train_loader, valid_loader):
    """Train GENOT.

    Args:
      train_loader: Data loader for the training data.
      valid_loader: Data loader for the validation data.
    """
    batch: Dict[str, jnp.array] = {}
    for iteration in range(self.iterations):
      batch = next(train_loader)

      (
          self.rng, rng_time, rng_resample, rng_noise, rng_latent_data_match,
          rng_step_fn
      ) = jax.random.split(self.rng, 6)
      batch_size = len(
          batch["source_lin"]
      ) if batch["source_lin"] is not None else len(batch["source_quad"])
      n_samples = batch_size * self.k_samples_per_x
      batch["time"] = self.time_sampler(rng_time, n_samples)
      batch["latent"] = self.latent_noise_fn(
          rng_noise, shape=(self.k_samples_per_x, batch_size)
      )

      tmat = self.match_fn(
          batch["source_lin"], batch["source_quad"], batch["target_lin"],
          batch["target_quad"]
      )

      batch["source"] = jnp.concatenate([
          batch[el]
          for el in ["source_lin", "source_quad"]
          if batch[el] is not None
      ],
                                        axis=1)
      batch["target"] = jnp.concatenate([
          batch[el]
          for el in ["target_lin", "target_quad"]
          if batch[el] is not None
      ],
                                        axis=1)

      batch = {
          k: v for k, v in batch.items() if k in
          ["source", "target", "source_conditions", "time", "noise", "latent"]
      }

      (batch["source"], batch["source_conditions"]
      ), (batch["target"],) = self._sample_conditional_indices_from_tmap(
          rng_resample,
          tmat,
          self.k_samples_per_x, (batch["source"], batch["source_conditions"]),
          (batch["target"],),
          source_is_balanced=(self.unbalancedness_handler.tau_a == 1.0)
      )

      if self.solver_latent_to_data is not None:
        tmats_latent_data = jnp.array(
            jax.vmap(self.match_latent_to_data_fn, 0,
                     0)(x=batch["latent"], y=batch["target"])
        )

        rng_latent_data_match = jax.random.split(
            rng_latent_data_match, self.k_samples_per_x
        )
        (batch["source"], batch["source_conditions"]
        ), (batch["target"],) = jax.vmap(self._resample_data, 0, 0)(
            rng_latent_data_match, tmats_latent_data,
            (batch["source"], batch["source_conditions"]), (batch["target"],)
        )
      batch = {
          key:
              jnp.reshape(arr, (batch_size * self.k_samples_per_x,
                                -1)) if arr is not None else None
          for key, arr in batch.items()
      }

      self.state_velocity_field, loss = self.step_fn(
          rng_step_fn, self.state_velocity_field, batch
      )
      if self.learn_rescaling:
        (
            self.state_eta, self.state_xi, eta_predictions, xi_predictions,
            loss_a, loss_b
        ) = self.unbalancedness_handler.step_fn(
            source=batch["source"],
            target=batch["target"],
            condition=batch["source_conditions"],
            a=tmat.sum(axis=1),
            b=tmat.sum(axis=0),
            state_eta=self.unbalancedness_handler.state_eta,
            state_xi=self.unbalancedness_handler.state_xi,
        )
      if iteration % self.valid_freq == 0:
        self._valid_step(valid_loader, iteration)

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        rng: jax.Array,
        state_velocity_field: train_state.TrainState,
        batch: Dict[str, jnp.array],
    ):

      def loss_fn(
          params: jnp.ndarray, batch: Dict[str, jnp.array],
          rng: jax.random.PRNGKeyArray
      ):
        x_t = self.flow.compute_xt(
            rng, batch["time"], batch["latent"], batch["target"]
        )
        apply_fn = functools.partial(
            state_velocity_field.apply_fn, {"params": params}
        )

        cond_input = jnp.concatenate([
            batch[el]
            for el in ["source", "source_conditions"]
            if batch[el] is not None
        ],
                                     axis=1)
        v_t = jax.vmap(apply_fn)(t=batch["time"], x=x_t, condition=cond_input)
        u_t = self.flow.compute_ut(
            batch["time"], batch["latent"], batch["target"]
        )
        return jnp.mean((v_t - u_t) ** 2)

      grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
      loss, grads = grad_fn(state_velocity_field.params, batch, rng)

      return state_velocity_field.apply_gradients(grads=grads), loss

    return step_fn

  def transport(
      self,
      source: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      rng: Optional[jax.Array] = None,
      forward: bool = True,
      **kwargs: Any,
  ) -> Union[jnp.array, diffrax.Solution, Optional[jnp.ndarray]]:
    """Transport data with the learnt plan.

    This method pushes-forward the `source` to its conditional distribution by
      solving the neural ODE parameterized by the
      :attr:`~ott.neural.flows.genot.velocity_field`

    Args:
      source: Data to transport.
      condition: Condition of the input data.
      rng: random seed for sampling from the latent distribution.
      forward: If `True` integrates forward, otherwise backwards.
      kwargs: Keyword arguments for the ODE solver.

    Returns:
      The push-forward or pull-back distribution defined by the learnt
      transport plan.

    """
    rng = utils.default_prng_key(rng)
    if not forward:
      raise NotImplementedError
    assert len(source) == len(condition) if condition is not None else True

    latent_batch = self.latent_noise_fn(rng, shape=(len(source),))
    cond_input = source if condition is None else jnp.concatenate([
        source, condition
    ],
                                                                  axis=-1)
    t0, t1 = (0.0, 1.0)

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

    return jax.vmap(solve_ode)(latent_batch, cond_input)

  def _valid_step(self, valid_loader, iter):
    """TODO."""
    next(valid_loader)

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

  def load(self, path: str) -> "GENOT":
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
