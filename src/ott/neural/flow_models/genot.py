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
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import diffrax
import optax
from flax.training import train_state

from ott import utils
from ott.neural.flow_models import flows, samplers

__all__ = ["GENOTBase", "GENOTLin", "GENOTQuad"]


class GENOTBase:
  """Base class for GENOT models (:cite:`klein_uscidda:23`).

  GENOT (Generative Entropic Neural Optimal Transport) is a neural solver
  for entropic OT prooblems, in the linear
  (:class:`ott.neural.flows.genot.GENOTLin`), the Gromov-Wasserstein, and
  the Fused Gromov-Wasserstein ((:class:`ott.neural.flows.genot.GENOTQUad`))
  setting.

  Args:
    velocity_field: Neural vector field parameterized by a neural network.
    input_dim: Dimension of the data in the source distribution.
    output_dim: Dimension of the data in the target distribution.
    cond_dim: Dimension of the conditioning variable.
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
      *,
      input_dim: int,
      output_dim: int,
      cond_dim: int,
      valid_freq: int,
      # TODO(michalk8)
      ot_matcher: Any,
      optimizer: optax.GradientTransformation,
      flow: Type[flows.BaseFlow] = flows.ConstantNoiseFlow(0.0),  # noqa: B008
      time_sampler: Callable[[jax.Array, int],
                             jnp.ndarray] = samplers.uniform_sampler,
      k_samples_per_x: int = 1,
      # TODO(michalk8)
      matcher_latent_to_data: Optional[Callable] = None,
      kwargs_solver_latent_to_data: Dict[str, Any] = types.MappingProxyType({}),
      fused_penalty: float = 0.0,
      callback_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                     Any]] = None,
      rng: Optional[jax.Array] = None,
  ):
    rng = utils.default_prng_key(rng)

    self.rng = utils.default_prng_key(rng)
    self.valid_freq = valid_freq
    self.velocity_field = velocity_field
    self.state_velocity_field: Optional[train_state.TrainState] = None
    self.flow = flow
    self.time_sampler = time_sampler
    self.optimizer = optimizer
    self.ot_matcher = ot_matcher
    self.latent_noise_fn = jax.tree_util.Partial(
        jax.random.multivariate_normal,
        mean=jnp.zeros((output_dim,)),
        cov=jnp.diag(jnp.ones((output_dim,)))
    )
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.cond_dim = cond_dim
    self.k_samples_per_x = k_samples_per_x

    # OT data-data matching parameters

    self.fused_penalty = fused_penalty

    # OT latent-data matching parameters
    self.matcher_latent_to_data = matcher_latent_to_data
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

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        rng: jax.Array,
        state_velocity_field: train_state.TrainState,
        time: jnp.ndarray,
        source: jnp.ndarray,
        target: jnp.ndarray,
        latent: jnp.ndarray,
        source_conditions: Optional[jnp.ndarray],
    ):

      def loss_fn(
          params: jnp.ndarray, time: jnp.ndarray, source: jnp.ndarray,
          target: jnp.ndarray, latent: jnp.ndarray,
          source_conditions: Optional[jnp.ndarray], rng: jax.Array
      ):
        x_t = self.flow.compute_xt(rng, time, latent, target)
        apply_fn = functools.partial(
            state_velocity_field.apply_fn, {"params": params}
        )

        cond_input = jnp.concatenate([
            source, source_conditions
        ], axis=1) if source_conditions is not None else source
        v_t = jax.vmap(apply_fn)(t=time, x=x_t, condition=cond_input)
        u_t = self.flow.compute_ut(time, latent, target)
        return jnp.mean((v_t - u_t) ** 2)

      grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
      loss, grads = grad_fn(
          state_velocity_field.params, time, source, target, latent,
          source_conditions, rng
      )

      return state_velocity_field.apply_gradients(grads=grads), loss

    return step_fn

  def transport(
      self,
      source: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      rng: Optional[jax.Array] = None,
      forward: bool = True,
      t_0: float = 0.0,
      t_1: float = 1.0,
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
      t_0: Starting time of integration of neural ODE.
      t_1: End time of integration of neural ODE.
      kwargs: Keyword arguments for the ODE solver.

    Returns:
      The push-forward or pull-back distribution defined by the learnt
      transport plan.

    """
    rng = utils.default_prng_key(rng)
    if not forward:
      raise NotImplementedError
    if condition is not None:
      assert len(source) == len(condition), (len(source), len(condition))
    latent_batch = self.latent_noise_fn(rng, shape=(len(source),))
    cond_input = source if condition is None else (
        jnp.concatenate([source, condition], axis=-1)
    )

    @jax.jit
    def solve_ode(input: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      ode_term = diffrax.ODETerm(
          lambda t, x, args: self.state_velocity_field.
          apply_fn({"params": self.state_velocity_field.params},
                   t=t,
                   x=x,
                   condition=cond)
      ),
      solver = kwargs.pop("solver", diffrax.Tsit5())
      stepsize_controller = kwargs.pop(
          "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
      )
      sol = diffrax.diffeqsolve(
          ode_term,
          solver,
          t0=t_0,
          t1=t_1,
          dt0=kwargs.pop("dt0", None),
          y0=input,
          stepsize_controller=stepsize_controller,
          **kwargs,
      )
      return sol.ys[0]

    return jax.vmap(solve_ode)(latent_batch, cond_input)

  def _valid_step(self, valid_loader, iter):
    pass

  @property
  def learn_rescaling(self) -> bool:
    """Whether to learn at least one rescaling factor."""
    return False

  def _reshape_samples(self, arrays: Tuple[jnp.ndarray, ...],
                       batch_size: int) -> Tuple[jnp.ndarray, ...]:
    return jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (batch_size * self.k_samples_per_x, -1)),
        arrays
    )

  def _learn_rescaling(
      self, source: jnp.ndarray, target: jnp.ndarray,
      source_conditions: Optional[jnp.ndarray], tmat: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    raise NotImplementedError


class GENOTLin(GENOTBase):
  """Implementation of GENOT-L (:cite:`klein:23`).

  GENOT-L (Generative Entropic Neural Optimal Transport, linear) is a
  neural solver for entropic (linear) OT problems.
  """

  def __call__(
      self,
      n_iters: int,
      train_source,
      train_target,
      valid_source,
      valid_target,
      valid_freq: int = 5000,
      rng: Optional[jax.Array] = None,
  ):
    """Train GENOTLin."""
    rng = utils.default_prng_key(rng)
    training_logs = {"loss": []}

    for it in range(n_iters):
      for batch_source, batch_target in zip(train_source, train_target):
        (
            rng, rng_resample, rng_noise, rng_time, rng_latent_data_match,
            rng_step_fn
        ) = jax.random.split(rng, 6)

        batch_source = jtu.tree_map(jnp.asarray, batch_source)
        batch_target = jtu.tree_map(jnp.asarray, batch_target)

        source = batch_source["lin"]
        source_conditions = batch_source.get("conditions", None)
        target = batch_target["lin"]

        batch_size = len(source)
        n_samples = batch_size * self.k_samples_per_x
        time = self.time_sampler(rng_time, n_samples)
        latent = self.latent_noise_fn(
            rng_noise, shape=(self.k_samples_per_x, batch_size)
        )

        tmat = self.ot_matcher.match_fn(
            source,
            target,
        )

        (source, source_conditions
        ), (target,) = self.ot_matcher.sample_conditional_indices_from_tmap(
            rng=rng_resample,
            conditional_distributions=tmat,
            k_samples_per_x=self.k_samples_per_x,
            source_arrays=(source, source_conditions),
            target_arrays=(target,),
            source_is_balanced=(self.ot_matcher.tau_a == 1.0)
        )

        if self.matcher_latent_to_data is not None:
          tmats_latent_data = jnp.array(
              jax.vmap(self.matcher_latent_to_data.match_fn, 0,
                       0)(x=latent, y=target)
          )

          rng_latent_data_match = jax.random.split(
              rng_latent_data_match, self.k_samples_per_x
          )
          (source, source_conditions
          ), (target,) = jax.vmap(self.ot_matcher.sample_joint, 0, 0)(
              rng_latent_data_match, tmats_latent_data,
              (source, source_conditions), (target,)
          )

        source, source_conditions, target, latent = self._reshape_samples(
            (source, source_conditions, target, latent), batch_size
        )
        self.state_velocity_field, loss = self.step_fn(
            rng_step_fn, self.state_velocity_field, time, source, target,
            latent, source_conditions
        )
        if self.learn_rescaling:
          eta_preds, xi_preds, loss_a, loss_b = self._learn_rescaling(
              source=source,
              target=target,
              condition=source_conditions,
              tmat=tmat
          )

        training_logs["loss"].append(float(loss))

        if it % valid_freq == 0:
          self._valid_step(valid_source, valid_target, it)


class GENOTQuad(GENOTBase):
  """Implementation of GENOT-Q and GENOT-F (:cite:`klein:23`).

  GENOT-Q (Generative Entropic Neural Optimal Transport, quadratic) and
  GENOT-F (Generative Entropic Neural Optimal Transport, fused) are neural
  solver for entropic Gromov-Wasserstein and entropic Fused Gromov-Wasserstein
  problems, respectively.
  """

  def __call__(
      self,
      n_iters: int,
      train_source,
      train_target,
      valid_source,
      valid_target,
      valid_freq: int = 5000,
      rng: Optional[jax.Array] = None,
  ):
    """Train GENOTQuad."""
    rng = utils.default_prng_key(rng)
    training_logs = {"loss": []}

    for it in range(n_iters):
      for batch_source, batch_target in zip(train_source, train_target):
        (
            rng, rng_resample, rng_noise, rng_time, rng_latent_data_match,
            rng_step_fn
        ) = jax.random.split(rng, 6)

        batch_source = jtu.tree_map(jnp.asarray, batch_source)
        batch_target = jtu.tree_map(jnp.asarray, batch_target)

        source_lin = batch_source.get("lin", None)
        source_quad = batch_source["quad"]
        source_conditions = batch_source.get("conditions", None)
        target_lin = batch_target.get("lin", None)
        target_quad = batch_target["quad"]

        batch_size = len(source_quad)
        n_samples = batch_size * self.k_samples_per_x
        time = self.time_sampler(rng_time, n_samples)
        latent = self.latent_noise_fn(
            rng_noise, shape=(self.k_samples_per_x, batch_size)
        )

        tmat = self.ot_matcher.match_fn(
            source_quad, target_quad, source_lin, target_lin
        )

        if self.ot_matcher.fused_penalty > 0.0:
          source = jnp.concatenate((source_lin, source_quad), axis=1)
          target = jnp.concatenate((target_lin, target_quad), axis=1)
        else:
          source = source_quad
          target = target_quad

        (source, source_conditions), (target,) = (
            self.ot_matcher.sample_conditional_indices_from_tmap(
                rng=rng_resample,
                conditional_distributions=tmat,
                k_samples_per_x=self.k_samples_per_x,
                source_arrays=(source, source_conditions),
                target_arrays=(target,),
                source_is_balanced=(self.ot_matcher.tau_a == 1.0)
            )
        )

        if self.matcher_latent_to_data is not None:
          tmats_latent_data = jnp.array(
              jax.vmap(self.matcher_latent_to_data.match_fn, 0,
                       0)(x=latent, y=target)
          )

          rng_latent_data_match = jax.random.split(
              rng_latent_data_match, self.k_samples_per_x
          )

          (source, source_conditions
          ), (target,) = jax.vmap(self.ot_matcher.sample_joint, 0, 0)(
              rng_latent_data_match, tmats_latent_data,
              (source, source_conditions), (target,)
          )

        source, source_conditions, target, latent = self._reshape_samples(
            (source, source_conditions, target, latent), batch_size
        )

        self.state_velocity_field, loss = self.step_fn(
            rng_step_fn, self.state_velocity_field, time, source, target,
            latent, source_conditions
        )
        if self.learn_rescaling:
          eta_preds, xi_preds, loss_a, loss_b = self._learn_rescaling(
              source=source,
              target=target,
              condition=source_conditions,
              tmat=tmat
          )
        training_logs["loss"].append(float(loss))

        if it % valid_freq == 0:
          self._valid_step(valid_source, valid_target, it)
