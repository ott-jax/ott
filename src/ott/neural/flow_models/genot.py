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
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import diffrax
from flax.training import train_state

from ott import utils
from ott.neural.flow_models import flows, models

__all__ = ["GENOTBase", "GENOTLin", "GENOTQuad"]


# TODO(michalk8): remove the base class?
class GENOTBase:
  """TODO :cite:`klein_uscidda:23`.

  Args:
    velocity_field: Neural vector field parameterized by a neural network.
    flow: Flow between latent distribution and target distribution.
    time_sampler: Sampler for the time.
      of an input sample, see algorithm TODO.
    data_match_fn: Linear OT solver to match the latent distribution
      with the conditional distribution.
    latent_match_fn: TODO.
    latent_noise_fn: TODO.
    k_samples_per_x: Number of samples drawn from the conditional distribution
    kwargs: TODO.
  """

  def __init__(
      self,
      velocity_field: models.VelocityField,
      flow: flows.BaseFlow,
      time_sampler: Callable[[jax.Array, int], jnp.ndarray],
      data_match_fn: Any,
      latent_match_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                         jnp.ndarray]] = None,
      # TODO(michalk8): add a default for this?
      latent_noise_fn: Optional[Callable[[jax.Array, Tuple[int, ...]],
                                         jnp.ndarray]] = None,
      k_samples_per_x: int = 1,
      **kwargs: Any,
  ):
    self.vf = velocity_field
    self.flow = flow
    self.time_sampler = time_sampler
    self.ot_matcher = data_match_fn
    if latent_match_fn is not None:
      latent_match_fn = jax.jit(jax.vmap(latent_match_fn, 0, 0))
    self.latent_match_fn = latent_match_fn
    self.latent_noise_fn = latent_noise_fn
    self.k_samples_per_x = k_samples_per_x

    self.vf_state = self.vf.create_train_state(**kwargs)
    self.step_fn = self._get_step_fn()

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        rng: jax.Array,
        vf_state: train_state.TrainState,
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
        apply_fn = functools.partial(vf_state.apply_fn, {"params": params})

        cond_input = jnp.concatenate([
            source, source_conditions
        ], axis=1) if source_conditions is not None else source
        v_t = jax.vmap(apply_fn)(t=time, x=x_t, condition=cond_input)
        u_t = self.flow.compute_ut(time, latent, target)
        return jnp.mean((v_t - u_t) ** 2)

      grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
      loss, grads = grad_fn(
          vf_state.params, time, source, target, latent, source_conditions, rng
      )

      return vf_state.apply_gradients(grads=grads), loss

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
    latent_batch = self.latent_noise_fn(rng, (len(source),))
    cond_input = source if condition is None else (
        jnp.concatenate([source, condition], axis=-1)
    )

    @jax.jit
    def solve_ode(input: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      ode_term = diffrax.ODETerm(
          lambda t, x, args: self.vf_state.
          apply_fn({"params": self.vf_state.params}, t=t, x=x, condition=cond)
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

  def _reshape_samples(self, arrays: Tuple[jnp.ndarray, ...],
                       batch_size: int) -> Tuple[jnp.ndarray, ...]:
    return jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (batch_size * self.k_samples_per_x, -1)),
        arrays
    )


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

    for _ in range(n_iters):
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
            rng_noise, (self.k_samples_per_x, batch_size)
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

        if self.latent_match_fn is not None:
          # already vmapped
          tmats_latent_data = self.latent_match_fn(latent, target)

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
        self.vf_state, loss = self.step_fn(
            rng_step_fn, self.vf_state, time, source, target, latent,
            source_conditions
        )

        training_logs["loss"].append(float(loss))


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

    for _ in range(n_iters):
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
            rng_noise, (self.k_samples_per_x, batch_size)
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

        if self.latent_match_fn is not None:
          # already vmapped
          tmats_latent_data = self.latent_match_fn(latent, target)

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

        self.vf_state, loss = self.step_fn(
            rng_step_fn, self.vf_state, time, source, target, latent,
            source_conditions
        )
        training_logs["loss"].append(float(loss))
