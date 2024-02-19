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
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import diffrax
import optax
from flax.training import train_state

from ott import utils
from ott.neural.flow_models import flows, models
from ott.neural.models import base_solver

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
  """(Optimal transport) flow matching :cite:`lipman:22`.

  Includes an extension to OT-FM :cite`tong:23`, :cite:`pooladian:23`.

  Args:
    input_dim: Dimension of the input data.
    velocity_field: Neural vector field parameterized by a neural network.
    flow: Flow between source and target distribution.
    time_sampler: Sampler for the time.
    optimizer: Optimizer for the ``velocity_field``.
    ot_matcher: TODO.
    unbalancedness_handler: TODO.
    rng: Random number generator.
  """

  # TODO(michalk8): in the future, `input_dim`, `optimizer` and `rng` will be
  # in a separate function
  def __init__(
      self,
      input_dim: int,
      velocity_field: models.VelocityField,
      flow: flows.BaseFlow,
      time_sampler: Callable[[jax.Array, int], jnp.ndarray],
      optimizer: optax.GradientTransformation,
      ot_matcher: Optional[base_solver.OTMatcherLinear] = None,
      unbalancedness_handler: Optional[base_solver.UnbalancednessHandler
                                      ] = None,
      rng: Optional[jax.Array] = None,
  ):
    rng = utils.default_prng_key(rng)

    self.input_dim = input_dim
    self.vf = velocity_field
    self.flow = flow
    self.time_sampler = time_sampler
    self.unbalancedness_handler = unbalancedness_handler
    self.ot_matcher = ot_matcher
    self.optimizer = optimizer

    self.vf_state = self.vf.create_train_state(
        rng, self.optimizer, self.input_dim
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

  # TODO(michalk8): refactor in the future PR to just do one step
  def __call__(  # noqa: D102
      self,
      n_iters: int,
      train_source,
      train_target,
      valid_source,
      valid_target,
      valid_freq: int = 5000,
      rng: Optional[jax.Array] = None,
  ) -> Dict[str, Any]:
    rng = utils.default_prng_key(rng)
    training_logs = {"loss": []}

    for it in range(n_iters):
      for batch_source, batch_target in zip(train_source, train_target):
        rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)

        batch_source = jtu.tree_map(jnp.asarray, batch_source)
        batch_target = jtu.tree_map(jnp.asarray, batch_target)

        source = batch_source["lin"]
        source_conditions = batch_source.get("conditions", None)
        target = batch_target["lin"]

        if self.ot_matcher is not None:
          tmat = self.ot_matcher.match_fn(source, target)
          (source, source_conditions), (target,) = self.ot_matcher.sample_joint(
              rng_resample, tmat, (source, source_conditions), (target,)
          )
        else:
          tmat = None

        self.vf_state, loss = self.step_fn(
            rng_step_fn, self.vf_state, source, target, source_conditions
        )
        training_logs["loss"].append(float(loss))

        if self.unbalancedness_handler is not None and tmat is not None:
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

        if it % valid_freq == 0:
          self._valid_step(valid_source, valid_target, it)

    return training_logs

  def transport(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      t0: float = 0.0,
      t1: float = 1.0,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Transport data with the learnt map.

    This method pushes-forward the data by solving the neural ODE
    parameterized by the velocity field.

    Args:
      x: Initial condition of the ODE of shape `(batch_size, ...)`.
      condition: Condition of the input data of shape `(batch_size, ...)`.
      t0: Starting point of integration.
      t1: End point of integration.
      kwargs: Keyword arguments for the ODE solver.

    Returns:
      The push-forward or pull-back distribution defined by the learnt
      transport plan.
    """

    def vf(
        t: jnp.ndarray, x: jnp.ndarray, cond: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
      return self.vf_state.apply_fn({"params": self.vf_state.params},
                                    t=t,
                                    x=x,
                                    condition=cond)

    def solve_ode(x: jnp.ndarray, cond: Optional[jnp.ndarray]) -> jnp.ndarray:
      ode_term = diffrax.ODETerm(vf)
      result = diffrax.diffeqsolve(
          ode_term,
          t0=t0,
          t1=t1,
          y0=x,
          args=cond,
          **kwargs,
      )
      return result.ys[0]

    kwargs.setdefault("dt0", None)
    kwargs.setdefault("solver", diffrax.Tsit5())
    kwargs.setdefault(
        "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
    )

    in_axes = [0, None if condition is None else 0]
    return jax.jit(jax.vmap(solve_ode, in_axes))(x, condition)

  def _valid_step(self, it: int, valid_source, valid_target) -> None:
    pass
