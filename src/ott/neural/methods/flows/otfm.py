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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import diffrax
from flax.training import train_state

from ott import utils
from ott.neural.methods.flows import dynamics
from ott.neural.networks import velocity_field
from ott.solvers import utils as solver_utils

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
  """(Optimal transport) flow matching :cite:`lipman:22`.

  With an extension to OT-FM :cite:`tong:23,pooladian:23`.

  Args:
    vf: Vector field parameterized by a neural network.
    flow: Flow between the source and the target distributions.
    match_fn: Function to match samples from the source and the target
      distributions. It has a ``(src, tgt) -> matching`` signature.
    time_sampler: Time sampler with a ``(rng, n_samples) -> time`` signature.
    kwargs: Keyword arguments for
      :meth:`~ott.neural.networks.velocity_field.VelocityField.create_train_state`.
  """

  def __init__(
      self,
      vf: velocity_field.VelocityField,
      flow: dynamics.BaseFlow,
      match_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                  jnp.ndarray]] = None,
      time_sampler: Callable[[jax.Array, int],
                             jnp.ndarray] = solver_utils.uniform_sampler,
      **kwargs: Any,
  ):
    self.vf = vf
    self.flow = flow
    self.time_sampler = time_sampler
    self.match_fn = match_fn

    self.vf_state = self.vf.create_train_state(
        input_dim=self.vf.output_dims[-1], **kwargs
    )
    self.step_fn = self._get_step_fn()

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        rng: jax.Array,
        vf_state: train_state.TrainState,
        source: jnp.ndarray,
        target: jnp.ndarray,
        source_conditions: Optional[jnp.ndarray],
    ) -> Tuple[Any, Any]:

      def loss_fn(
          params: jnp.ndarray, t: jnp.ndarray, source: jnp.ndarray,
          target: jnp.ndarray, source_conditions: Optional[jnp.ndarray],
          rng: jax.Array
      ) -> jnp.ndarray:
        rng_flow, rng_dropout = jax.random.split(rng, 2)
        x_t = self.flow.compute_xt(rng_flow, t, source, target)
        v_t = vf_state.apply_fn({"params": params},
                                t,
                                x_t,
                                source_conditions,
                                rngs={"dropout": rng_dropout})
        u_t = self.flow.compute_ut(t, source, target)

        return jnp.mean((v_t - u_t) ** 2)

      batch_size = len(source)
      key_t, key_model = jax.random.split(rng, 2)
      t = self.time_sampler(key_t, batch_size)
      grad_fn = jax.value_and_grad(loss_fn)
      loss, grads = grad_fn(
          vf_state.params, t, source, target, source_conditions, key_model
      )
      return vf_state.apply_gradients(grads=grads), loss

    return step_fn

  def __call__(  # noqa: D102
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      *,
      n_iters: int,
      rng: Optional[jax.Array] = None,
  ) -> Dict[str, List[float]]:
    """Train the OTFlowMatching model.

    Args:
      loader: Data loader returning a dictionary with possible keys
      `src_lin`, `tgt_lin`, `src_condition`.
      n_iters: Number of iterations to train the model.
      rng: Random number generator.

    Returns:
      Training logs.
    """
    rng = utils.default_prng_key(rng)
    training_logs = {"loss": []}
    for batch in loader:
      rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)

      batch = jtu.tree_map(jnp.asarray, batch)

      src, tgt = batch["src_lin"], batch["tgt_lin"]
      src_cond = batch.get("src_condition")

      if self.match_fn is not None:
        tmat = self.match_fn(src, tgt)
        src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
        src, tgt = src[src_ixs], tgt[tgt_ixs]
        src_cond = None if src_cond is None else src_cond[src_ixs]

      self.vf_state, loss = self.step_fn(
          rng_step_fn,
          self.vf_state,
          src,
          tgt,
          src_cond,
      )

      training_logs["loss"].append(float(loss))
      if len(training_logs["loss"]) >= n_iters:
        break

    return training_logs

  def transport(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      t0: float = 0.0,
      t1: float = 1.0,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Transport data with the learned map.

    This method pushes-forward the data by solving the neural ODE
    parameterized by the velocity field.

    Args:
      x: Initial condition of the ODE of shape ``[batch_size, ...]``.
      condition: Condition of the input data of shape ``[batch_size, ...]``.
      t0: Starting point of integration.
      t1: End point of integration.
      kwargs: Keyword arguments for the ODE solver.

    Returns:
      The push-forward or pull-back distribution defined by the learned
      transport plan.
    """

    def vf(
        t: jnp.ndarray, x: jnp.ndarray, cond: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
      params = self.vf_state.params
      return self.vf_state.apply_fn({"params": params}, t, x, cond, train=False)

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
