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
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import diffrax
from flax.training import train_state

from ott import utils
from ott.neural.flow_models import flows, models
from ott.neural.flow_models import utils as flow_utils

__all__ = ["GENOT"]


# TODO(michalk8): remove the base class?
class GENOT:
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
      # TODO(mcihalk8): all args are optional
      data_match_fn: Callable[
          [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
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
    self.data_match_fn = data_match_fn
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

  def __call__(
      self,
      loader: Any,
      n_iters: int,
      rng: Optional[jax.Array] = None
  ) -> Dict[str, List[float]]:
    """TODO."""

    def prepare_data(
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray], Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
      src_lin, src_quad = batch.get("src_lin"), batch.get("src_quad")
      tgt_lin, tgt_quad = batch.get("tgt_lin"), batch.get("tgt_quad")
      arrs = src_lin, tgt_lin, src_quad, tgt_quad

      if src_quad is None and tgt_quad is None:  # lin
        src, tgt = src_lin, tgt_lin
      elif src_lin is None and tgt_lin is None:  # quad
        src, tgt = src_quad, tgt_quad
      elif all(arr is not None for arr in arrs):  # fused quad
        src = jnp.concatenate([src_lin, src_quad], axis=1)
        tgt = jnp.concatenate([tgt_lin, tgt_quad], axis=1)
      else:
        raise RuntimeError("TODO")

      # TODO(michalk8): filter `None` from the `arrs`?
      return (src, batch.get("src_condition"), tgt), arrs

    rng = utils.default_prng_key(rng)
    training_logs = {"loss": []}
    for batch in loader:
      rng = jax.random.split(rng, 6)
      rng, rng_resample, rng_noise, rng_time, rng_latent, rng_step_fn = rng

      batch = jtu.tree_map(jnp.asarray, batch)
      (src, src_cond, tgt), data = prepare_data(batch)

      time = self.time_sampler(rng_time, len(src) * self.k_samples_per_x)
      latent = self.latent_noise_fn(rng_noise, (self.k_samples_per_x, len(src)))

      tmat = self.data_match_fn(*data)  # (n, m)
      src_ixs, tgt_ixs = flow_utils.sample_conditional(  # (n, k), (m, k)
          rng_resample,
          tmat,
          k=self.k_samples_per_x,
          uniform_marginals=True,  # TODO(michalk8): expose
      )

      src = src[src_ixs].swapaxes(0, 1)  # (k, n, ...)
      tgt = tgt[tgt_ixs].swapaxes(0, 1)  # (k, m, ...)
      if src_cond is not None:
        src_cond = src_cond[src_ixs].swapaxes(0, 1)  # (k, n, ...)

      if self.latent_match_fn is not None:
        src, src_cond, tgt = self._match_latent(rng, src, src_cond, latent, tgt)

      src = src.reshape(-1, *src.shape[2:])  # (k * bs, ...)
      tgt = tgt.reshape(-1, *tgt.shape[2:])
      latent = latent.reshape(-1, *latent.shape[2:])
      if src_cond is not None:
        src_cond = src_cond.reshape(-1, *src_cond.shape[2:])

      self.vf_state, loss = self.step_fn(
          rng_step_fn, self.vf_state, time, src, tgt, latent, src_cond
      )
      training_logs["loss"].append(float(loss))

    return training_logs

  def _match_latent(
      self, rng: jax.Array, src: jnp.ndarray, src_cond: Optional[jnp.ndarray],
      latent: jnp.ndarray, tgt: jnp.ndarray
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:

    def resample(
        rng: jax.Array, src: jnp.ndarray, src_cond: Optional[jnp.ndarray],
        tgt: jnp.ndarray, latent: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
      tmat = self.latent_match_fn(latent, tgt)  # (n, k)

      src_ixs, tgt_ixs = flow_utils.sample_joint(rng, tmat)  # (n,), (m,)
      src, tgt = src[src_ixs], tgt[tgt_ixs]
      if src_cond is not None:
        src_cond = src_cond[src_ixs]

      return src, src_cond, tgt

    cond_axis = None if src_cond is None else 0
    in_axes, out_axes = (0, 0, cond_axis, 0, 0), (0, None, 0)
    resample_fn = jax.jit(jax.vmap(resample, in_axes, out_axes))

    rngs = jax.random.split(rng, self.k_samples_per_x)
    return resample_fn(rngs, src, src_cond, tgt, latent)

  def transport(
      self,
      source: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      t0: float = 0.0,
      t1: float = 1.0,
      rng: Optional[jax.Array] = None,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Transport data with the learnt plan.

    This method pushes-forward the `source` to its conditional distribution by
      solving the neural ODE parameterized by the
      :attr:`~ott.neural.flows.genot.velocity_field`

    Args:
      source: Data to transport.
      condition: Condition of the input data.
      t0: Starting time of integration of neural ODE.
      t1: End time of integration of neural ODE.
      rng: random seed for sampling from the latent distribution.
      kwargs: Keyword arguments for the ODE solver.

    Returns:
      The push-forward or pull-back distribution defined by the learnt
      transport plan.
    """

    def vf(t: jnp.ndarray, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      params = self.vf_state.params
      return self.vf_state.apply_fn({"params": params}, t, x, cond)

    def solve_ode(x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      ode_term = diffrax.ODETerm(vf)
      sol = diffrax.diffeqsolve(
          ode_term,
          t0=t0,
          t1=t1,
          y0=x,
          args=cond,
          **kwargs,
      )
      return sol.ys[0]

    kwargs.setdefault("dt0", None)
    kwargs.setdefault("solver", diffrax.Tsit5())
    kwargs.setdefault(
        "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
    )

    rng = utils.default_prng_key(rng)
    latent = self.latent_noise_fn(rng, (len(source),))

    if condition is not None:
      source = jnp.concatenate([source, condition], axis=-1)

    return jax.jit(jax.vmap(solve_ode))(latent, source)
