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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

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

__all__ = ["GENOT"]

LinTerm = Tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray],
                 Optional[jnp.ndarray]]
DataMatchFn = Union[Callable[[LinTerm], jnp.ndarray], Callable[[QuadTerm],
                                                               jnp.ndarray]]


class GENOT:
  """Generative Entropic Neural Optimal Transport :cite:`klein_uscidda:23`.

  GENOT is a framework for learning neural optimal transport plans between
  two distributions. It allows for learning linear and quadratic
  (Fused) Gromov-Wasserstein couplings, in both the balanced and
  the unbalanced setting.

  Args:
    vf: Vector field parameterized by a neural network.
    flow: Flow between the latent and the target distributions.
    data_match_fn: Function to match samples from the source and the target
      distributions. Depending on the data passed in :meth:`__call__`, it has
      the following signature:

      - ``(src_lin, tgt_lin) -> matching`` - linear matching.
      - ``(src_quad, tgt_quad, src_lin, tgt_lin) -> matching`` -
        quadratic (fused) GW matching. In the pure GW setting, both ``src_lin``
        and ``tgt_lin`` will be set to :obj:`None`.

    source_dim: Dimensionality of the source distribution.
    target_dim: Dimensionality of the target distribution.
    condition_dim: Dimension of the conditions. If :obj:`None`, the underlying
      velocity field has no conditions.
    time_sampler: Time sampler with a ``(rng, n_samples) -> time`` signature.
    latent_noise_fn: Function to sample from the latent distribution in the
      target space with a ``(rng, shape) -> noise`` signature.
      If :obj:`None`, multivariate normal distribution is used.
    latent_match_fn: Function to match samples from the latent distribution
      and the samples from the conditional distribution with a
      ``(latent, samples) -> matching`` signature. If :obj:`None`, no matching
      is performed.
    n_samples_per_src: Number of samples drawn from the conditional distribution
      per one source sample.
    kwargs: Keyword arguments for
      :meth:`~ott.neural.networks.velocity_field.VelocityField.create_train_state`.
  """  # noqa: E501

  def __init__(
      self,
      vf: velocity_field.VelocityField,
      flow: dynamics.BaseFlow,
      data_match_fn: DataMatchFn,
      *,
      source_dim: int,
      target_dim: int,
      condition_dim: Optional[int] = None,
      time_sampler: Callable[[jax.Array, int],
                             jnp.ndarray] = solver_utils.uniform_sampler,
      latent_noise_fn: Optional[Callable[[jax.Array, Tuple[int, ...]],
                                         jnp.ndarray]] = None,
      latent_match_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                         jnp.ndarray]] = None,
      n_samples_per_src: int = 1,
      **kwargs: Any,
  ):
    self.vf = vf
    self.flow = flow
    self.data_match_fn = data_match_fn
    self.time_sampler = time_sampler
    if latent_noise_fn is None:
      latent_noise_fn = functools.partial(_multivariate_normal, dim=target_dim)
    self.latent_noise_fn = latent_noise_fn
    self.latent_match_fn = latent_match_fn
    self.n_samples_per_src = n_samples_per_src

    self.vf_state = self.vf.create_train_state(
        input_dim=target_dim,
        condition_dim=source_dim + (condition_dim or 0),
        **kwargs
    )
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
      ) -> jnp.ndarray:
        rng_flow, rng_dropout = jax.random.split(rng, 2)
        x_t = self.flow.compute_xt(rng_flow, time, latent, target)
        if source_conditions is None:
          cond = source
        else:
          cond = jnp.concatenate([source, source_conditions], axis=-1)

        v_t = vf_state.apply_fn({"params": params},
                                time,
                                x_t,
                                cond,
                                rngs={"dropout": rng_dropout})
        u_t = self.flow.compute_ut(time, latent, target)

        return jnp.mean((v_t - u_t) ** 2)

      grad_fn = jax.value_and_grad(loss_fn)
      loss, grads = grad_fn(
          vf_state.params, time, source, target, latent, source_conditions, rng
      )

      return loss, vf_state.apply_gradients(grads=grads)

    return step_fn

  def __call__(
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      n_iters: int,
      rng: Optional[jax.Array] = None
  ) -> Dict[str, List[float]]:
    """Train the GENOT model.

    Args:
      loader: Data loader returning a dictionary with possible keys
        `src_lin`, `tgt_lin`, `src_quad`, `tgt_quad`, `src_conditions`.
      n_iters: Number of iterations to train the model.
      rng: Random key for seeding.

    Returns:
      Training logs.
    """

    def prepare_data(
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray],
               Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray],
                     Optional[jnp.ndarray]]]:
      src_lin, src_quad = batch.get("src_lin"), batch.get("src_quad")
      tgt_lin, tgt_quad = batch.get("tgt_lin"), batch.get("tgt_quad")

      if src_quad is None and tgt_quad is None:  # lin
        src, tgt = src_lin, tgt_lin
        arrs = src_lin, tgt_lin
      elif src_lin is None and tgt_lin is None:  # quad
        src, tgt = src_quad, tgt_quad
        arrs = src_quad, tgt_quad
      elif all(
          arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)
      ):  # fused quad
        src = jnp.concatenate([src_lin, src_quad], axis=1)
        tgt = jnp.concatenate([tgt_lin, tgt_quad], axis=1)
        arrs = src_quad, tgt_quad, src_lin, tgt_lin
      else:
        raise RuntimeError("Cannot infer OT problem type from data.")

      return (src, batch.get("src_condition"), tgt), arrs

    rng = utils.default_prng_key(rng)
    training_logs = {"loss": []}
    for batch in loader:
      rng = jax.random.split(rng, 5)
      rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

      batch = jtu.tree_map(jnp.asarray, batch)
      (src, src_cond, tgt), matching_data = prepare_data(batch)

      n = src.shape[0]
      time = self.time_sampler(rng_time, n * self.n_samples_per_src)
      latent = self.latent_noise_fn(rng_noise, (n, self.n_samples_per_src))

      tmat = self.data_match_fn(*matching_data)  # (n, m)
      src_ixs, tgt_ixs = solver_utils.sample_conditional(  # (n, k), (m, k)
          rng_resample,
          tmat,
          k=self.n_samples_per_src,
      )

      src, tgt = src[src_ixs], tgt[tgt_ixs]  # (n, k, ...),  # (m, k, ...)
      if src_cond is not None:
        src_cond = src_cond[src_ixs]

      if self.latent_match_fn is not None:
        src, src_cond, tgt = self._match_latent(rng, src, src_cond, latent, tgt)

      src = src.reshape(-1, *src.shape[2:])  # (n * k, ...)
      tgt = tgt.reshape(-1, *tgt.shape[2:])  # (m * k, ...)
      latent = latent.reshape(-1, *latent.shape[2:])
      if src_cond is not None:
        src_cond = src_cond.reshape(-1, *src_cond.shape[2:])

      loss, self.vf_state = self.step_fn(
          rng_step_fn, self.vf_state, time, src, tgt, latent, src_cond
      )

      training_logs["loss"].append(float(loss))
      if len(training_logs["loss"]) >= n_iters:
        break

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

      src_ixs, tgt_ixs = solver_utils.sample_joint(rng, tmat)  # (n,), (m,)
      src, tgt = src[src_ixs], tgt[tgt_ixs]
      if src_cond is not None:
        src_cond = src_cond[src_ixs]

      return src, src_cond, tgt

    cond_axis = None if src_cond is None else 1
    in_axes, out_axes = (0, 1, cond_axis, 1, 1), (1, cond_axis, 1)
    resample_fn = jax.jit(jax.vmap(resample, in_axes, out_axes))

    rngs = jax.random.split(rng, self.n_samples_per_src)
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
    """Transport data with the learned plan.

    This function pushes forward the source distribution to its conditional
    distribution by solving the neural ODE.

    Args:
      source: Data to transport.
      condition: Condition of the input data.
      t0: Starting time of integration of neural ODE.
      t1: End time of integration of neural ODE.
      rng: Random generate used to sample from the latent distribution.
      kwargs: Keyword arguments for :func:`~diffrax.odesolve`.

    Returns:
      The push-forward defined by the learned transport plan.
    """

    def vf(t: jnp.ndarray, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      params = self.vf_state.params
      return self.vf_state.apply_fn({"params": params}, t, x, cond, train=False)

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


def _multivariate_normal(
    rng: jax.Array,
    shape: Tuple[int, ...],
    dim: int,
    mean: float = 0.0,
    cov: float = 1.0
) -> jnp.ndarray:
  mean = jnp.full(dim, fill_value=mean)
  cov = jnp.diag(jnp.full(dim, fill_value=cov))
  return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)
