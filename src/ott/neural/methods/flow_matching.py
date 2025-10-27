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
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

import diffrax
import optax
from flax import nnx

__all__ = [
    "flow_matching_step",
    "interpolate_samples",
    "evaluate_velocity_field",
    "curvature",
    "gaussian_nll",
]

DivState = Tuple[jax.Array, jax.Array]  # velocity, divergence
Batch = Dict[Literal["t", "x_t", "v_t", "cond"], jax.Array]


def flow_matching_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: Batch,
    *,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = optax.squared_error,
    model_callback_fn: Optional[Callable[[nnx.Module], None]] = None,
    rngs: Optional[nnx.Rngs] = None,
) -> Dict[Literal["loss", "grad_norm"], jax.Array]:
  """Perform a flow matching step.

  Args:
    model: Velocity field with a signature ``(t, x_t, cond, rngs=...) -> v_t``.
    optimizer: Optimizer.
    batch: Batch containing the following elements:

      - ``'t'`` - time, array of shape ``[batch,]``.
      - ``'x_t'`` - position, array of shape ``[batch, ...]``.
      - ``'v_t'`` - target velocity, array of shape ``[batch, ...]``.
      - ``'cond'`` - condition (optional), array of shape ``[batch, ...]``.
    loss_fn: Loss function with a signature ``(pred, target) -> loss``.
    model_callback_fn: Function with a signature ``(model) -> None``, e.g., to
      update an :class:`~ott.neural.networks.velocity_field.EMA` of the model.
    rngs: Random number generator used for, e.g., dropout, passed to the model.

  Returns:
    Updates the parameters in-place and returns the loss and the gradient norm.
  """

  def compute_loss(model: nnx.Module, rngs: nnx.Rngs) -> jax.Array:
    t, x_t, v_t = batch["t"], batch["x_t"], batch["v_t"]
    cond = batch.get("cond")
    v_pred = model(t, x_t, cond, rngs=rngs)
    return loss_fn(v_pred, v_t).mean()

  loss, grads = nnx.value_and_grad(compute_loss)(model, rngs)
  if "model" in inspect.signature(optimizer.update).parameters:
    optimizer.update(model, grads)
  else:
    # for flax version < 0.11.0
    optimizer.update(grads)
  grad_norm = optax.global_norm(grads)

  if model_callback_fn is not None:
    model_callback_fn(model)

  return {"loss": loss, "grad_norm": grad_norm}


def interpolate_samples(
    rng: jax.Array,
    x0: jax.Array,
    x1: jax.Array,
    cond: Optional[jax.Array] = None,
    *,
    time_sampler: Optional[Callable[[jax.Array, Tuple[int], jnp.dtype],
                                    jax.Array]] = None
) -> Batch:
  """Sample time and interpolate.

  Args:
    rng: Random number generator.
    x0: Source samples at :math:`t_0`, array of shape ``[batch, ...]``.
    x1: Target samples at :math:`t_1`, array of shape ``[batch, ...]``.
    cond: Condition.
    time_sampler: Time sampler with signature ``(rng, shape, dtype) -> time``.

  Returns:
    Dictionary containing the following values:

    - ``'t'`` - time, array of shape ``[batch,]``.
    - ``'x_t'`` - position :math:`x_t`, array of shape ``[batch, ...]``.
    - ``'v_t'`` - target velocity :math:`x_1 - x_0`,
      array of shape ``[batch, ...]``.
    - ``'cond'`` - condition (optional), array of shape ``[batch, ...]``.
  """
  if time_sampler is None:
    time_sampler = jr.uniform

  batch_size = len(x0)
  t = time_sampler(rng, (batch_size,), x0.dtype)
  assert t.shape == (batch_size,), (t.shape, (batch_size,))
  t_ = jnp.expand_dims(t, axis=range(1, x0.ndim))

  batch = {
      "t": t,
      "x_t": (1.0 - t_) * x0 + t_ * x1,
      "v_t": x1 - x0,
  }
  if cond is not None:
    batch["cond"] = cond
  return batch


def evaluate_velocity_field(
    model: nnx.Module,
    x: Union[jax.Array, Any],
    cond: Optional[jax.Array] = None,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    reverse: bool = False,
    num_steps: Optional[int] = None,
    solver: Optional[diffrax.AbstractSolver] = None,
    save_trajectory_kwargs: Optional[Dict[str, Any]] = None,
    save_velocity_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> diffrax.Solution:
  """Solve an ODE.

  Args:
    model: Velocity field with a signature ``(t, x_t, cond) -> v_t``.
    x: Initial point of shape ``[*dims]``.
    cond: Condition of shape ``[*cond_dims]``.
    t0: Start time of the integration.
    t1: End time of the integration.
    reverse: Whether to integrate from :math:`t_1` to :math:`t_0`.
    num_steps: Number of steps used for solvers with a constant step size.
    solver: ODE solver. If :obj:`None` and ``step_size = None``,
      use :class:`~diffrax.Dopri5`. Otherwise use :class:`~diffrax.Euler`.
    save_velocity_kwargs: Keyword arguments for :class:`~diffrax.SubSaveAt`
      used to store the velocities along the integration path.
      The velocity will be saved in :class:`out.ys['v_t'] <diffrax.Solution>`.
    save_trajectory_kwargs: Keyword arguments for :class:`~diffrax.SubSaveAt`
      used to store the positions along the integration path.
      The trajectory will be saved in :class:`out.ys['x_t'] <diffrax.Solution>`.
    kwargs: Keyword arguments for :func:`~diffrax.diffeqsolve`.

  Returns:
    The ODE solution.
  """
  if isinstance(num_steps, int):
    step_size = 1.0 / num_steps
    stepsize_controller = diffrax.ConstantStepSize()
    solver = diffrax.Euler() if solver is None else solver
    kwargs["max_steps"] = num_steps
  else:
    step_size = None
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    solver = diffrax.Dopri5() if solver is None else solver

  if reverse:
    step_size = None if step_size is None else -step_size
    t0, t1 = t1, t0

  default_velocity_fn = jtu.Partial(_velocity, model=model)
  # internally, we allow for passing custom velocity functions:
  # this is used when computing the gaussian NLL, as we need to
  # both integrate the state and the divergence of the velocity field
  velocity_fn = kwargs.pop("_velocity_fn", default_velocity_fn)

  subs = {}
  if save_velocity_kwargs:
    saveat = diffrax.SubSaveAt(fn=default_velocity_fn, **save_velocity_kwargs)
    subs["v_t"] = saveat
  if save_trajectory_kwargs:
    saveat = diffrax.SubSaveAt(
        fn=lambda _, x_t, __: x_t, **save_trajectory_kwargs
    )
    subs["x_t"] = saveat

  if subs:
    kwargs["saveat"] = diffrax.SaveAt(subs=subs)

  return diffrax.diffeqsolve(
      diffrax.ODETerm(velocity_fn),
      t0=t0,
      t1=t1,
      y0=x,
      args=cond,
      solver=solver,
      dt0=step_size,
      stepsize_controller=stepsize_controller,
      **kwargs,
  )


def curvature(
    model: nnx.Module,
    x0: jax.Array,
    cond: Optional[jax.Array] = None,
    *,
    ts: Union[int, jax.Array, Sequence[float]],
    drop_last_velocity: Optional[bool] = None,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = optax.squared_error,
    **kwargs: Any,
) -> Tuple[jax.Array, diffrax.Solution]:
  """Compute the curvature :cite:`lee:23`.

  Also known as straightness in :cite:`liu:22`.

  Args:
    model: Velocity field with a signature ``(t, x_t, cond) -> v_t``.
    x0: Initial point of shape ``[*dims]``.
    cond: Condition of shape ``[*cond_dims]``.
    ts: Time points at which velocities are computed and stored.
      If :class:`int`, use linearly-spaced interval ``[t0, t1]``
      with ``ts`` steps.
    drop_last_velocity: Whether to remove the velocity at ``ts[-1]``.
      when computing the curvature. If :obj:`None`, don't include it when
      ``ts[-1] == 1.0``.
    loss_fn: Loss function with a signature ``(pred, target) -> loss``.
    kwargs: Keyword arguments for :func:`evaluate_velocity_field`.

  Returns:
    The curvature and the ODE solution.
  """
  if isinstance(ts, int):
    assert ts > 0, f"Number of steps must be positive, got {ts}."
    t0, t1 = kwargs.get("t0", 0.0), kwargs.get("t1", 1.0)
    ts = np.linspace(t0, t1, ts)
  if drop_last_velocity is None:
    drop_last_velocity = ts[-1] == 1.0

  sol = evaluate_velocity_field(
      model,
      x0,
      cond,
      reverse=False,
      save_trajectory_kwargs={"t1": True},  # save only at `t1`
      save_velocity_kwargs={"ts": ts},  # save `v_t` at specified times
      **kwargs,
  )
  x1 = sol.ys["x_t"][-1]
  v_t = sol.ys["v_t"][:-1] if drop_last_velocity else sol.ys["v_t"]

  steps = len(ts) - drop_last_velocity
  assert x0.shape == x1.shape, (x0.shape, x1.shape)
  assert v_t.shape == (steps, *x0.shape), (v_t.shape, (steps, *x0.shape))

  ref_velocity = (x1 - x0)
  curv = jax.vmap(loss_fn, in_axes=[0, None])(v_t, ref_velocity).mean()
  return curv, sol


def gaussian_nll(
    model: nnx.Module,
    x1: jax.Array,
    cond: Optional[jax.Array] = None,
    *,
    noise: Optional[jax.Array] = None,
    stddev: float = 1.0,
    **kwargs: Any,
) -> Tuple[jax.Array, diffrax.Solution]:
  """Compute the Gaussian negative log-likelihood.

  Args:
    model: Velocity model with a signature ``(t, x_t, cond) -> v_t``.
    x1: Initial point of shape ``[*dims]``.
    cond: Condition ``[*cond_dims]``.
    noise: Array of shape ``[num_noise_samples, ...]`` used for the Hutchinson's
      trace estimate of the divergence of the velocity field. If :obj:`None`,
      compute the exact divergence using :func:`jax.jacrev`.
    stddev: Standard deviation of the Gaussian distribution.
    kwargs: Keyword arguments for :func:`evaluate_velocity_field`.

  Returns:
    The Gaussian negative log-likelihood in bits-per-dimension.
  """
  if noise is not None:
    _, *noise_shape = noise.shape  # [batch, ...]
    assert x1.shape == tuple(noise_shape), (x1.shape, noise_shape)
    velocity_fn = functools.partial(_hutchinson_divergence, h=noise)
  else:
    velocity_fn = _exact_divergence

  sol = evaluate_velocity_field(
      model,
      (x1, jnp.zeros([])),  # initial point, divergence
      cond,
      reverse=True,
      saveat=diffrax.SaveAt(t1=True),
      save_trajectory_kwargs=None,
      save_velocity_kwargs=None,
      _velocity_fn=jtu.Partial(velocity_fn, model=model),
      **kwargs,
  )

  x0, neg_int01_div_v = sol.ys
  assert x0.shape == (1, *x1.shape), (x0.shape, (1, *x1.shape))
  assert neg_int01_div_v.shape == (1,), neg_int01_div_v.shape

  k = np.prod(x0.shape)
  logp0_x0 = -0.5 * ((x0 / stddev) ** 2).sum()
  logp0_x0 = logp0_x0 - 0.5 * k * jnp.log(2.0 * jnp.pi) - k * jnp.log(stddev)
  nll = -(logp0_x0 + neg_int01_div_v[0])
  return nll, sol


def _velocity(
    t: jax.Array, x_t: jax.Array, cond: Optional[jax.Array], model: nnx.Module
) -> jax.Array:
  cond = None if cond is None else cond[None]
  return model(t[None], x_t[None], cond).squeeze(0)


def _exact_divergence(
    t: jax.Array, state_t: DivState, cond: Optional[jax.Array], *,
    model: nnx.Module
) -> DivState:

  def divergence_v(
      t: jax.Array, x: jax.Array, cond: Optional[jax.Array]
  ) -> jax.Array:
    # divergence of fwd velocity field
    jacobian = jax.jacrev(_velocity, argnums=1)(t, x, cond, model)
    jacobian = jacobian.reshape(np.prod(x.shape), np.prod(x.shape))
    return jnp.trace(jacobian)

  x_t, _ = state_t
  v_t = _velocity(t, x_t, cond, model=model)
  div_t = divergence_v(t, x_t, cond)
  return v_t, div_t


def _hutchinson_divergence(
    t: jax.Array, state_t: DivState, cond: Optional[jax.Array], *,
    model: nnx.Module, h: jax.Array
) -> DivState:
  x_t, _ = state_t
  v_t, vjp = jax.vjp(lambda x: _velocity(t, x, cond, model=model), x_t)
  (Dvh,) = jax.vmap(vjp)(h)
  div_t = jax.vmap(jnp.vdot, in_axes=[0, 0])(h, Dvh).mean()
  return v_t, div_t
