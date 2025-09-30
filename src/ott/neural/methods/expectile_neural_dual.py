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
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import optax
from flax import linen as nn
from flax.core import frozen_dict

from ott import utils
from ott.geometry import costs
from ott.neural.networks import potentials
from ott.problems.linear import potentials as dual_potentials

__all__ = ["ENOTPotentials", "PotentialModelWrapper", "ExpectileNeuralDual"]

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable[[int, dual_potentials.DualPotentials], None]


@jtu.register_static
class ENOTPotentials(dual_potentials.DualPotentials):
  """The dual potentials of the ENOT method :cite:`buzun:24`.

  Args:
    grad_f: Gradient of the first dual potential function.
    g: The second dual potential function.
    cost_fn: The cost function used to solve the OT problem.
  """

  def __init__(
      self,
      grad_f: potentials.PotentialGradientFn_t,
      g: potentials.PotentialValueFn_t,
      *,
      cost_fn: costs.TICost,
  ):
    self.__grad_f = grad_f

    def f_potential(x: jnp.ndarray) -> jnp.ndarray:
      y_hat = cost_fn.twist_operator(x, grad_f(x), False)
      y_hat = jax.lax.stop_gradient(y_hat)
      return -g(y_hat) + cost_fn(x, y_hat)

    super().__init__(f_potential, g, cost_fn=cost_fn)

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return jax.vmap(self.__grad_f)


class PotentialModelWrapper(potentials.BasePotential):
  """Wrapper class for the neural models.

  Implements a potential value or a vector field.

  Args:
    model: Network architecture of the potential.
    is_potential: Model the potential if ``True``, otherwise
      model the gradient of the potential.
  """

  model: nn.Module
  is_potential: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply model and optionally add l2 norm or x."""
    z: jnp.ndarray = self.model(x)
    if self.is_potential:
      z = z.squeeze()
    return z

  def potential_gradient_fn(
      self, params: frozen_dict.FrozenDict[str, jnp.ndarray]
  ) -> potentials.PotentialGradientFn_t:
    """A vector function or gradient of the potential."""
    if self.is_potential:
      return jax.grad(self.potential_value_fn(params))
    return lambda x: self.apply({"params": params}, x)


class ExpectileNeuralDual:
  r"""Expectile-regularized Neural Optimal Transport (ENOT) :cite:`buzun:24`.

  It solves the dual optimal transport problem for a specified cost function
  :math:`c(x, y)` between two measures :math:`\alpha` and :math:`\beta` in
  :math:`d`-dimensional Euclidean space with additional regularization on
  :term:`dual Kantorovich potentials`. The expectile regularization enforces
  binding conditions on the learning dual potentials :math:`f` and :math:`g`.
  The main optimization objective is

  .. math::

    \sup_{g \in L_1(\beta)} \inf_{T: \, R^d \to R^d} \big[
      \mathbb{E}_{\alpha}[c(x, T(x))] + \mathbb{E}_{\beta} [g(y)]
      - \mathbb{E}_{\alpha} [g(T(x))]
    \big],

  where :math:`T(x)` is the transport mapping from :math:`\alpha`
  to :math:`\beta` expressed through :math:`\nabla f(x)`. The explicit formula
  for that transport depends on the twist operator of the cost function.
  The regularization term is

  .. math::

    \mathbb{E} \mathcal{L}_{\tau} \big(
      c(x, T(x)) - g(T(x)) - c(x, y)  + g(y)
    \big),

  where :math:`\mathcal{L}_{\tau}` is the least asymmetrically weighted
  squares loss from expectile regression.

  Args:
    dim_data: Input dimensionality of data required for network init.
    neural_f: Network architecture for potential :math:f or
      its gradient :math:`\nabla f`.
    neural_g: Network architecture for potential :math:`g`.
    optimizer_f: Optimizer function for potential :math:`f`.
    optimizer_g: Optimizer function for potential :math:`g`.
    cost_fn: Translation invariant cost function.
    expectile: Parameter of the expectile loss (:math:`\tau`).
      Suggested values range is :math:`[0.9, 1.0)`.
    expectile_loss_coef: Expectile loss weight.
      Suggested values range is :math:`[0.3, 1.0]`.
    num_train_iters: Number of total training iterations.
    valid_freq: Frequency with which model is validated.
    log_freq: Frequency with training and validation are logged.
    logging: Option to return logs.
    rng: Random key used for seeding for network initializations.
  """

  def __init__(
      self,
      dim_data: int,
      neural_f: Optional[nn.Module] = None,
      neural_g: Optional[nn.Module] = None,
      optimizer_f: Optional[optax.GradientTransformation] = None,
      optimizer_g: Optional[optax.GradientTransformation] = None,
      cost_fn: Optional[costs.TICost] = None,
      expectile: float = 0.99,
      expectile_loss_coef: float = 1.0,
      num_train_iters: int = 20000,
      valid_freq: int = 1000,
      log_freq: int = 1000,
      logging: bool = False,
      rng: Optional[jax.Array] = None
  ):
    self.num_train_iters = num_train_iters
    self.valid_freq = valid_freq
    self.log_freq = log_freq
    self.logging = logging
    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    self.expectile = expectile
    self.expectile_loss_coef = expectile_loss_coef
    self.train_batch_cost = jax.vmap(self.cost_fn)

    # set default optimizers
    if optimizer_f is None:
      optimizer_f = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.9)
    if optimizer_g is None:
      optimizer_g = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.9)

    # set default neural architectures
    if neural_f is None:
      neural_f = potentials.MLP(
          dim_hidden=[128, 128, 128, 128, 1], act_fn=jax.nn.elu
      )
    if neural_g is None:
      neural_g = potentials.MLP(
          dim_hidden=[128, 128, 128, 128, 1], act_fn=jax.nn.elu
      )

    self.neural_f = PotentialModelWrapper(model=neural_f, is_potential=True)
    self.neural_g = PotentialModelWrapper(model=neural_g, is_potential=True)

    rng = utils.default_prng_key(rng)
    rng_f, rng_g = jax.random.split(rng, 2)

    self.state_f = self.neural_f.create_train_state(
        rng_f, optimizer_f, (dim_data,)
    )
    self.state_g = self.neural_g.create_train_state(
        rng_g, optimizer_g, (dim_data,)
    )

    self.train_step = self._get_train_step()
    self.valid_step = self._get_valid_step()

  def __call__(
      self,
      trainloader_source: Iterator[jnp.ndarray],
      trainloader_target: Iterator[jnp.ndarray],
      validloader_source: Iterator[jnp.ndarray],
      validloader_target: Iterator[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Union[ENOTPotentials, Tuple[ENOTPotentials, Train_t]]:
    """Train and return the Kantorovich dual potentials."""
    logs = self.train_fn(
        trainloader_source,
        trainloader_target,
        validloader_source,
        validloader_target,
        callback=callback,
    )
    res = self.to_dual_potentials()

    return (res, logs) if self.logging else res

  def train_fn(
      self,
      trainloader_source: Iterator[jnp.ndarray],
      trainloader_target: Iterator[jnp.ndarray],
      validloader_source: Iterator[jnp.ndarray],
      validloader_target: Iterator[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Train_t:
    """Training and validation."""
    try:
      from tqdm.auto import tqdm
    except ImportError:
      tqdm = lambda _: _

    train_batch, valid_batch = {}, {}

    train_logs = {"loss_f": [], "loss_g": [], "w_dist": [], "directions": []}
    valid_logs = {"loss_f": [], "loss_g": [], "w_dist": []}

    for step in tqdm(range(self.num_train_iters)):

      update_forward = (step % 2 == 0)

      if update_forward:
        train_batch["source"] = jnp.asarray(next(trainloader_source))
        train_batch["target"] = jnp.asarray(next(trainloader_target))
        (self.state_f, self.state_g, loss, loss_f, loss_g,
         w_dist) = self.train_step(self.state_f, self.state_g, train_batch)
      else:
        train_batch["target"] = jnp.asarray(next(trainloader_source))
        train_batch["source"] = jnp.asarray(next(trainloader_target))
        (self.state_g, self.state_f, loss, loss_g, loss_f,
         w_dist) = self.train_step(self.state_g, self.state_f, train_batch)

      if self.logging and step % self.log_freq == 0:
        self._update_logs(train_logs, loss_f, loss_g, w_dist)

      if callback is not None:
        _ = callback(step, self.to_dual_potentials())

      if step != 0 and step % self.valid_freq == 0:
        valid_batch["source"] = jnp.asarray(next(validloader_source))
        valid_batch["target"] = jnp.asarray(next(validloader_target))

        valid_loss_f, valid_loss_g, valid_w_dist = self.valid_step(
            self.state_f, self.state_g, valid_batch
        )

        if self.logging:
          self._update_logs(
              valid_logs, valid_loss_f, valid_loss_g, valid_w_dist
          )

    return {"train_logs": train_logs, "valid_logs": valid_logs}

  def _get_train_step(
      self
  ) -> Callable[[
      potentials.PotentialTrainState, potentials.PotentialTrainState, Dict[
          str, jnp.ndarray]
  ], Tuple[potentials.PotentialTrainState, potentials.PotentialTrainState,
           jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    @jax.jit
    def step_fn(state_f, state_g, batch):
      grad_fn = jax.value_and_grad(self._loss_fn, argnums=[0, 1], has_aux=True)
      (loss, (loss_f, loss_g, w_dist)), (grads_f, grads_g) = grad_fn(
          state_f.params,
          state_g.params,
          state_f.potential_gradient_fn,
          state_g.potential_value_fn,
          batch,
      )

      return (
          state_f.apply_gradients(grads=grads_f),
          state_g.apply_gradients(grads=grads_g), loss, loss_f, loss_g, w_dist
      )

    return step_fn

  def _get_valid_step(
      self
  ) -> Callable[[
      potentials.PotentialTrainState, potentials.PotentialTrainState, Dict[
          str, jnp.ndarray]
  ], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    @jax.jit
    def step_fn(state_f, state_g, batch):
      loss, (loss_f, loss_g, w_dist) = self._loss_fn(
          state_f.params,
          state_g.params,
          state_f.potential_gradient_fn,
          state_g.potential_value_fn,
          batch,
      )

      return loss_f, loss_g, w_dist

    return step_fn

  def _expectile_loss(self, diff: jnp.ndarray) -> jnp.ndarray:
    """Loss of the expectile regression :cite:`buzun:24`."""
    weight = jnp.where(diff >= 0, self.expectile, (1 - self.expectile))
    return weight * diff ** 2

  def _get_g_value_partial(
      self, params_g: frozen_dict.FrozenDict[str, jnp.ndarray],
      g_value: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                        potentials.PotentialValueFn_t]
  ):

    g_value_partial = jax.vmap(g_value(params_g))
    g_value_partial_detach = jax.vmap(g_value(jax.lax.stop_gradient(params_g)))

    return g_value_partial, g_value_partial_detach

  def _loss_fn(
      self, params_f: frozen_dict.FrozenDict[str, jnp.ndarray],
      params_g: frozen_dict.FrozenDict[str, jnp.ndarray],
      gradient_f: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                           potentials.PotentialGradientFn_t],
      g_value: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                        potentials.PotentialValueFn_t], batch: Dict[str,
                                                                    jnp.ndarray]
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    source, target = batch["source"], batch["target"]

    g_value_partial, g_value_partial_detach = self._get_g_value_partial(
        params_g, g_value
    )
    batch_cost = self.train_batch_cost

    transport = ENOTPotentials(
        gradient_f(params_f),
        g_value(params_g),
        cost_fn=self.cost_fn,
    ).transport

    target_hat = transport(source)
    target_hat_detach = jax.lax.stop_gradient(target_hat)

    g_target = g_value_partial(target)
    g_star_source = batch_cost(source, target_hat_detach
                              ) - g_value_partial(target_hat_detach)

    diff_1 = jax.lax.stop_gradient(
        g_star_source - batch_cost(source, target)
    ) + g_target
    reg_loss_1 = self._expectile_loss(diff_1).mean()

    diff_2 = jax.lax.stop_gradient(
        g_target - batch_cost(source, target)
    ) + g_star_source
    reg_loss_2 = self._expectile_loss(diff_2).mean()

    reg_loss = (reg_loss_1 + reg_loss_2) * self.expectile_loss_coef
    dual_loss = -(g_target + g_star_source).mean()
    amor_loss = (
        batch_cost(source, target_hat) - g_value_partial_detach(target_hat)
    ).mean()

    loss = reg_loss + dual_loss + amor_loss
    f_loss = amor_loss
    g_loss = reg_loss + dual_loss
    w_dist = g_star_source.mean() + g_target.mean()

    return loss, (f_loss, g_loss, w_dist)

  def to_dual_potentials(self) -> ENOTPotentials:
    """Return the Kantorovich dual potentials from the trained potentials."""
    f_grad_partial = self.state_f.potential_gradient_fn(self.state_f.params)
    g_value_partial = self.state_g.potential_value_fn(self.state_g.params, None)

    return ENOTPotentials(
        f_grad_partial,
        g_value_partial,
        cost_fn=self.cost_fn,
    )

  @staticmethod
  def _update_logs(
      logs: Dict[str, List[Union[float, str]]],
      loss_f: jnp.ndarray,
      loss_g: jnp.ndarray,
      w_dist: jnp.ndarray,
  ) -> None:
    logs["loss_f"].append(float(loss_f))
    logs["loss_g"].append(float(loss_g))
    logs["w_dist"].append(float(w_dist))
