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
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
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
from ott.neural.networks.potentials import (
    PotentialGradientFn_t,
    PotentialTrainState,
    PotentialValueFn_t,
)
from ott.problems.linear.potentials import DualPotentials

__all__ = [
    "ENOTPotentials", "PotentialModelWrapper", "MLP", "ExpectileNeuralDual"
]

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable[[int, DualPotentials], None]
CostFn_t = TypeVar("CostFn_t", bound=costs.CostFn)


@jtu.register_pytree_node_class
class ENOTPotentials(DualPotentials):
  r"""The dual potentials for bidirectional mapping with TICost.

  Args:
  g: The second dual potential function.
  grad_f: Gradient of the first dual potential function.
  cost_fn: The translation invariant (TI) cost function used to solve
  the OT problem.
  """

  def __init__(
      self, g: PotentialValueFn_t, grad_f: PotentialGradientFn_t,
      cost_fn: costs.CostFn, is_bidirectional: bool, corr: bool
  ):

    self.is_bidirectional = is_bidirectional

    if is_bidirectional and not corr:
      grad_h_inv = jax.grad(cost_fn.h_legendre)
      transport = lambda x: x - grad_h_inv(grad_f(x))
    else:
      transport = lambda x: grad_f(x)

    conjugate_cost = jnp.dot if corr else cost_fn

    def g_cost_conjugate(x: jnp.ndarray) -> jnp.ndarray:
      y_hat = jax.lax.stop_gradient(transport(x))
      return -g(y_hat) + conjugate_cost(x, y_hat)

    super().__init__(g_cost_conjugate, g, cost_fn=cost_fn, corr=corr)
    self.__grad_f = grad_f

    assert isinstance(cost_fn, costs.TICost), (
        "Cost must be a `TICost` and "
        "provide access to Legendre transform of `h`."
    )

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return jax.vmap(self.__grad_f)

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport from source to the target distribution or vice-versa."""
    if self.is_bidirectional:
      return super().transport(vec, forward)

    vec = jnp.atleast_2d(vec)
    assert (forward
            is True), ("Only forward mapping (source -> target) is supported.")
    return self._grad_f(vec)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [], {
        "g": self._g,
        "grad_f": self._grad_f,
        "cost_fn": self.cost_fn,
        "is_bidirectional": self.is_bidirectional,
        "corr": self._corr
    }


class PotentialModelWrapper(nn.Module):
  r"""Wrapper class for the neural models.

  Implements a potential value or a vector field.
  """

  is_potential: bool  # Implements a potential value or a vector field.
  add_l2_norm: bool  # If true, l2 norm is added to the potential.
  model: nn.Module  # The potential model.

  def apply(
      self, params: frozen_dict.FrozenDict[str, jnp.ndarray], x: jnp.ndarray
  ) -> jnp.ndarray:
    """Apply the potential model, optionally add squared Euclidean norm."""
    z: jnp.ndarray = self.model.apply({"params": params}, x)

    if self.is_potential:
      z = z.squeeze()

    if self.is_potential and self.add_l2_norm:
      z = z + 0.5 * jnp.dot(x, x)
    if not self.is_potential and self.add_l2_norm:
      z = z + x

    return z

  def potential_value_fn(
      self, params: frozen_dict.FrozenDict[str, jnp.ndarray]
  ) -> PotentialValueFn_t:
    """Return a function giving the value of the potential."""
    assert (
        self.is_potential is True
    ), "A model should be potential to get potential_value_fn."
    return lambda x: self.apply(params, x)

  def potential_gradient_fn(
      self, params: frozen_dict.FrozenDict[str, jnp.ndarray]
  ) -> PotentialGradientFn_t:
    """A vector function or gradient of the potential."""
    if self.is_potential:
      return jax.grad(self.potential_value_fn(params))
    return lambda x: self.apply(params, x)

  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.GradientTransformation,
      input: Union[int, Tuple[int, ...]],
      **kwargs: Any,
  ) -> PotentialTrainState:
    """Create initial training state."""
    params = self.model.init(rng, jnp.ones(input))["params"]

    return PotentialTrainState.create(
        apply_fn=self.apply,
        params=params,
        tx=optimizer,
        potential_value_fn=self.potential_value_fn,
        potential_gradient_fn=self.potential_gradient_fn,
        **kwargs
    )


class MLP(nn.Module):
  """A simple MLP model of a potential used in default initialization."""

  dim_hidden: Sequence[int]
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply MLP transform."""
    for feat in self.dim_hidden[:-1]:
      x = self.act_fn(nn.Dense(feat)(x))
    return nn.Dense(self.dim_hidden[-1])(x)


class ExpectileNeuralDual(Generic[CostFn_t]):
  r"""Expectile-regularized Neural Optimal Transport (ENOT) :cite:`buzun:24`.

  It solves the dual optimal transport problem for a specified cost function
  c(x, y) between two measures :math:`\alpha` and :math:`\beta` in
  :math:`d`-dimensional Euclidean space with additional regularization on
  Kantorovich potentials. The expectile regularization enforces binding
  conditions on the learning dual potentials :math:`f` and :math:`g`.
  The main optimization objective is

  .. math::

    \sup_{g \in L_1(\beta)} \inf_{T: \, R^d \to R^d} \big[
      \mathbb{E}_{\alpha}[c(x, T(x))] + \mathbb{E}_{\beta} [g(y)]
      - \mathbb{E}_{\alpha} [g(T(x))]
    \big],

  where :math:`T(x)` is the transport mapping from :math:`\alpha`
  to :math:`\beta` expressed through :math:`\nabla f(x)`. The explicit
  formula depends on the cost function and ``is_bidirectional`` training
  option. The regularization term is

  .. math::

    \mathbb{E} \mathcal{L}_{\tau} \big(
      c(x, T(x)) - g(T(x)) - c(x, y)  + g(y)
    \big),

  where :math:`\mathcal{L}_{\tau}` is the least asymmetrically weighted
  squares loss from expectile regression.

  The :class:`flax.linen.Module` potentials for ``neural_f``
  and ``neural_g`` can

  1. both provide the values of the potentials :math:`f` and :math:`g`, or
  2. ``neural_f`` can provide the gradient :math:`\nabla f` for mapping T.

  Args:
    dim_data: input dimensionality of data required for network init.
    neural_f: network architecture for potential :math:`f`.
    neural_g: network architecture for potential :math:`g`.
    optimizer_f: optimizer function for potential :math:`f`.
    optimizer_g: optimizer function for potential :math:`g`.
    cost_fn: cost function of the OT problem.
    is_bidirectional: alternate between updating the forward and backward
      directions. Inspired from :cite:`jacobs:20`.
    num_train_iters: number of total training iterations.
    valid_freq: frequency with which model is validated.
    log_freq: frequency with training and validation are logged.
    logging: option to return logs.
    rng: random key used for seeding for network initializations.
    expectile: parameter of the expectile loss (:math:`\tau`).
      Suggested values range is [0.9, 1.0).
    expectile_loss_coef: expectile loss weight.
      Suggested values range is [0.3, 1.0].
  """

  def __init__(
      self,
      dim_data: int,
      neural_f: Optional[nn.Module] = None,
      neural_g: Optional[nn.Module] = None,
      optimizer_f: Optional[optax.GradientTransformation] = None,
      optimizer_g: Optional[optax.GradientTransformation] = None,
      cost_fn: Optional[CostFn_t] = None,
      is_bidirectional: bool = True,
      num_train_iters: int = 20000,
      valid_freq: int = 1000,
      log_freq: int = 1000,
      logging: bool = False,
      rng: Optional[jax.Array] = None,
      use_dot_product: bool = False,
      expectile: float = 0.99,
      expectile_loss_coef: float = 1.0
  ):
    self.num_train_iters = num_train_iters
    self.valid_freq = valid_freq
    self.log_freq = log_freq
    self.logging = logging
    self.cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    self.expectile = expectile
    self.expectile_loss_coef = expectile_loss_coef
    self.is_bidirectional = is_bidirectional
    self.use_dot_product = use_dot_product

    if use_dot_product:
      self.train_batch_cost = lambda x, y: -jax.vmap(jnp.dot)(x, y)
    else:
      self.train_batch_cost = jax.vmap(self.cost_fn)

    assert (isinstance(cost_fn, costs.TICost) or not is_bidirectional), (
        "is_bidirectional=True can only be used with a translation invariant"
        "cost (TICost)"
    )

    assert (isinstance(cost_fn, costs.SqEuclidean) or not use_dot_product
           ), ("use_dot_product=True can only be used with SqEuclidean cost")

    # set default optimizers
    if optimizer_f is None:
      optimizer_f = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.9)
    if optimizer_g is None:
      optimizer_g = optax.adam(learning_rate=3e-4, b1=0.9, b2=0.9)

    # set default neural architectures
    if neural_f is None:
      last_dim = 1 if is_bidirectional else dim_data
      neural_f = MLP(
          dim_hidden=[128, 128, 128, 128, last_dim], act_fn=jax.nn.elu
      )
    if neural_g is None:
      neural_g = MLP(dim_hidden=[128, 128, 128, 128, 1], act_fn=jax.nn.elu)

    self.neural_f = PotentialModelWrapper(
        model=neural_f,
        is_potential=is_bidirectional,
        add_l2_norm=self.use_dot_product
    )
    self.neural_g = PotentialModelWrapper(
        model=neural_g, is_potential=True, add_l2_norm=self.use_dot_product
    )

    rng = utils.default_prng_key(rng)
    rng, rng_f, rng_g = jax.random.split(rng, 3)

    self.state_f = self.neural_f.create_train_state(
        rng_f, optimizer_f, (1, dim_data)
    )
    self.state_g = self.neural_g.create_train_state(
        rng_g, optimizer_g, (1, dim_data)
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
  ) -> Union[DualPotentials, Tuple[DualPotentials, Train_t]]:
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

      update_forward = (step % 2 == 0) or not self.is_bidirectional

      if update_forward:
        train_batch["source"] = jnp.asarray(next(trainloader_source))
        train_batch["target"] = jnp.asarray(next(trainloader_target))
        (self.state_f, self.state_g, loss, loss_f, loss_g,
         w_dist) = self.train_step(self.state_f, self.state_g, train_batch)
      else:
        train_batch["target"] = jnp.asarray(next(trainloader_source))
        train_batch["source"] = jnp.asarray(next(trainloader_target))
        (self.state_g, self.state_f, loss, loss_f, loss_g,
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
  ) -> Callable[
      [PotentialTrainState, PotentialTrainState, Dict[str, jnp.ndarray]], Tuple[
          PotentialTrainState, PotentialTrainState, jnp.ndarray, jnp.ndarray,
          jnp.ndarray, jnp.ndarray]]:
    loss_fn = self._loss_fn

    @jax.jit
    def step_fn(state_f, state_g, batch):
      grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
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
  ) -> Callable[
      [PotentialTrainState, PotentialTrainState, Dict[str, jnp.ndarray]], Tuple[
          jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    loss_fn = self._loss_fn

    @jax.jit
    def step_fn(state_f, state_g, batch):
      grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
      (loss, (loss_f, loss_g, w_dist)), _ = grad_fn(
          state_f.params,
          state_g.params,
          state_f.potential_gradient_fn,
          state_g.potential_value_fn,
          batch,
      )

      return loss_f, loss_g, w_dist

    return step_fn

  def _expectile_loss(self, diff: jnp.ndarray) -> jnp.ndarray:
    weight = jnp.where(diff >= 0, self.expectile, (1 - self.expectile))
    return weight * diff ** 2

  def _loss_fn(
      self, params_f: frozen_dict.FrozenDict[str, jnp.ndarray],
      params_g: frozen_dict.FrozenDict[str, jnp.ndarray],
      gradient_f: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                           PotentialGradientFn_t],
      g_value: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                        PotentialValueFn_t], batch: Dict[str, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    if self.is_bidirectional and not self.use_dot_product:
      grad_h_inv = jax.vmap(jax.grad(self.cost_fn.h_legendre))
      transport = lambda grad_f, source: source - grad_h_inv(grad_f(source))
    else:
      transport = lambda grad_f, source: grad_f(source)

    source, target = batch["source"], batch["target"]

    f_grad_partial = jax.vmap(gradient_f(params_f))
    g_value_partial = jax.vmap(g_value(params_g))
    g_value_partial_stop_grad = jax.vmap(
        g_value(jax.lax.stop_gradient(params_g))
    )
    batch_cost = self.train_batch_cost

    if self.use_dot_product:
      g_value_partial = lambda y: -jax.vmap(g_value(params_g))(y)
      g_value_partial_stop_grad = lambda y: -jax.vmap(
          g_value(jax.lax.stop_gradient(params_g))
      )(
          y
      )

    target_hat = transport(f_grad_partial, source)
    target_hat_detach = jax.lax.stop_gradient(target_hat)

    g_target = g_value_partial(target)
    g_star_source = batch_cost(source, target_hat_detach
                              ) - g_value_partial(target_hat_detach)

    diff_1 = jax.lax.stop_gradient(g_star_source - batch_cost(source, target))\
      + g_target
    reg_loss_1 = self._expectile_loss(diff_1).mean()

    diff_2 = jax.lax.stop_gradient(g_target - batch_cost(source, target))\
      + g_star_source
    reg_loss_2 = self._expectile_loss(diff_2).mean()

    reg_loss = (reg_loss_1 + reg_loss_2) * self.expectile_loss_coef
    dual_loss = -(g_target.mean() + g_star_source.mean())
    amor_loss = (
        batch_cost(source, target_hat) - g_value_partial_stop_grad(target_hat)
    ).mean()

    loss = reg_loss + dual_loss + amor_loss
    w_dist = (g_target.mean() + g_star_source.mean())

    if self.use_dot_product:
      w_dist = jnp.mean(jnp.sum(source ** 2, axis=-1)) + \
               jnp.mean(jnp.sum(target ** 2, axis=-1)) + \
               2 * w_dist

    return loss, (dual_loss, amor_loss, w_dist)

  def to_dual_potentials(self) -> DualPotentials:
    """Return the Kantorovich dual potentials from the trained potentials."""
    f_grad_partial = self.state_f.potential_gradient_fn(self.state_f.params)
    g_value_partial = self.state_g.potential_value_fn(self.state_g.params)

    return ENOTPotentials(
        g_value_partial, f_grad_partial, self.cost_fn, self.is_bidirectional,
        self.use_dot_product
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
