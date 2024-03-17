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
import optax
from flax import linen as nn
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state
from jax._src.basearray import Array as Array

from ott import utils
from ott.geometry import costs
from ott.problems.linear.potentials import DualPotentials

__all__ = [
    "BidirectionalPotentials", "ForwardPotentials", "EuclideanPotentials",
    "DotCost", "PotentialModelWrapper", "PotentialTrainState", "MLP",
    "ExpectileNeuralDual"
]

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable[[int, DualPotentials], None]
PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]
CostFn_t = TypeVar("CostFn_t", bound=costs.CostFn)


class BidirectionalPotentials(DualPotentials):
  """The Kantorovich dual potentials functions, trained for bidirectional 
  mapping with TICost.

  Args:
  g: The second dual potential function.
  grad_f: Gradient of the first dual potential function.
  cost_fn: The translation invariant (TI) cost function used to solve 
  the OT problem.
  """

  def __init__(
      self, g: PotentialValueFn_t, grad_f: PotentialGradientFn_t,
      cost_fn: costs.TICost
  ):

    def g_cost_conjugate(x: jnp.ndarray) -> jnp.ndarray:
      grad_h_inv = jax.grad(cost_fn.h_legendre)
      y_hat = jax.lax.stop_gradient(x - grad_h_inv(grad_f(x)))
      return -g(y_hat) + cost_fn(x, y_hat)

    super().__init__(g_cost_conjugate, g, cost_fn=cost_fn, corr=False)
    self.__grad_f = grad_f

    assert isinstance(cost_fn, costs.TICost), (
        "Cost must be a `TICost` and "
        "provide access to Legendre transform of `h`."
    )

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return jax.vmap(self.__grad_f)

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport the points from source to the target distribution 
    or vice-versa.
    """
    vec = jnp.atleast_2d(vec)
    if forward:
      return vec - self._grad_h_inv(self._grad_f(vec))
    return vec - self._grad_h_inv(self._grad_g(vec))


class ForwardPotentials(DualPotentials):
  """The Kantorovich dual potentials functions, trained for only 
  forward mapping.

  Args:
  g: The second dual potential function.
  grad_f: Gradient of the first dual potential function.
  cost_fn: The cost function used to solve the OT problem.
  """

  def __init__(
      self, g: PotentialValueFn_t, grad_f: PotentialGradientFn_t,
      cost_fn: costs.CostFn
  ):

    def g_cost_conjugate(x: jnp.ndarray) -> jnp.ndarray:
      y_hat = jax.lax.stop_gradient(grad_f(x))
      return -g(y_hat) + cost_fn(x, y_hat)

    super().__init__(g_cost_conjugate, g, cost_fn=cost_fn, corr=False)
    self.__grad_f = grad_f

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return jax.vmap(self.__grad_f)

  def transport(self, source: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport the points from source to the target distribution."""
    source = jnp.atleast_2d(source)
    assert (forward is True), (
      "Only forward mapping (source -> target) is supported."
    )
    return self._grad_f(source)


class EuclideanPotentials(DualPotentials):
  """The Kantorovich dual potentials functions, trained with 
  scalar product operation.

  Args:
  g: The second dual potential function.
  grad_f: Gradient of the first dual potential function.
  """

  def __init__(self, g: PotentialValueFn_t, grad_f: PotentialGradientFn_t):

    def g_conjugate(x: jnp.ndarray) -> jnp.ndarray:
      y_hat = jax.lax.stop_gradient(grad_f(x))
      return -g(y_hat) + jnp.dot(x, y_hat)

    super().__init__(g_conjugate, g, cost_fn=costs.SqEuclidean(), corr=True)
    self.__grad_f = grad_f

  @property
  def _grad_f(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return jax.vmap(self.__grad_f)

  def transport(self, vec: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
    """Transport the points from source to the target distribution 
    or vice-versa.
    """
    vec = jnp.atleast_2d(vec)
    return self._grad_f(vec) if forward else self._grad_g(vec)

  def distance(self, src: jnp.ndarray, tgt: jnp.ndarray) -> float:
    """W2 distance."""
    src, tgt = jnp.atleast_2d(src), jnp.atleast_2d(tgt)
    f = jax.vmap(self.f)
    g = jax.vmap(self.g)
    corr = jnp.mean(f(src)) + jnp.mean(g(tgt))
    return -2.0 * corr \
      + jnp.mean(jnp.sum(src ** 2, axis=-1)) \
      + jnp.mean(jnp.sum(tgt ** 2, axis=-1))
    

def expectile_loss(
    adv: jnp.ndarray, diff: jnp.ndarray, expectile: float = 0.9
) -> jnp.ndarray:
  weight = jnp.where(adv >= 0, expectile, (1 - expectile))
  return weight * diff ** 2


@jax.tree_util.register_pytree_node_class
class DotCost(costs.CostFn):
  """The cost function that is used in neural OT training with 
  scalar product.
  """

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Dot product of vectors."""
    return jnp.dot(x, y)

  @classmethod
  def _padder(cls, dim: int) -> jnp.ndarray:
    return jnp.ones((1, dim))


class PotentialTrainState(train_state.TrainState):
  r"""Adds information about the model's value and gradient to the state.
  The gradient may differ from composition of jax.grad and value functions.

  Args:
  potential_value_fn: the potential's value function.
  potential_gradient_fn: the potential's gradient function.
  """

  potential_value_fn: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                               PotentialValueFn_t] = struct.field(
                                   pytree_node=False
                               )

  potential_gradient_fn: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                                  PotentialGradientFn_t] = struct.field(
                                      pytree_node=False
                                  )


class PotentialModelWrapper(nn.Module):
  """Wrapper class for the neural models that implement a potential value 
  or a vector field."""

  is_potential: bool  # The module implements a potential value or a vector field.
  add_l2_norm: bool   # If true, squared Euclidean norm is added to the potential.
  model: nn.Module    # The potential model.

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
  """A simple MLP model of a potential or a vector field used in default 
  initialization."""

  dim_hidden: Sequence[int]
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply MLP transform."""
    for feat in self.dim_hidden[:-1]:
      x = self.act_fn(nn.Dense(feat)(x))
    return nn.Dense(self.dim_hidden[-1])(x)


class ExpectileNeuralDual(Generic[CostFn_t]):
  r"""Expectile-regularized Neural Optimal Transport (ENOT) 
  method :cite:`buzun:24`.

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
  directions.
  Inspired from :cite:`jacobs:20`.
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
    self.use_dot_product = (type(cost_fn) == DotCost)

    assert (
      isinstance(cost_fn, (costs.TICost, DotCost)) or not is_bidirectional
    ), (
      "is_bidirectional=True can only be used with a translation invariant" 
      "cost (TICost) or DotCost."
    )

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

  def _get_train_step(self) -> Callable[
      [PotentialTrainState, PotentialTrainState, Dict[str, jnp.ndarray]], Tuple[
          PotentialTrainState, PotentialTrainState, jnp.ndarray, jnp.ndarray,
          jnp.ndarray, jnp.ndarray]]:
    loss_fn = self._euclidean_loss_fn if self.use_dot_product else self._loss_fn

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

  def _get_valid_step(self) -> Callable[
      [PotentialTrainState, PotentialTrainState, Dict[str, jnp.ndarray]], Tuple[
          jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    loss_fn = self._euclidean_loss_fn if self.use_dot_product else self._loss_fn

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

  def _loss_fn(
      self, params_f: frozen_dict.FrozenDict[str, jnp.ndarray],
      params_g: frozen_dict.FrozenDict[str, jnp.ndarray],
      gradient_f: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                           PotentialGradientFn_t],
      g_value: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                        PotentialValueFn_t], batch: Dict[str, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    if self.is_bidirectional:
      grad_h_inv = jax.vmap(jax.grad(self.cost_fn.h_legendre))
      transport = lambda grad_f, source: source - grad_h_inv(grad_f(source))
    else:
      transport = lambda grad_f, source: grad_f(source)

    def g_c_transform(x, y):
      return batch_cost(x,y) \ 
      - jax.vmap(g_value(jax.lax.stop_gradient(params_g)))(y)

    source, target = batch["source"], batch["target"]

    f_grad_partial = jax.vmap(gradient_f(params_f))
    g_value_partial = jax.vmap(g_value(params_g))
    batch_cost = jax.vmap(self.cost_fn)

    target_hat = transport(f_grad_partial, source)
    target_hat_detach = jax.lax.stop_gradient(target_hat)

    g_target = g_value_partial(target)
    g_star_source = batch_cost(source, target_hat_detach
                              ) - g_value_partial(target_hat_detach)

    diff_1 = jax.lax.stop_gradient(
        -batch_cost(source, target) + g_c_transform(source, target_hat_detach)
    ) + g_target
    reg_loss_1 = expectile_loss(diff_1, diff_1, self.expectile).mean()

    diff_2 = jax.lax.stop_gradient(
        -g_c_transform(source, target)
    ) + g_star_source
    reg_loss_2 = expectile_loss(diff_2, diff_2, self.expectile).mean()

    reg_loss = (reg_loss_1 + reg_loss_2) * self.expectile_loss_coef
    dual_loss = -(g_target.mean() + g_star_source.mean())
    amor_loss = g_c_transform(source, target_hat).mean()

    loss = reg_loss + dual_loss + amor_loss

    w_dist = (g_target.mean() + g_star_source.mean())

    return loss, (dual_loss, amor_loss, w_dist)

  def _euclidean_loss_fn(
      self, params_f: frozen_dict.FrozenDict[str, jnp.ndarray],
      params_g: frozen_dict.FrozenDict[str, jnp.ndarray],
      gradient_f: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                           PotentialGradientFn_t],
      g_value: Callable[[frozen_dict.FrozenDict[str, jnp.ndarray]],
                        PotentialValueFn_t], batch: Dict[str, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    source, target = batch["source"], batch["target"]

    g_value_partial = jax.vmap(g_value(params_g))
    batch_dot = jax.vmap(jnp.dot)

    target_hat = jax.vmap(gradient_f(params_f))(source)
    target_hat_detach = jax.lax.stop_gradient(target_hat)

    g_target = g_value_partial(target)
    g_star_source = batch_dot(source, target_hat_detach
                             ) - g_value_partial(target_hat_detach)

    def g_c_transform(x, y):
      return -batch_dot(x, y) \
        + jax.vmap(g_value(jax.lax.stop_gradient(params_g)))(y)

    diff_1 = jax.lax.stop_gradient(
        batch_dot(source, target) + g_c_transform(source, target_hat_detach)
    ) - g_target
    reg_loss_1 = expectile_loss(diff_1, diff_1, self.expectile).mean()

    diff_2 = jax.lax.stop_gradient(
        -g_c_transform(source, target)
    ) - g_star_source
    reg_loss_2 = expectile_loss(diff_2, diff_2, self.expectile).mean()

    reg_loss = (reg_loss_1 + reg_loss_2) * self.expectile_loss_coef
    dual_loss = g_target.mean() + g_star_source.mean()
    amor_loss = g_c_transform(source, target_hat).mean()

    loss = reg_loss + dual_loss + amor_loss

    C = jnp.mean(jnp.sum(source ** 2, axis=-1)) \ 
      + jnp.mean(jnp.sum(target ** 2, axis=-1))

    w2_dist = C - 2. * (g_target.mean() + g_star_source.mean())

    return loss, (dual_loss, amor_loss, w2_dist)

  def to_dual_potentials(self) -> DualPotentials:
    """Return the Kantorovich dual potentials from the trained potentials."""

    f_grad_partial = self.state_f.potential_gradient_fn(self.state_f.params)
    g_value_partial = self.state_g.potential_value_fn(self.state_g.params)

    if self.use_dot_product:
      return EuclideanPotentials(g_value_partial, f_grad_partial)
    if self.is_bidirectional:
      return BidirectionalPotentials(
          g_value_partial, f_grad_partial, self.cost_fn
      )
    return ForwardPotentials(g_value_partial, f_grad_partial, self.cost_fn)

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
