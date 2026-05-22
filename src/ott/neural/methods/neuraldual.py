# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
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

import optax
from flax import nnx

from ott import utils
from ott.geometry import costs
from ott.neural.networks import icnn, potentials
from ott.neural.networks.layers import conjugate
from ott.problems.linear import potentials as dual_potentials

__all__ = ["W2NeuralDual"]

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable[[int, dual_potentials.DualPotentials], None]

PotentialValueFn_t = potentials.PotentialValueFn_t
PotentialGradientFn_t = potentials.PotentialGradientFn_t


def _value_fn(
    model: nnx.Module,
    other_value_fn: Optional[Callable] = None,
) -> PotentialValueFn_t:
  """Get a scalar value function from an NNX model.

  For potential models (``is_potential=True``), returns ``model(x)``.
  For gradient models, reconstructs via the envelope theorem.
  """
  if model.is_potential:
    return lambda x: model(x)

  assert other_value_fn is not None, (
      "The value of a gradient-based potential depends on the other potential."
  )

  def value_fn(x: jnp.ndarray) -> jnp.ndarray:
    squeeze = x.ndim == 1
    if squeeze:
      x = jnp.expand_dims(x, 0)
    grad_g_x = jax.lax.stop_gradient(model(x))
    value = -other_value_fn(grad_g_x) + jax.vmap(jnp.dot)(grad_g_x, x)
    return value.squeeze(0) if squeeze else value

  return value_fn


def _gradient_fn(model: nnx.Module) -> PotentialGradientFn_t:
  """Get a gradient function from an NNX model.

  For potential models, returns ``vmap(grad(model))``.
  For gradient models, returns ``model`` directly.
  """
  if model.is_potential:
    return jax.vmap(jax.grad(lambda x: model(x)))
  return lambda x: model(x)


class W2NeuralDual:
  r"""Solver for the Wasserstein-2 Kantorovich dual between Euclidean spaces.

  Learn the Wasserstein-2 optimal transport between two measures
  :math:`\alpha` and :math:`\beta` in :math:`n`-dimensional Euclidean space,
  denoted source and target, respectively. This is achieved by parameterizing
  a Kantorovich potential :math:`f_\theta: \mathbb{R}^n\rightarrow\mathbb{R}`
  associated with the :math:`\alpha` measure with an
  :class:`~ott.neural.networks.icnn.ICNN` or a
  :class:`~ott.neural.networks.potentials.PotentialMLP`, where
  :math:`\nabla f` transports source to target cells. This potential is learned
  by optimizing the dual form associated with the negative inner product cost

  .. math::

    \text{argsup}_{\theta}\; -\mathbb{E}_{x\sim\alpha}[f_\theta(x)] -
    \mathbb{E}_{y\sim\beta}[f^\star_\theta(y)],

  where :math:`f^\star(y) := -\inf_{x\in\mathbb{R}^n} f(x)-\langle x, y\rangle`
  is the convex conjugate.
  :math:`\nabla f^\star` transports from the target
  to source cells and provides the inverse optimal
  transport map from :math:`\beta` to :math:`\alpha`.
  This solver estimates the conjugate :math:`f^\star`
  with a neural approximation :math:`g` that is fine-tuned
  with :class:`~ott.neural.networks.layers.conjugate.FenchelConjugateSolver`,
  which is a combination further described in :cite:`amos:23`.

  The potentials for ``neural_f`` and ``neural_g`` can

  1. both provide the values of the potentials :math:`f` and :math:`g`, or
  2. one of them can provide the gradient mapping e.g., :math:`\nabla f`
     or :math:`\nabla g` where the potential's value can be obtained
     via the Fenchel conjugate as discussed in :cite:`amos:23`.

  The potential's value or gradient mapping is specified via the
  ``is_potential`` property of the model.

  Args:
    dim_data: input dimensionality of data required for network init
    neural_f: NNX network for potential :math:`f`. Must expose an
      ``is_potential`` property.
    neural_g: NNX network for the conjugate potential
      :math:`g\approx f^\star`
    optimizer_f: optimizer function for potential :math:`f`
    optimizer_g: optimizer function for the conjugate potential :math:`g`
    num_train_iters: number of total training iterations
    num_inner_iters: number of training iterations of :math:`g` per iteration
      of :math:`f`
    back_and_forth: alternate between updating the forward and backward
      directions. Inspired from :cite:`jacobs:20`
    valid_freq: frequency with which model is validated
    log_freq: frequency with training and validation are logged
    logging: option to return logs
    rng: random key used for seeding for network initializations
    conjugate_solver: numerical solver for the Fenchel conjugate.
    amortization_loss: amortization loss for the conjugate
      :math:`g\approx f^\star`. Options are `'objective'` :cite:`makkuva:20` or
      `'regression'` :cite:`amos:23`.
    parallel_updates: Update :math:`f` and :math:`g` at the same time
  """

  def __init__(
      self,
      dim_data: int,
      neural_f: Optional[nnx.Module] = None,
      neural_g: Optional[nnx.Module] = None,
      optimizer_f: Optional[optax.OptState] = None,
      optimizer_g: Optional[optax.OptState] = None,
      num_train_iters: int = 20000,
      num_inner_iters: int = 1,
      back_and_forth: Optional[bool] = None,
      valid_freq: int = 1000,
      log_freq: int = 1000,
      logging: bool = False,
      rng: Optional[jax.Array] = None,
      conjugate_solver: Optional[conjugate.FenchelConjugateSolver
                                ] = conjugate.DEFAULT_CONJUGATE_SOLVER,
      amortization_loss: Literal["objective", "regression"] = "regression",
      parallel_updates: bool = True,
  ):
    self.num_train_iters = num_train_iters
    self.num_inner_iters = num_inner_iters
    self.back_and_forth = back_and_forth
    self.valid_freq = valid_freq
    self.log_freq = log_freq
    self.logging = logging
    self.parallel_updates = parallel_updates
    self.conjugate_solver = conjugate_solver
    self.amortization_loss = amortization_loss

    # set default optimizers
    if optimizer_f is None:
      optimizer_f = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)
    if optimizer_g is None:
      optimizer_g = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)

    # set default neural architectures
    rng = utils.default_prng_key(rng)
    rng, rng_init_f, rng_init_g = jax.random.split(rng, 3)
    if neural_f is None:
      neural_f = icnn.ICNN(
          input_dim=dim_data,
          dim_hidden=[64, 64, 64, 64],
          rngs=nnx.Rngs(rng_init_f),
      )
    if neural_g is None:
      neural_g = icnn.ICNN(
          input_dim=dim_data,
          dim_hidden=[64, 64, 64, 64],
          rngs=nnx.Rngs(rng_init_g),
      )
    self.neural_f = neural_f
    self.neural_g = neural_g

    # set optimizers and step functions
    self.setup(neural_f, neural_g, optimizer_f, optimizer_g)

  def setup(
      self,
      neural_f: nnx.Module,
      neural_g: nnx.Module,
      optimizer_f: optax.OptState,
      optimizer_g: optax.OptState,
  ) -> None:
    """Setup all components required to train the network."""
    self.opt_f = nnx.Optimizer(neural_f, optimizer_f, wrt=nnx.Param)
    self.opt_g = nnx.Optimizer(neural_g, optimizer_g, wrt=nnx.Param)

    # default to using back_and_forth with the non-convex models
    if self.back_and_forth is None:
      self.back_and_forth = isinstance(neural_f, potentials.PotentialMLP)

    if self.num_inner_iters == 1 and self.parallel_updates:
      self.train_step_parallel = self._get_parallel_step_fn(train=True)
      self.valid_step_parallel = self._get_parallel_step_fn(train=False)
      self.train_fn = self.train_neuraldual_parallel
    else:
      if self.parallel_updates:
        warnings.warn(
            "parallel_updates set to True but disabling it "
            "because num_inner_iters>1",
            stacklevel=2
        )
      if self.back_and_forth:
        raise NotImplementedError(
            "back_and_forth not implemented without parallel updates"
        )
      self.train_step_f = self._get_alternating_step_fn(
          train=True, to_optimize="f"
      )
      self.valid_step_f = self._get_alternating_step_fn(
          train=False, to_optimize="f"
      )
      self.train_step_g = self._get_alternating_step_fn(
          train=True, to_optimize="g"
      )
      self.valid_step_g = self._get_alternating_step_fn(
          train=False, to_optimize="g"
      )
      self.train_fn = self.train_neuraldual_alternating

  def __call__(  # noqa: D102
      self,
      trainloader_source: Iterator[jnp.ndarray],
      trainloader_target: Iterator[jnp.ndarray],
      validloader_source: Iterator[jnp.ndarray],
      validloader_target: Iterator[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Union[dual_potentials.DualPotentials,
             Tuple[dual_potentials.DualPotentials, Train_t]]:
    logs = self.train_fn(
        trainloader_source,
        trainloader_target,
        validloader_source,
        validloader_target,
        callback=callback,
    )
    res = self.to_dual_potentials()

    return (res, logs) if self.logging else res

  # ---- loss computation (shared by both parallel and alternating) ---------

  def _compute_losses(
      self,
      model_f: nnx.Module,
      model_g: nnx.Module,
      batch: Dict[str, jnp.ndarray],
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute all losses.

    Returns:
      ``(dual_loss, amor_loss, W2_dist)``
    """
    source, target = batch["source"], batch["target"]

    g_gradient = _gradient_fn(model_g)
    init_source_hat = g_gradient(target)

    def g_value_partial(y: jnp.ndarray) -> jnp.ndarray:
      return _value_fn(model_g)(y)

    f_value_partial = _value_fn(model_f, g_value_partial)

    if self.conjugate_solver is not None:
      finetune_source_hat = lambda y, x_init: self.conjugate_solver.solve(
          f_value_partial, y, x_init=x_init
      ).grad
      finetune_source_hat = jax.vmap(finetune_source_hat)
      source_hat_detach = jax.lax.stop_gradient(
          finetune_source_hat(target, init_source_hat)
      )
    else:
      source_hat_detach = init_source_hat

    batch_dot = jax.vmap(jnp.dot)

    f_source = f_value_partial(source)
    f_star_target = batch_dot(source_hat_detach,
                              target) - f_value_partial(source_hat_detach)
    dual_source = f_source.mean()
    dual_target = f_star_target.mean()
    dual_loss = dual_source + dual_target

    if self.amortization_loss == "regression":
      amor_loss = ((init_source_hat - source_hat_detach) ** 2).mean()
    elif self.amortization_loss == "objective":
      # Stop gradients through f's parameters only (not inputs)
      f_graphdef, f_state = nnx.split(model_f)
      f_state_stopped = jax.lax.stop_gradient(f_state)
      model_f_detached = nnx.merge(f_graphdef, f_state_stopped)
      f_value_detached = _value_fn(model_f_detached, g_value_partial)
      amor_loss = (
          f_value_detached(init_source_hat) -
          batch_dot(init_source_hat, target)
      ).mean()
    else:
      raise ValueError("Amortization loss has been misspecified.")

    # compute Wasserstein-2 distance
    C = jnp.mean(jnp.sum(source ** 2, axis=-1)) + \
        jnp.mean(jnp.sum(target ** 2, axis=-1))
    W2_dist = C - 2.0 * (f_source.mean() + f_star_target.mean())

    return dual_loss, amor_loss, W2_dist

  # ---- parallel step functions -------------------------------------------

  def _get_parallel_step_fn(self, train: bool):
    """Create parallel training/validation step function."""
    _diff_both = (nnx.DiffState(0, nnx.Param), nnx.DiffState(1, nnx.Param))

    @nnx.jit
    def train_step(model_f, model_g, opt_f, opt_g, batch):

      def loss_fn_both(model_f, model_g):
        dual_loss, amor_loss, _ = self._compute_losses(model_f, model_g, batch)
        return dual_loss + amor_loss

      # Differentiate w.r.t. both models
      loss, (grads_f, grads_g) = nnx.value_and_grad(
          loss_fn_both, argnums=_diff_both
      )(model_f, model_g)

      opt_f.update(model_f, grads_f)
      opt_g.update(model_g, grads_g)

      # Recompute individual losses for logging
      dual_loss, amor_loss, W2_dist = self._compute_losses(
          model_f, model_g, batch
      )
      return loss, dual_loss, amor_loss, W2_dist

    @nnx.jit
    def valid_step(model_f, model_g, batch):
      dual_loss, amor_loss, W2_dist = self._compute_losses(
          model_f, model_g, batch
      )
      return dual_loss, amor_loss, W2_dist

    return train_step if train else valid_step

  # ---- alternating step functions ----------------------------------------

  def _get_alternating_step_fn(
      self, train: bool, to_optimize: Literal["f", "g"]
  ):
    """Create alternating training/validation step function."""
    _diff_f = nnx.DiffState(0, nnx.Param)
    _diff_g = nnx.DiffState(1, nnx.Param)

    @nnx.jit
    def train_step_f(model_f, model_g, opt_f, batch):

      def loss_fn(model_f, model_g):
        dual_loss, _, _ = self._compute_losses(model_f, model_g, batch)
        return dual_loss

      grads = nnx.grad(loss_fn, argnums=_diff_f)(model_f, model_g)
      opt_f.update(model_f, grads)

      dual_loss, _, W2_dist = self._compute_losses(model_f, model_g, batch)
      return dual_loss, W2_dist

    @nnx.jit
    def train_step_g(model_f, model_g, opt_g, batch):

      def loss_fn(model_f, model_g):
        _, amor_loss, _ = self._compute_losses(model_f, model_g, batch)
        return amor_loss

      grads = nnx.grad(loss_fn, argnums=_diff_g)(model_f, model_g)
      opt_g.update(model_g, grads)

      _, amor_loss, W2_dist = self._compute_losses(model_f, model_g, batch)
      return amor_loss, W2_dist

    @nnx.jit
    def valid_step(model_f, model_g, batch):
      dual_loss, amor_loss, W2_dist = self._compute_losses(
          model_f, model_g, batch
      )
      if to_optimize == "f":
        return dual_loss, W2_dist
      return amor_loss, W2_dist

    if train:
      return train_step_f if to_optimize == "f" else train_step_g
    return valid_step

  # ---- training loops ----------------------------------------------------

  def train_neuraldual_parallel(
      self,
      trainloader_source: Iterator[jnp.ndarray],
      trainloader_target: Iterator[jnp.ndarray],
      validloader_source: Iterator[jnp.ndarray],
      validloader_target: Iterator[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Train_t:
    """Training and validation with parallel updates."""
    try:
      from tqdm.auto import tqdm
    except ImportError:
      tqdm = lambda _: _
    # define dict to contain source and target batch
    train_batch, valid_batch = {}, {}

    # set logging dictionaries
    train_logs = {"loss_f": [], "loss_g": [], "w_dist": [], "directions": []}
    valid_logs = {"loss_f": [], "loss_g": [], "w_dist": []}

    for step in tqdm(range(self.num_train_iters)):
      update_forward = not self.back_and_forth or step % 2 == 0
      if update_forward:
        train_batch["source"] = jnp.asarray(next(trainloader_source))
        train_batch["target"] = jnp.asarray(next(trainloader_target))
        (loss, loss_f, loss_g, w_dist) = self.train_step_parallel(
            self.neural_f,
            self.neural_g,
            self.opt_f,
            self.opt_g,
            train_batch,
        )
      else:
        train_batch["target"] = jnp.asarray(next(trainloader_source))
        train_batch["source"] = jnp.asarray(next(trainloader_target))
        (loss, loss_f, loss_g, w_dist) = self.train_step_parallel(
            self.neural_g,
            self.neural_f,
            self.opt_g,
            self.opt_f,
            train_batch,
        )

      if self.logging and step % self.log_freq == 0:
        self._update_logs(train_logs, loss_f, loss_g, w_dist)
        train_logs["directions"].append(
            "forward" if update_forward else "backward"
        )

      if callback is not None:
        _ = callback(step, self.to_dual_potentials())

      # report the loss on an validation dataset periodically
      if step != 0 and step % self.valid_freq == 0:
        # get batch
        valid_batch["source"] = jnp.asarray(next(validloader_source))
        valid_batch["target"] = jnp.asarray(next(validloader_target))

        valid_loss_f, valid_loss_g, valid_w_dist = self.valid_step_parallel(
            self.neural_f,
            self.neural_g,
            valid_batch,
        )

        if self.logging:
          self._update_logs(
              valid_logs, valid_loss_f, valid_loss_g, valid_w_dist
          )

    return {"train_logs": train_logs, "valid_logs": valid_logs}

  def train_neuraldual_alternating(
      self,
      trainloader_source: Iterator[jnp.ndarray],
      trainloader_target: Iterator[jnp.ndarray],
      validloader_source: Iterator[jnp.ndarray],
      validloader_target: Iterator[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Train_t:
    """Training and validation with alternating updates."""
    try:
      from tqdm.auto import tqdm
    except ImportError:
      tqdm = lambda _: _
    # define dict to contain source and target batch
    batch_g, batch_f, valid_batch = {}, {}, {}

    # set logging dictionaries
    train_logs = {"loss_f": [], "loss_g": [], "w_dist": []}
    valid_logs = {"loss_f": [], "loss_g": [], "w_dist": []}

    for step in tqdm(range(self.num_train_iters)):
      # execute training steps
      for _ in range(self.num_inner_iters):
        # get train batch for potential g
        batch_g["source"] = jnp.asarray(next(trainloader_source))
        batch_g["target"] = jnp.asarray(next(trainloader_target))

        loss_g, _ = self.train_step_g(
            self.neural_f, self.neural_g, self.opt_g, batch_g
        )

      # get train batch for potential f
      batch_f["source"] = jnp.asarray(next(trainloader_source))
      batch_f["target"] = jnp.asarray(next(trainloader_target))

      loss_f, w_dist = self.train_step_f(
          self.neural_f, self.neural_g, self.opt_f, batch_f
      )

      if callback is not None:
        callback(step, self.to_dual_potentials())

      if self.logging and step % self.log_freq == 0:
        self._update_logs(train_logs, loss_f, loss_g, w_dist)

      # report the loss on validation dataset periodically
      if step != 0 and step % self.valid_freq == 0:
        # get batch
        valid_batch["source"] = jnp.asarray(next(validloader_source))
        valid_batch["target"] = jnp.asarray(next(validloader_target))

        valid_loss_f, _ = self.valid_step_f(
            self.neural_f, self.neural_g, valid_batch
        )
        valid_loss_g, valid_w_dist = self.valid_step_g(
            self.neural_f, self.neural_g, valid_batch
        )

        if self.logging:
          self._update_logs(
              valid_logs, valid_loss_f, valid_loss_g, valid_w_dist
          )

    return {"train_logs": train_logs, "valid_logs": valid_logs}

  def to_dual_potentials(
      self, finetune_g: bool = True
  ) -> dual_potentials.DualPotentials:
    r"""Return the Kantorovich dual potentials from the trained potentials.

    Args:
      finetune_g: Run the conjugate solver to fine-tune the prediction.

    Returns:
      A dual potential object
    """
    f_value = _value_fn(self.neural_f)
    g_value_prediction = _value_fn(self.neural_g, f_value)

    def g_value_finetuned(y: jnp.ndarray) -> jnp.ndarray:
      x_hat = jax.grad(g_value_prediction)(y)
      grad_g_y = jax.lax.stop_gradient(
          self.conjugate_solver.solve(f_value, y, x_init=x_hat).grad
      )
      return -f_value(grad_g_y) + jnp.dot(grad_g_y, y)

    if not finetune_g or self.conjugate_solver is None:
      g_value = g_value_prediction
    else:
      g_value = g_value_finetuned

    # switch from grad-convex potentials to quadratic - convex parameterization
    return dual_potentials.DualPotentials(
        f=lambda x: 0.5 * jnp.sum(x ** 2) - f_value(x),
        g=lambda x: 0.5 * jnp.sum(x ** 2) - g_value(x),
        cost_fn=costs.SqEuclidean(),
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
