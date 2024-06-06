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

from ott import utils
from ott.geometry import costs
from ott.neural.networks import icnn, potentials
from ott.neural.networks.layers import conjugate
from ott.problems.linear import potentials as dual_potentials

__all__ = ["W2NeuralDual"]

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable[[int, dual_potentials.DualPotentials], None]


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

  The :class:`~ott.neural.networks.potentials.BasePotential` potentials for
  ``neural_f`` and ``neural_g`` can

  1. both provide the values of the potentials :math:`f` and :math:`g`, or
  2. one of them can provide the gradient mapping e.g., :math:`\nabla f`
     or :math:`\nabla g` where the potential's value can be obtained
     via the Fenchel conjugate as discussed in :cite:`amos:23`.

  The potential's value or gradient mapping is specified via
  :attr:`~ott.neural.networks.potentials.BasePotential.is_potential`.

  Args:
    dim_data: input dimensionality of data required for network init
    neural_f: network architecture for potential :math:`f`.
    neural_g: network architecture for the conjugate potential
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
    pos_weights: option to train networks with positive weights or regularizer
    beta: regularization parameter when not training with positive weights
    conjugate_solver: numerical solver for the Fenchel conjugate.
    amortization_loss: amortization loss for the conjugate
      :math:`g\approx f^\star`. Options are `'objective'` :cite:`makkuva:20` or
      `'regression'` :cite:`amos:23`.
    parallel_updates: Update :math:`f` and :math:`g` at the same time
  """

  def __init__(
      self,
      dim_data: int,
      neural_f: Optional[potentials.BasePotential] = None,
      neural_g: Optional[potentials.BasePotential] = None,
      optimizer_f: Optional[optax.OptState] = None,
      optimizer_g: Optional[optax.OptState] = None,
      num_train_iters: int = 20000,
      num_inner_iters: int = 1,
      back_and_forth: Optional[bool] = None,
      valid_freq: int = 1000,
      log_freq: int = 1000,
      logging: bool = False,
      rng: Optional[jax.Array] = None,
      pos_weights: bool = True,
      beta: float = 1.0,
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
    self.pos_weights = pos_weights
    self.beta = beta
    self.parallel_updates = parallel_updates
    self.conjugate_solver = conjugate_solver
    self.amortization_loss = amortization_loss

    # set default optimizers
    if optimizer_f is None:
      optimizer_f = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)
    if optimizer_g is None:
      optimizer_g = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)

    # set default neural architectures
    if neural_f is None:
      neural_f = icnn.ICNN(dim_data=dim_data, dim_hidden=[64, 64, 64, 64])
    if neural_g is None:
      neural_g = icnn.ICNN(dim_data=dim_data, dim_hidden=[64, 64, 64, 64])
    self.neural_f = neural_f
    self.neural_g = neural_g

    # set optimizer and networks
    self.setup(
        utils.default_prng_key(rng),
        neural_f,
        neural_g,
        dim_data,
        optimizer_f,
        optimizer_g,
    )

  def setup(
      self,
      rng: jax.Array,
      neural_f: potentials.BasePotential,
      neural_g: potentials.BasePotential,
      dim_data: int,
      optimizer_f: optax.OptState,
      optimizer_g: optax.OptState,
  ) -> None:
    """Setup all components required to train the network."""
    # split random number generator
    rng, rng_f, rng_g = jax.random.split(rng, 3)

    # check setting of network architectures
    warn_str = f"Setting of ICNN and the positive weights setting of the " \
        f"`W2NeuralDual` are not consistent. Proceeding with " \
        f"the `W2NeuralDual` setting, with positive weights " \
        f"being {self.pos_weights}."
    if isinstance(
        neural_f, icnn.ICNN
    ) and neural_f.pos_weights is not self.pos_weights:
      warnings.warn(warn_str, stacklevel=2)
      neural_f.pos_weights = self.pos_weights

    if isinstance(
        neural_g, icnn.ICNN
    ) and neural_g.pos_weights is not self.pos_weights:
      warnings.warn(warn_str, stacklevel=2)
      neural_g.pos_weights = self.pos_weights

    self.state_f = neural_f.create_train_state(
        rng_f,
        optimizer_f,
        (1, dim_data),  # also include the batch dimension
    )
    self.state_g = neural_g.create_train_state(
        rng_g,
        optimizer_g,
        (1, dim_data),
    )

    # default to using back_and_forth with the non-convex models
    if self.back_and_forth is None:
      self.back_and_forth = isinstance(neural_f, potentials.PotentialMLP)

    if self.num_inner_iters == 1 and self.parallel_updates:
      self.train_step_parallel = self.get_step_fn(
          train=True, to_optimize="both"
      )
      self.valid_step_parallel = self.get_step_fn(
          train=False, to_optimize="both"
      )
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
      self.train_step_f = self.get_step_fn(train=True, to_optimize="f")
      self.valid_step_f = self.get_step_fn(train=False, to_optimize="f")
      self.train_step_g = self.get_step_fn(train=True, to_optimize="g")
      self.valid_step_g = self.get_step_fn(train=False, to_optimize="g")
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
        (self.state_f, self.state_g, loss, loss_f, loss_g,
         w_dist) = self.train_step_parallel(
             self.state_f,
             self.state_g,
             train_batch,
         )
      else:
        train_batch["target"] = jnp.asarray(next(trainloader_source))
        train_batch["source"] = jnp.asarray(next(trainloader_target))
        (self.state_g, self.state_f, loss, loss_f, loss_g,
         w_dist) = self.train_step_parallel(
             self.state_g,
             self.state_f,
             train_batch,
         )

      if self.logging and step % self.log_freq == 0:
        self._update_logs(train_logs, loss_f, loss_g, w_dist)
        train_logs["directions"].append(
            "forward" if update_forward else "backward"
        )

      if callback is not None:
        _ = callback(step, self.to_dual_potentials())

      if not self.pos_weights:
        # Only clip the weights of the f network
        self.state_f = self.state_f.replace(
            params=self._clip_weights_icnn(self.state_f.params)
        )

      # report the loss on an validation dataset periodically
      if step != 0 and step % self.valid_freq == 0:
        # get batch
        valid_batch["source"] = jnp.asarray(next(validloader_source))
        valid_batch["target"] = jnp.asarray(next(validloader_target))

        valid_loss_f, valid_loss_g, valid_w_dist = self.valid_step_parallel(
            self.state_f,
            self.state_g,
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

        self.state_g, loss_g, _ = self.train_step_g(
            self.state_f, self.state_g, batch_g
        )

      # get train batch for potential f
      batch_f["source"] = jnp.asarray(next(trainloader_source))
      batch_f["target"] = jnp.asarray(next(trainloader_target))

      self.state_f, loss_f, w_dist = self.train_step_f(
          self.state_f, self.state_g, batch_f
      )
      if not self.pos_weights:
        # Only clip the weights of the f network
        self.state_f = self.state_f.replace(
            params=self._clip_weights_icnn(self.state_f.params)
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
            self.state_f, self.state_g, valid_batch
        )
        valid_loss_g, valid_w_dist = self.valid_step_g(
            self.state_f, self.state_g, valid_batch
        )

        if self.logging:
          self._update_logs(
              valid_logs, valid_loss_f, valid_loss_g, valid_w_dist
          )

    return {"train_logs": train_logs, "valid_logs": valid_logs}

  def get_step_fn(
      self, train: bool, to_optimize: Literal["f", "g", "parallel", "both"]
  ):
    """Create a parallel training and evaluation function."""

    def loss_fn(params_f, params_g, f_value, g_value, g_gradient, batch):
      """Loss function for both potentials."""
      # get two distributions
      source, target = batch["source"], batch["target"]

      init_source_hat = g_gradient(params_g)(target)

      def g_value_partial(y: jnp.ndarray) -> jnp.ndarray:
        """Lazy way of evaluating g if f's computation needs it."""
        return g_value(params_g)(y)

      f_value_partial = f_value(params_f, g_value_partial)
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
        f_value_parameters_detached = f_value(
            jax.lax.stop_gradient(params_f), g_value_partial
        )
        amor_loss = (
            f_value_parameters_detached(init_source_hat) -
            batch_dot(init_source_hat, target)
        ).mean()
      else:
        raise ValueError("Amortization loss has been misspecified.")

      if to_optimize == "both":
        loss = dual_loss + amor_loss
      elif to_optimize == "f":
        loss = dual_loss
      elif to_optimize == "g":
        loss = amor_loss
      else:
        raise ValueError(
            f"Optimization target {to_optimize} has been misspecified."
        )

      if not self.pos_weights:
        # Penalize the weights of both networks, even though one
        # of them will be exactly clipped.
        # Having both here is necessary in case this is being called with
        # the potentials reversed with the back_and_forth.
        loss += self.beta * self._penalize_weights_icnn(params_f) + \
            self.beta * self._penalize_weights_icnn(params_g)

      # compute Wasserstein-2 distance
      C = jnp.mean(jnp.sum(source ** 2, axis=-1)) + \
          jnp.mean(jnp.sum(target ** 2, axis=-1))
      W2_dist = C - 2.0 * (f_source.mean() + f_star_target.mean())

      return loss, (dual_loss, amor_loss, W2_dist)

    @jax.jit
    def step_fn(state_f, state_g, batch):
      """Step function of either training or validation."""
      grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
      if train:
        # compute loss and gradients
        (loss, (loss_f, loss_g, W2_dist)), (grads_f, grads_g) = grad_fn(
            state_f.params,
            state_g.params,
            state_f.potential_value_fn,
            state_g.potential_value_fn,
            state_g.potential_gradient_fn,
            batch,
        )
        # update state
        if to_optimize == "both":
          return (
              state_f.apply_gradients(grads=grads_f),
              state_g.apply_gradients(grads=grads_g), loss, loss_f, loss_g,
              W2_dist
          )
        if to_optimize == "f":
          return state_f.apply_gradients(grads=grads_f), loss_f, W2_dist
        if to_optimize == "g":
          return state_g.apply_gradients(grads=grads_g), loss_g, W2_dist
        raise ValueError("Optimization target has been misspecified.")

      # compute loss and gradients
      (loss, (loss_f, loss_g, W2_dist)), _ = grad_fn(
          state_f.params,
          state_g.params,
          state_f.potential_value_fn,
          state_g.potential_value_fn,
          state_g.potential_gradient_fn,
          batch,
      )

      # do not update state
      if to_optimize == "both":
        return loss_f, loss_g, W2_dist
      if to_optimize == "f":
        return loss_f, W2_dist
      if to_optimize == "g":
        return loss_g, W2_dist
      raise ValueError("Optimization target has been misspecified.")

    return step_fn

  def to_dual_potentials(
      self, finetune_g: bool = True
  ) -> dual_potentials.DualPotentials:
    r"""Return the Kantorovich dual potentials from the trained potentials.

    Args:
      finetune_g: Run the conjugate solver to fine-tune the prediction.

    Returns:
      A dual potential object
    """
    f_value = self.state_f.potential_value_fn(self.state_f.params)
    g_value_prediction = self.state_g.potential_value_fn(
        self.state_g.params, f_value
    )

    def g_value_finetuned(y: jnp.ndarray) -> jnp.ndarray:
      x_hat = jax.grad(g_value_prediction)(y)
      grad_g_y = jax.lax.stop_gradient(
          self.conjugate_solver.solve(f_value, y, x_init=x_hat).grad
      )
      return -f_value(grad_g_y) + jnp.dot(grad_g_y, y)

    return dual_potentials.DualPotentials(
        f=f_value,
        g=g_value_prediction if not finetune_g or self.conjugate_solver is None
        else g_value_finetuned,
        cost_fn=costs.SqEuclidean(),
        corr=True
    )

  @staticmethod
  def _clip_weights_icnn(params):
    for k in params:
      if k.startswith("w_z"):
        params[k]["kernel"] = jnp.clip(params[k]["kernel"], 0.0)

    return params

  @staticmethod
  def _penalize_weights_icnn(params: Dict[str, jnp.ndarray]) -> float:
    penalty = 0.0
    for k, param in params.items():
      if k.startswith("w_z"):
        penalty += jnp.linalg.norm(jax.nn.relu(-param["kernel"]))
    return penalty

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
