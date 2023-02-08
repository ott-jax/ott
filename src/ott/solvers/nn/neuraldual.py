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
"""A Jax implementation of the ICNN based Kantorovich dual."""

import warnings
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core
from flax.core.frozen_dict import FrozenDict
from jax.lax import stop_gradient

from ott.geometry import costs
from ott.problems.linear import potentials
from ott.solvers.nn import icnn
from ott.solvers.nn.conjugate_solver import ConjugateSolver, FenchelConjugateLBFGS

__all__ = ["W2NeuralDual"]

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable[[int, potentials.DualPotentials], None]

DEFAULT_CONJUGATE_SOLVER = FenchelConjugateLBFGS(
    gtol=1e-5,
    max_iter=20,
    max_linesearch_iter=20,
    linesearch_type='backtracking',
)


class W2NeuralDual:
  r"""Solver for the Wasserstein-2 Kantorovich dual between Euclidean spaces.

  Learn the Wasserstein-2 optimal transport between two measures
  :math:`\alpha` and :math:`\beta` in
  :math:`n`-dimensional Euclidean space,
  denoted source and target, respectively.
  This is achieved by parameterizing a Kantorovich potential
  :math:`f_\theta: \mathbb{R}^n\rightarrow\mathbb{R}`
  associated with the :math:`\alpha` measure with
  an input-convex neural network or MLP where
  :math:`\nabla f` transports source to target cells.
  This potential is learned by optimizing the dual
  form associated with the negative inner product cost

  .. math::

    \argsup_{\theta}\; -\E_{x\sim\alpha}[f_\theta(x)] -
      \E_{y\sim\beta}[f^\star_\theta(y)]`,

  where
  :math:`f^\star(y) := -\inf_{x\in\gX} f(x)-\langle x, y\rangle`
  is the convex conjugate.
  :math:`\nabla f^\star` transports from the target
  to source cells.

  TODO(bamos): Describe how :math:`g` approximates the convex conjugate :math:`f^\star(y):
  and how :cite:`makkuva:20` and :cite:`amos:22a` learn it.

  TODO(bamos): Describe the conjugate solver.

  TODO(bamos): Decide on defaults here (conjugate solver, num_inner_iters,
  amortization_loss, parallel_updates)

  Args:
    input_dim: input dimensionality of data required for network init
    neural_f: network architecture for potential :math:`f`
    neural_g: network architecture for the conjugate potential :math:`g\approx f^\star`
    optimizer_f: optimizer function for potential :math:`f`
    optimizer_g: optimizer function for the conjugate potential :math:`g`
    num_train_iters: number of total training iterations
    num_inner_iters: number of training iterations of :math:`g` per iteration of :math:`f`
    back_and_forth: alternative between updating the forward and backward directions.
      Inspired from from :cite:`jacobs2020fast`
    valid_freq: frequency with which model is validated
    log_freq: frequency with training and validation are logged
    logging: option to return logs
    seed: random seed for network initializations
    pos_weights: option to train networks with positive weights or regularizer
    beta: regularization parameter when not training with positive weights
    conjugate_solver: numerical solver for the Fenchel conjugate.
    amortization_loss: amortization loss for the conjugate :math:`g\approx f^\star`.
      Options are 'objective' :cite:`makkuva:20` or 'regression' :cite:`amos:22a`.
    parallel_updates: Update :math:`f` and :math`g` at the same time
  """

  def __init__(
      self,
      input_dim: int,
      neural_f: Optional[nn.Module] = None,
      neural_g: Optional[nn.Module] = None,
      optimizer_f: Optional[optax.OptState] = None,
      optimizer_g: Optional[optax.OptState] = None,
      num_train_iters: int = 50000,
      num_inner_iters: int = 1,
      back_and_forth: bool = True,
      valid_freq: int = 1000,
      log_freq: int = 1000,
      logging: bool = False,
      seed: int = 0,
      pos_weights: bool = True,
      beta: float = 1.0,
      conjugate_solver: Optional[ConjugateSolver] = DEFAULT_CONJUGATE_SOLVER,
      amortization_loss: Literal['objective', 'regression'] = 'regression',
      parallel_updates: bool = True,
      init_f_params: Optional[FrozenDict] = None,
      init_g_params: Optional[FrozenDict] = None,
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

    # set random key
    rng = jax.random.PRNGKey(seed)

    # set default optimizers
    if optimizer_f is None:
      optimizer_f = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)
    if optimizer_g is None:
      optimizer_g = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)

    # set default neural architectures
    if neural_f is None:
      neural_f = icnn.ICNN(dim_data=input_dim, dim_hidden=[64, 64, 64, 64])
    if neural_g is None:
      neural_g = icnn.ICNN(dim_data=input_dim, dim_hidden=[64, 64, 64, 64])

    # set optimizer and networks
    self.setup(
        rng, neural_f, neural_g, input_dim, optimizer_f, optimizer_g,
        init_f_params, init_g_params
    )

  def setup(
      self, rng: jnp.ndarray, neural_f: icnn.ICNN, neural_g: icnn.ICNN,
      input_dim: int, optimizer_f: optax.OptState, optimizer_g: optax.OptState,
      init_f_params: Optional[FrozenDict], init_g_params: Optional[FrozenDict]
  ) -> None:
    """Setup all components required to train the network."""
    # split random key
    rng, rng_f, rng_g = jax.random.split(rng, 3)

    # check setting of network architectures
    warn_str = f"Setting of ICNN and the positive weights setting of the " \
        f"`W2NeuralDual` are not consistent. Proceeding with " \
        f"the `W2NeuralDual` setting, with positive weights " \
        f"being {self.pos_weights}."
    if isinstance(
        neural_f, icnn.ICNN
    ) and neural_f.pos_weights is not self.pos_weights:
      warnings.warn(warn_str)
      neural_f.pos_weights = self.pos_weights

    if isinstance(
        neural_g, icnn.ICNN
    ) and neural_g.pos_weights is not self.pos_weights:
      warnings.warn(warn_str)
      neural_g.pos_weights = self.pos_weights

    self.state_f = neural_f.create_train_state(
        rng_f, optimizer_f, input_dim, init_f_params
    )
    self.state_g = neural_g.create_train_state(
        rng_g, optimizer_g, input_dim, init_g_params
    )

    # Assume g defines the potential unless this attribute is set.
    self.g_returns_potential = getattr(neural_g, 'returns_potential', True)

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
            'parallel_updates set to True but disabling it because num_inner_iters>1'
        )
      self.train_step_f = self.get_step_fn(train=True, to_optimize="f")
      self.valid_step_f = self.get_step_fn(train=False, to_optimize="f")
      self.train_step_g = self.get_step_fn(train=True, to_optimize="g")
      self.valid_step_g = self.get_step_fn(train=False, to_optimize="g")
      self.train_fn = self.train_neuraldual_alternating

  def __call__(
      self,
      trainloader_source: Iterable[jnp.ndarray],
      trainloader_target: Iterable[jnp.ndarray],
      validloader_source: Iterable[jnp.ndarray],
      validloader_target: Iterable[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Union[potentials.DualPotentials, Tuple[potentials.DualPotentials,
                                              Train_t]]:
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
      trainloader_source: Iterable[jnp.ndarray],
      trainloader_target: Iterable[jnp.ndarray],
      validloader_source: Iterable[jnp.ndarray],
      validloader_target: Iterable[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Train_t:
    """Implementation of the training and validation with parallel updates."""  # noqa: D401
    try:
      from tqdm.auto import tqdm
    except ImportError:
      tqdm = lambda _: _
    # define dict to contain source and target batch
    train_batch = {}
    valid_batch = {}

    # set logging dictionaries
    train_logs = {"train_loss_f": [], "train_loss_g": [], "train_w_dist": []}
    valid_logs = {"valid_loss_f": [], "valid_loss_g": [], "valid_w_dist": []}

    for step in tqdm(range(self.num_train_iters)):
      if not self.back_and_forth or step % 2 == 0:
        # Update the forward direction
        train_batch["source"] = jnp.asarray(next(trainloader_source))
        train_batch["target"] = jnp.asarray(next(trainloader_target))
        self.state_f, self.state_g, loss, loss_f, loss_g, w_dist = self.train_step_parallel(
            self.state_f,
            self.state_g,
            train_batch,
        )
      else:
        # Update the backward direction
        train_batch["target"] = jnp.asarray(next(trainloader_source))
        train_batch["source"] = jnp.asarray(next(trainloader_target))
        self.state_g, self.state_f, loss, loss_f, loss_g, w_dist = self.train_step_parallel(
            self.state_g,
            self.state_f,
            train_batch,
        )

      if callback is not None:
        _ = callback(step, self.to_dual_potentials())

      if not self.pos_weights:
        self.state_f = self.state_f.replace(
            params=self._clip_weights_icnn(self.state_f.params)
        )

      if self.logging and step % self.log_freq == 0:
        train_logs["train_loss_f"].append(float(loss_f))
        train_logs["train_loss_g"].append(float(loss_g))
        train_logs["train_w_dist"].append(float(w_dist))

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
          # log training progress
          valid_logs["valid_loss_f"].append(float(valid_loss_f))
          valid_logs["valid_loss_g"].append(float(valid_loss_g))
          valid_logs["valid_w_dist"].append(float(valid_w_dist))

    return {"train_logs": train_logs, "valid_logs": valid_logs}

  def train_neuraldual_alternating(
      self,
      trainloader_source: Iterable[jnp.ndarray],
      trainloader_target: Iterable[jnp.ndarray],
      validloader_source: Iterable[jnp.ndarray],
      validloader_target: Iterable[jnp.ndarray],
      callback: Optional[Callback_t] = None,
  ) -> Train_t:
    """Implementation of the training and validation with alternating updates."""  # noqa: D401
    try:
      from tqdm.auto import tqdm
    except ImportError:
      tqdm = lambda _: _
    # define dict to contain source and target batch
    batch_g = {}
    batch_f = {}
    valid_batch = {}

    # set logging dictionaries
    train_logs = {"train_loss_f": [], "train_loss_g": [], "train_w_dist": []}
    valid_logs = {"valid_loss_f": [], "valid_loss_g": [], "valid_w_dist": []}

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
        self.state_f = self.state_f.replace(
            params=self._clip_weights_icnn(self.state_f.params)
        )

      if callback is not None:
        callback(step, self.to_dual_potentials())

      if self.logging and step % self.log_freq == 0:
        train_logs["train_loss_f"].append(float(loss_f))
        train_logs["train_loss_g"].append(float(loss_g))
        train_logs["train_w_dist"].append(float(w_dist))

      # report the loss on an validuation dataset periodically
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
          # log training progress
          valid_logs["valid_loss_f"].append(float(valid_loss_f))
          valid_logs["valid_loss_g"].append(float(valid_loss_g))
          valid_logs["valid_w_dist"].append(float(valid_w_dist))

    return {"train_logs": train_logs, "valid_logs": valid_logs}

  def get_step_fn(
      self, train: bool, to_optimize: Literal["f", "g", "parallel"]
  ):
    """Create a parallel training and evaluation function."""

    def loss_fn(params_f, params_g, f, g, batch):
      """Loss function for both potentials."""
      # get two distributions
      source, target = batch["source"], batch["target"]

      if self.g_returns_potential:
        g_grad = jax.vmap(
            lambda y: jax.grad(g, argnums=1)({
                "params": params_g
            }, y)
        )
        init_source_hat = g_grad(target)
      else:
        init_source_hat = g({'params': params_g}, target)

      f_apply = lambda x: f({'params': params_f}, x)
      if self.conjugate_solver is not None:
        finetune_source_hat = lambda y, x_init: self.conjugate_solver.solve(
            f_apply, y, x_init=x_init
        ).grad
        finetune_source_hat = jax.vmap(finetune_source_hat)
        source_hat_detach = stop_gradient(
            finetune_source_hat(target, init_source_hat)
        )
      else:
        source_hat_detach = init_source_hat

      batch_dot = jax.vmap(jnp.dot)

      f_source = f_apply(source)
      f_star_target = batch_dot(source_hat_detach,
                                target) - f_apply(source_hat_detach)
      dual_source = f_source.mean()
      dual_target = f_star_target.mean()
      dual_loss = dual_source + dual_target

      if self.amortization_loss == 'regression':
        amor_loss = ((init_source_hat - source_hat_detach) ** 2).mean()
      elif self.amortization_loss == 'objective':
        f_apply_parameters_detached = lambda x: f({
            'params': stop_gradient(params_f)
        }, x)
        amor_loss = (
            f_apply_parameters_detached(init_source_hat) -
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

      # compute final wasserstein distance
      dist = -1.  # TODO(bamos): Add back

      return loss, (dual_loss, amor_loss, dist)

    @jax.jit
    def step_fn(state_f, state_g, batch):
      """Step function of either training or validation."""
      grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
      if train:
        # compute loss and gradients
        (loss, (loss_f, loss_g, dist)), (grads_f, grads_g) = grad_fn(
            state_f.params,
            state_g.params,
            state_f.apply_fn,
            state_g.apply_fn,
            batch,
        )

        # update state
        if to_optimize == "both":
          return state_f.apply_gradients(grads=grads_f), \
              state_g.apply_gradients(grads=grads_g), \
              loss, loss_f, loss_g, dist
        elif to_optimize == "f":
          return state_f.apply_gradients(grads=grads_f), \
              loss_f, dist
        elif to_optimize == "g":
          return state_g.apply_gradients(grads=grads_g), \
              loss_g, dist
        else:
          raise ValueError("Optimization target has been misspecified.")

      else:
        # compute loss and gradients
        (loss, (loss_f, loss_g, dist)), (grads_f, grads_g) = grad_fn(
            state_f.params,
            state_g.params,
            state_f.apply_fn,
            state_g.apply_fn,
            batch,
        )

        # do not update state
        if to_optimize == "both":
          return loss_f, loss_g, dist
        elif to_optimize == "f":
          return loss_f, dist
        elif to_optimize == "g":
          return loss_g, dist
        else:
          raise ValueError("Optimization target has been misspecified.")

    return step_fn

  def to_dual_potentials(
      self, finetune_g: bool = True
  ) -> potentials.DualPotentials:
    r"""Return the Kantorovich dual potentials from the trained potentials.

    If `g` returns the gradient of the g potential, \nabla_y g,
    i.e. `g.returns_potential == False`,
    construct the value of the potential with

    .. math::
      g(y) = -f(\nabla_y g(y)) + y^T \nabla_y g(y)

    where :math:`\nabla_y g(y)` is detached for the envelope theorem
    to give the appropriate first derivatives of this construction.

    Args:
      finetune_g: Run the conjugate solver to finetune the prediction.
    """
    f = lambda x: self.state_f.apply_fn({"params": self.state_f.params}, x)

    def g_prediction(y):
      if self.g_returns_potential:
        return self.state_g.apply_fn({"params": self.state_g.params}, y)
      else:
        squeeze = y.ndim == 1
        if squeeze:
          y = jnp.expand_dims(y, 0)
        grad_g_y = stop_gradient(
            self.state_g.apply_fn({"params": self.state_g.params}, y)
        )
        g_y = -f(grad_g_y) + jax.vmap(jnp.dot)(grad_g_y, y)
        return g_y.squeeze(0) if squeeze else g_y

    def g_finetuned(y):
      x_hat = jax.grad(g_prediction)(y)
      grad_g_y = stop_gradient(
          self.conjugate_solver.solve(f, y, x_init=x_hat).grad
      )
      g_y = -f(grad_g_y) + jnp.dot(grad_g_y, y)
      return g_y

    g = g_prediction if not finetune_g else g_finetuned
    return potentials.DualPotentials(
        f, g, cost_fn=costs.SqEuclidean(), corr=True
    )

  @staticmethod
  def _clip_weights_icnn(params):
    params = params.unfreeze()
    for k in params.keys():
      if k.startswith("w_z"):
        params[k]["kernel"] = jnp.clip(params[k]["kernel"], a_min=0)

    return core.freeze(params)

  @staticmethod
  def _penalize_weights_icnn(params: Dict[str, jnp.ndarray]) -> float:
    penalty = 0.0
    for k, param in params.items():
      if k.startswith("w_z"):
        penalty += jnp.linalg.norm(jax.nn.relu(-param["kernel"]))
    return penalty
