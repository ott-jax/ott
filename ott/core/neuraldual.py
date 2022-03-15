# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A Jax implementation of the ICNN based Kantorovich dual."""

from typing import Iterator, Optional
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax.core import freeze
from optax._src import base
import warnings
from tqdm import tqdm
from ott.core import icnn


class NeuralDualSolver:
  r"""Solver of the ICNN-based Kantorovich dual.

  The algorithm is described in:
  Optimal transport mapping via input convex neural networks,
  Makkuva-Taghvaei-Lee-Oh, ICML'20.
  http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf

  Args:
    input_dim: input dimensionality of data required for network init
    neural_f: network architecture for potential f
    neural_g: network architecture for potential g
    optimizer_f: optimizer function for potential f
    optimizer_g: optimizer function for potential g
    num_train_iters: number of total training iterations
    num_inner_iters: number of training iterations of g per iteration of f
    valid_freq: frequency with which model is validated
    log_freq: frequency with training and validation are logged
    logging: option to return logs
    seed: random seed for network initialiations
    pos_weights: option to train networks with potitive weights or regularizer
    beta: regularization parameter when not training with positive weights

  Returns:
    the `NeuralDual` containing the optimal dual potentials f and g
  """

  def __init__(self,
               input_dim: int,
               neural_f: Optional[nn.Module] = None,
               neural_g: Optional[nn.Module] = None,
               optimizer_f: Optional[base.GradientTransformation] = None,
               optimizer_g: Optional[base.GradientTransformation] = None,
               num_train_iters: int = 100,
               num_inner_iters: int = 10,
               valid_freq: int = 100,
               log_freq: int = 100,
               logging: bool = False,
               seed: int = 0,
               pos_weights: bool = True,
               beta: int = 1.0):
    self.num_train_iters = num_train_iters
    self.num_inner_iters = num_inner_iters
    self.valid_freq = valid_freq
    self.log_freq = log_freq
    self.logging = logging
    self.pos_weights = pos_weights
    self.beta = beta

    # set random key
    rng = jax.random.PRNGKey(seed)

    # set default optimizers
    if optimizer_f is None:
      optimizer_f = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)
    if optimizer_g is None:
      optimizer_g = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.9, eps=1e-8)

    # set default neural architectures
    if neural_f is None:
      neural_f = icnn.ICNN(dim_hidden=[64, 64, 64, 64])
    if neural_g is None:
      neural_g = icnn.ICNN(dim_hidden=[64, 64, 64, 64])

    # set optimizer and networks
    self.setup(rng, neural_f, neural_g, input_dim, optimizer_f, optimizer_g)

  def setup(self, rng, neural_f, neural_g, input_dim, optimizer_f, optimizer_g):
    """Setup all components required to train the `NeuralDual`."""
    # split random key
    rng, rng_f, rng_g = jax.random.split(rng, 3)

    # check setting of network architectures
    if (neural_f.pos_weights != self.pos_weights
       or neural_g.pos_weights != self.pos_weights):
      warnings.warn(f"Setting of ICNN and the positive weights setting of the \
                      `NeuralDualSolver` are not consistent. Proceeding with \
                      the `NeuralDualSolver` setting, with positive weigths \
                      being {self.positive_weights}.")
      neural_f.pos_weights = self.pos_weights
      neural_g.pos_weights = self.pos_weights

    self.state_f = self.create_train_state(
        rng_f, neural_f, optimizer_f, input_dim)
    self.state_g = self.create_train_state(
        rng_g, neural_g, optimizer_g, input_dim)

    # define train and valid step functions
    self.train_step_f = self.get_step_fn(train=True, to_optimize='f')
    self.valid_step_f = self.get_step_fn(train=False, to_optimize='f')

    self.train_step_g = self.get_step_fn(train=True, to_optimize='g')
    self.valid_step_g = self.get_step_fn(train=False, to_optimize='g')

  def __call__(self,
               trainloader_source: Iterator[jnp.ndarray],
               trainloader_target: Iterator[jnp.ndarray],
               validloader_source: Iterator[jnp.ndarray],
               validloader_target: Iterator[jnp.ndarray],) -> 'NeuralDual':
    logs = self.train_neuraldual(
      trainloader_source, trainloader_target,
      validloader_source, validloader_target
    )
    if self.logging:
      return NeuralDual(self.state_f, self.state_g), logs
    else:
      return NeuralDual(self.state_f, self.state_g)

  def train_neuraldual(self, trainloader_source, trainloader_target,
                       validloader_source, validloader_target):
    """Implementation of the training and validation script."""

    # define dict to contain source and target batch
    batch_g = {}
    batch_f = {}
    valid_batch = {}

    # set logging dictionaries
    train_logs = {
      'train_loss_f': [],
      'train_loss_g': [],
      'train_w_dist': []
    }
    valid_logs = {
      'valid_loss_f': [],
      'valid_loss_g': [],
      'valid_w_dist': []
    }

    for step in tqdm(range(self.num_train_iters)):
      # execute training steps
      for _ in range(self.num_inner_iters):
        # get train batch for potential g
        batch_g['source'] = jnp.array(next(trainloader_source))
        batch_g['target'] = jnp.array(next(trainloader_target))

        self.state_g, loss_g, _ = self.train_step_g(
          self.state_f, self.state_g, batch_g)

      # get train batch for potential f
      batch_f['source'] = jnp.array(next(trainloader_source))
      batch_f['target'] = jnp.array(next(trainloader_target))

      self.state_f, loss_f, w_dist = self.train_step_f(
        self.state_f, self.state_g, batch_f)
      if not self.pos_weights:
        self.state_f = self.state_f.replace(
          params=self.clip_weights_icnn(self.state_f.params))

      # log to wandb
      if self.logging and step % self.log_freq == 0:
        train_logs['train_loss_f'].append(float(loss_f))
        train_logs['train_loss_g'].append(float(loss_g))
        train_logs['train_w_dist'].append(float(w_dist))

      # report the loss on an validuation dataset periodically
      if (step != 0 and step % self.valid_freq == 0):
          # get batch
          valid_batch['source'] = jnp.array(next(validloader_source))
          valid_batch['target'] = jnp.array(next(validloader_target))

          valid_loss_f, _ = self.valid_step_f(
            self.state_f, self.state_g, valid_batch)
          valid_loss_g, valid_w_dist = self.valid_step_g(
            self.state_f, self.state_g, valid_batch)

          if self.logging:
            # log training progress
            valid_logs['valid_loss_f'].append(float(valid_loss_f))
            valid_logs['valid_loss_g'].append(float(valid_loss_g))
            valid_logs['valid_w_dist'].append(float(valid_w_dist))

    return {'train_logs': train_logs, 'valid_logs': valid_logs}

  def get_step_fn(self, train, to_optimize='g'):
    """Create a one-step training and evaluation function."""

    def loss_fn(params_f, params_g, f, g, batch):
      """Loss function for potential f."""
      # get two distributions
      source, target = batch['source'], batch['target']

      # get loss terms of kantorovich dual
      f_t = f({'params': params_f}, batch['target'])

      grad_g_s = jax.vmap(lambda x: jax.grad(g, argnums=1)(
        {'params': params_g}, x))(batch['source'])

      f_grad_g_s = f({'params': params_f}, grad_g_s)

      s_dot_grad_g_s = jnp.sum(source * grad_g_s, axis=1)

      s_sq = jnp.sum(source * source, axis=1)
      t_sq = jnp.sum(target * target, axis=1)

      # compute final wasserstein distance
      dist = 2 * jnp.mean(f_grad_g_s - f_t - s_dot_grad_g_s
                          + 0.5 * t_sq + 0.5 * s_sq)

      loss_f = jnp.mean(f_t - f_grad_g_s)
      loss_g = jnp.mean(f_grad_g_s - s_dot_grad_g_s)

      if to_optimize == 'f':
        return loss_f, dist
      elif to_optimize == 'g':
        if not self.pos_weights:
          penalty = self.penalize_weights_icnn(params_g)
          loss_g += self.beta * penalty
        return loss_g, dist
      else:
        raise ValueError('Optimization target has been misspecified.')

    @jax.jit
    def step_fn(state_f, state_g, batch):
      """Step function of either training or validation."""

      if to_optimize == 'f':
        grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        state = state_f
      elif to_optimize == 'g':
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        state = state_g
      else:
        raise ValueError('Potential to be optimize might be misspecified.')

      if train:
        # compute loss and gradients
        (loss, dist), grads = grad_fn(
          state_f.params, state_g.params,
          state_f.apply_fn, state_g.apply_fn, batch)

        # update state
        return state.apply_gradients(grads=grads), loss, dist

      else:
        # compute loss and gradients
        (loss, dist), _ = grad_fn(
          state_f.params, state_g.params,
          state_f.apply_fn, state_g.apply_fn, batch)

        # do not update state
        return loss, dist

    return step_fn

  def create_train_state(self, rng, model, optimizer, input):
    """Creates initial `TrainState`."""

    params = model.init(rng, jnp.ones(input))['params']
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

  def clip_weights_icnn(params):
    params = params.unfreeze()
    for k in params.keys():
        if (k.startswith('w_z')):
            params[k]['kernel'] = jnp.clip(params[k]['kernel'], a_min=0)

    return freeze(params)

  def penalize_weights_icnn(self, params):
    penalty = 0
    for k in params.keys():
        if (k.startswith('w_z')):
            penalty += jnp.linalg.norm(jax.nn.relu(-params[k]['kernel']))
    return penalty


@jax.tree_util.register_pytree_node_class
class NeuralDual:
  r"""Neural Kantorovich dual.

  Attributes:
    state_f: optimal potential f
    state_g: optimal potential g
  """

  def __init__(self, state_f, state_g):
    self.state_f = state_f
    self.state_g = state_g

  def tree_flatten(self):
    return ((self.state_f, self.state_g), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)

  @property
  def f(self):
    return self.state_f

  @property
  def g(self):
    return self.state_g

  def transport(self, data: jnp.ndarray) -> jnp.ndarray:
    """Transport source data samples with potential g."""

    return jax.vmap(lambda x: jax.grad(self.g.apply_fn, argnums=1)(
      {'params': self.g.params}, x))(data)

  def inverse_transport(self, data: jnp.ndarray) -> jnp.ndarray:
    """Transport source data samples with potential g."""

    return jax.vmap(lambda x: jax.grad(self.f.apply_fn, argnums=1)(
      {'params': self.f.params}, x))(data)

  def distance(self,
               source: jnp.ndarray,
               target: jnp.ndarray) -> float:
    """Given potentials f and g, compute the overall distance."""

    f_t = self.f.apply_fn({'params': self.f.params}, target)

    grad_g_s = jax.vmap(lambda x: jax.grad(self.g.apply_fn, argnums=1)(
      {'params': self.g.params}, x))(source)

    f_grad_g_s = self.f.apply_fn({'params': self.f.params}, grad_g_s)

    s_dot_grad_g_s = jnp.sum(source * grad_g_s, axis=1)

    s_sq = jnp.sum(source * source, axis=1)
    t_sq = jnp.sum(target * target, axis=1)

    # compute final wasserstein distance
    dist = 2 * jnp.mean(f_grad_g_s - f_t - s_dot_grad_g_s
                        + 0.5 * t_sq + 0.5 * s_sq)
    return dist
