# Copyright 2022 Google LLC.
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
"""Train for the soft-error loss."""

import collections
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import tensorflow_datasets as tfds
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import checkpoints, common_utils

from ott.examples.soft_error import data, losses
from ott.examples.soft_error import model as model_lib


def initialized(key, height, width, model):
  """Initialize the model parameters."""
  input_shape = (1, height, width, 3)

  @jax.jit
  def init(*args):
    return model.init(*args)

  variables = init({'params': key}, jnp.ones(input_shape, jnp.float32))
  model_state, params = variables.pop('params')
  return params, model_state


def compute_metrics(logits, labels, loss_fn):
  loss = loss_fn(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = jax.lax.pmean(metrics, axis_name='batch')
  return metrics


def train_step(apply_fn, loss_fn, state, batch):
  """Perform a single training step."""

  def compute_loss(params):
    variables = {'params': params, **state.model_state}
    logits = apply_fn(variables, batch['image'])
    loss = loss_fn(logits, batch['label'])
    return loss, logits

  grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
  aux, grad = grad_fn(state.optimizer.target)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')
  logits = aux[1]
  new_optimizer = state.optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, batch['label'], loss_fn=loss_fn)
  new_state = state.replace(step=state.step + 1, optimizer=new_optimizer)
  return new_state, metrics


def eval_step(apply_fn, loss_fn, state, batch):
  params = state.optimizer.target
  variables = {'params': params, **state.model_state}
  logits = apply_fn(variables, batch['image'], train=False, mutable=False)
  return compute_metrics(logits, batch['label'], loss_fn=loss_fn)


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  model_state: Any


def create_train_state(rng, config, model, height, width):
  """Create initial training state."""
  params, model_state = initialized(rng, height, width, model)
  optimizer = flax.optim.Adam(learning_rate=config.learning_rate).create(params)
  state = TrainState(step=0, optimizer=optimizer, model_state=model_state)
  return state


def log(results, epoch, summary, train=True, summary_writer=None):
  """Log the metrics to stderr and tensorboard."""
  if jax.host_id() != 0:
    return

  phase = 'train' if train else 'eval'
  for key in ('loss', 'accuracy'):
    results[f'{phase}_{key}'].append((epoch + 1, summary[key]))
  print(
      '{} epoch: {}, loss: {:.3f}, accuracy: {:.2%}'.format(
          phase, epoch + 1, summary['loss'], summary['accuracy']
      )
  )

  for key, val in summary.items():
    summary_writer.scalar(f'{phase}_{key}', val, epoch)


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.host_id() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


def train_and_evaluate(
    workdir: str, config: ml_collections.ConfigDict, seed: int = 0
):
  """Execute model training and evaluation loop."""
  rng = jax.random.PRNGKey(seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

  loss_fn = losses.get(config.loss)
  local_batch_size = config.batch_size // jax.host_count()
  dataset_builder = tfds.builder(config.dataset)
  info = dataset_builder.info
  height, width = info.features['image'].shape[:2]
  train_iter = data.create_input_iter(
      dataset_builder, local_batch_size, train=True
  )
  eval_iter = data.create_input_iter(
      dataset_builder, local_batch_size, train=False
  )
  steps_per_epoch = info.splits['train'].num_examples // config.batch_size
  num_steps = int(steps_per_epoch * config.num_epochs)
  num_validation_examples = info.splits['test'].num_examples
  steps_per_eval = num_validation_examples // config.batch_size

  num_classes = info.features['label'].num_classes
  model = model_lib.CNN(num_classes=num_classes, dtype=jnp.float32)
  state = create_train_state(rng, config, model, height, width)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, model.apply, loss_fn), axis_name='batch'
  )
  p_eval_step = jax.pmap(
      functools.partial(eval_step, model.apply, loss_fn), axis_name='batch'
  )

  results = collections.defaultdict(list)
  epoch_metrics = []
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state=state, batch=batch)
    epoch_metrics.append(metrics)

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      epoch_metrics = common_utils.get_metrics(epoch_metrics)
      summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
      log(results, epoch, summary, train=True, summary_writer=summary_writer)

      epoch_metrics = []
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      log(results, epoch, summary, train=False, summary_writer=summary_writer)

      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  return results, state
