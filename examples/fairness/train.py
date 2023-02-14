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
"""Training a network on the adult dataset with fairnes constraints."""

import collections
import functools
from typing import Any

import ml_collections

import flax
import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import checkpoints, common_utils

from ott.examples.fairness import data, losses, models


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  model_state: Any


def initialized(key, model, size):
  """Initialize the model."""

  @jax.jit
  def init(*args):
    return model.init(*args)

  variables = init({'params': key}, jnp.ones((1, size)))
  model_state, params = variables.pop('params')
  return params, model_state


def create_train_state(rng, config, model, size):
  """Create initial training state."""
  params, model_state = initialized(rng, model, size)
  optimizer = flax.optim.Adam(learning_rate=config.learning_rate).create(params)
  state = TrainState(step=0, optimizer=optimizer, model_state=model_state)
  return state


def train_step(apply_fn, config, state, batch):
  """Perform a single training step."""
  regularizer = functools.partial(
      losses.fairness_regularizer,
      quantization=config.quantization,
      num_groups=config.num_groups,
      epsilon=config.epsilon
  )

  def compute_loss(params):
    variables = {'params': params, **state.model_state}
    logits = apply_fn(variables, batch['features'], train=True)
    loss = losses.binary_cross_entropy(logits, batch['label'])
    reg = (
        regularizer(logits, batch['protected']) if config.fair_weight > 0 else 0
    )
    return loss + config.fair_weight * reg, logits

  grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
  aux, grad = grad_fn(state.optimizer.target)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')
  logits = aux[1]
  new_optimizer = state.optimizer.apply_gradient(grad)
  metrics = losses.compute_metrics(logits, batch['label'])
  new_state = state.replace(step=state.step + 1, optimizer=new_optimizer)
  return new_state, metrics


def eval_step(apply_fn, state, batch):
  params = state.optimizer.target
  variables = {'params': params, **state.model_state}
  logits = apply_fn(variables, batch['features'], train=False, mutable=False)
  return losses.compute_metrics(logits, batch['label'])


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

  if summary_writer is None:
    return

  for key, val in summary.items():
    summary_writer.scalar(f'{phase}_{key}', val, epoch)


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.host_id() != 0:
    return
  # Gets train state from the first replica.
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

  local_batch_size = config.batch_size // jax.host_count()
  train_ds, test_ds, dims = data.load_train_test(config)
  train_iter = data.generate(
      train_ds, batch_size=local_batch_size, num_epochs=config.num_epochs
  )
  train_iter = jax_utils.prefetch_to_device(train_iter, 8)

  model = models.AdultModel(
      encoder_cls=functools.partial(
          models.FeaturesEncoder, input_dims=dims, embed_dim=config.embed_dim
      ),
      hidden=config.hidden_layers
  )

  state = create_train_state(rng, config, model, sum(dims))
  state = restore_checkpoint(state, workdir)
  step_offset = int(state.step)
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, model.apply, config), axis_name='batch'
  )
  p_eval_step = jax.pmap(
      functools.partial(eval_step, model.apply), axis_name='batch'
  )

  steps_per_epoch = train_ds[0].shape[0] // config.batch_size
  num_steps = steps_per_epoch * config.num_epochs

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
      for eval_batch in data.generate(test_ds, batch_size=local_batch_size):
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      log(results, epoch, summary, train=False, summary_writer=summary_writer)
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  return results, state
