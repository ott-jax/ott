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
"""Runs the training of the network on CIFAR10."""

from typing import Sequence

import jax
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

from ott.examples.fairness import train

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir', '/tmp/soft_error/', 'Directory to store model data.'
)
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True
)
flags.DEFINE_integer('seed', 0, 'Random seed')
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX host: %d / %d', jax.host_id(), jax.host_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f'host_id: {jax.host_id()}, host_count: {jax.host_count()}'
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir'
  )

  train.train_and_evaluate(FLAGS.workdir, FLAGS.config, FLAGS.seed)


if __name__ == '__main__':
  app.run(main)
