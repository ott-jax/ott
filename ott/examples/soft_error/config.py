# coding=utf-8
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.dataset = 'cifar10'
  config.learning_rate = 1e-4
  config.batch_size = 64
  config.num_epochs = 100
  config.loss = 'soft_error'
  return config
