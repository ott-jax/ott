# coding=utf-8
"""Configuration to train a fairness aware classifier on the adult dataset."""

import ml_collections


def get_config():
  """Returns a ConfigDict."""

  config = ml_collections.ConfigDict()
  config.folder = '/tmp/adult_dataset/'
  config.training_filename = 'adult.data'
  config.test_filename = 'adult.test'
  config.info_filename = 'adult.names'
  config.protected = 'sex'

  config.batch_size = 256
  config.num_epochs = 20
  config.embed_dim = 16
  config.hidden_layers = (64, 64)
  config.learning_rate = 1e-4

  config.epsilon = 1e-3
  config.quantization = 16
  config.num_groups = 2
  config.fair_weight = 1.0
  return config
