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
"""Configuration to train a fairness aware classifier on the adult dataset."""

import ml_collections


def get_config():
  """Return a ConfigDict."""
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
