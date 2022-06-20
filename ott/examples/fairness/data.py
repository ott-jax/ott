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
"""Loads the adult dataset data."""

import os

import jax
import numpy as np
import pandas as pd

open_fn = open


def load_df(
    data_path: str,
    info_path: str,
    protected: str,
    strip_target: bool = True,
    **kwargs
):
  """Load a pandas dataframe from two filenames."""
  with open_fn(data_path, 'r') as fp:
    df = pd.read_csv(fp, skipinitialspace=True, header=None, **kwargs)

  headers = []
  targets = []
  with open_fn(info_path, 'r') as fp:
    for line in fp:
      if line.startswith('|') or not line.strip():
        continue

      parts = line.split(':')
      if len(parts) > 1:
        headers.append(parts[0])
      else:
        pattern = '\n\t.' if strip_target else '\n\t'
        targets = [x.strip() for x in line.strip(pattern).split(',')]

  # Finds the index of the column target.
  df2 = (df == targets[1]).any(axis=0)
  target_idx = df2.index[df2][0]
  if len(headers) < len(df.columns):
    headers.insert(target_idx, 'target')
  df.columns = headers
  target = df.columns[target_idx]

  # Change targets and protected columns to integers
  for col in [protected, target]:
    vs = sorted(df[col].unique())
    df[col] = df[col].map(vs.index)

  return df


def categoricals_to_onehots(df):
  """Turn string features into onehot vectors."""
  categoricals = {
      k: df[k].unique().tolist()
      for k in df.columns
      if not pd.api.types.is_numeric_dtype(df[k])
  }

  def onehots(row):
    result = {}
    for col in row.keys():
      category = categoricals.get(col, None)
      if category is not None:
        result[col] = np.zeros(len(category))
        result[col][category.index(row[col])] = 1.0
    return pd.Series(result)

  return df.apply(onehots, axis=1)


def whiten(df, reference_df=None, target='target'):
  """Make the numerical data have zero means and unit variance."""
  df_ref = df if reference_df is None else reference_df
  cols = [
      k for k in df.columns
      if pd.api.types.is_numeric_dtype(df[k]) and k != target
  ]
  df_num = df[cols].astype(np.float32)
  df_ref = df_ref[cols].astype(np.float32)
  return (df_num - df_ref.mean()) / df_ref.std()


def get_dims(data):
  """Given a record array, extract the dimensions of each column."""
  x, _ = data
  dims = [x[0][name].shape for name in x.dtype.names]
  return [1 if not d else d[0] for d in dims]


def load_train_test(config):
  """Load the training data, the test data and the dimensions of the input."""
  train_path = os.path.join(config.folder, config.training_filename)
  test_path = os.path.join(config.folder, config.test_filename)
  info_path = os.path.join(config.folder, config.info_filename)

  train_df = load_df(train_path, info_path, config.protected, strip_target=True)
  test_df = load_df(
      test_path, info_path, config.protected, strip_target=False, skiprows=1
  )

  result = []
  for df, ref_df in zip((train_df, test_df), (None, train_df)):
    target_df = df['target']
    protected_df = df[config.protected]
    protected_df.name = 'protected'
    num_df = whiten(df, reference_df=ref_df, target='target')
    cat_df = categoricals_to_onehots(df)
    x = pd.concat([num_df, cat_df], axis=1).to_records(index=False)
    y_true = pd.concat([protected_df, target_df],
                       axis=1).to_records(index=False)
    result.append((x, y_true))

  dims = [x[0][name].shape for name in result[0][0].dtype.names]
  dims = [1 if not d else d[0] for d in dims]
  return tuple(result) + (dims,)


def flatten(record):
  """Turn the record array into a flat numpy array."""
  result = [np.stack(record[name]) for name in record.dtype.names]
  result = [e[:, np.newaxis] if len(e.shape) == 1 else e for e in result]
  return np.concatenate(result, axis=1)


def prepare_batch_for_pmap(batch):
  """Prepare the batch with the proper shapes for multi-devices setups."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, batch)


def generate(data, batch_size: int = 256, num_epochs: int = 1):
  """Generate batches of examples, shuffling after each 'epoch'."""
  x, y_true = data
  size = x.shape[0]
  round_num_examples = (size // batch_size) * batch_size
  num_epochs = round_num_examples if num_epochs is None else num_epochs

  count = 0
  while count < num_epochs:
    order = np.arange(size)
    np.random.shuffle(order)
    x = x[order]
    y_true = y_true[order]
    for i in range(0, round_num_examples, batch_size):
      end = i + batch_size
      yield prepare_batch_for_pmap({
          'features': flatten(x[i:end]),
          'label': y_true[i:end].target,
          'protected': y_true[i:end].protected,
      })
    count += 1
