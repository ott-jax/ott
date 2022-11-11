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
"""Data loading and data augmentation."""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils


def random_shift(img, ratio=0.1):
  """Apply a random shift on an input image."""
  height, _ = img.shape[:2]  # Assumes squared images.
  size = tf.random.uniform(
      shape=(2,),
      minval=int((1 - ratio) * height),
      maxval=height,
      dtype=tf.int32
  )
  size = tf.concat((size, [3]), axis=0)
  img = tf.image.random_crop(img, size)

  deltas = tf.constant([32, 32, 3]) - size
  for _ in tf.range(deltas[0]):
    img = tf.pad(
        img, [tf.random.shuffle([1, 0]), [0, 0], [0, 0]], mode='SYMMETRIC'
    )
  for _ in tf.range(deltas[1]):
    img = tf.pad(
        img, [[0, 0], tf.random.shuffle([1, 0]), [0, 0]], mode='SYMMETRIC'
    )
  return img


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size: int, train: bool):
  """Create an iterator over the training / test set."""
  split = tfds.Split.TRAIN if train else tfds.Split.TEST
  ds = dataset_builder.as_dataset(split=split)
  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=0)

  def augment(inputs):
    im = inputs['image']
    im = tf.image.random_flip_left_right(im)
    im = random_shift(im, ratio=0.1)
    inputs['image'] = im
    return inputs

  def prepare(inputs):
    im = inputs['image']
    inputs['image'] = tf.cast(im, tf.float32) / 255.0
    inputs['label'] = tf.one_hot(inputs['label'], 10)
    inputs.pop('id')
    return inputs

  ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.cache()
  if train:
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(tf.data.AUTOTUNE)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 8)
  return it
