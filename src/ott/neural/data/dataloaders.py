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
#import tensorflow as tf


class ConditionalDataLoader:  #TODO(@MUCDK) uncomment, resolve installation issues with TF
  pass

  #def __init__(
  #    self, rng: jax.random.KeyArray, dataloaders: Dict[str, tf.Dataloader],
  #    p: jax.Array
  #) -> None:
  #  super().__init__()
  #  self.rng = rng
  #  self.conditions = dataloaders.keys()
  #  self.p = p

  #def __next__(self) -> jnp.ndarray:
  #  self.rng, rng = jax.random.split(self.rng, 2)
  #  condition = jax.random.choice(rng, self.conditions, p=self.p)
  #  return next(self.dataloaders[condition])
