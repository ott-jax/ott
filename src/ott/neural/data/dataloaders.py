
#import tensorflow as tf


class ConditionalDataLoader:
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
