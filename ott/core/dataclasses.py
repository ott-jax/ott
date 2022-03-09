# coding=utf-8
"""pytree_nodes Dataclasses."""

import dataclasses
import jax


def register_pytree_node(cls):
  """Decorator to register dataclasses as pytree_nodes."""
  cls = dataclasses.dataclass()(cls)
  flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
  unflatten = lambda d, children: cls(**d.unflatten(children))
  jax.tree_util.register_pytree_node(cls, flatten, unflatten)
  return cls
