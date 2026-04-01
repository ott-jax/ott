from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax.numpy as jnp

import flax.linen as nn
import optax

from ott.neural.networks.potentials import BasePotential, PotentialTrainState


class ConditionalPerturbationNetwork(BasePotential):
  """Condition-aware perturbation network for OT maps."""

  dim_hidden: Sequence[int] = None
  dim_data: int = None
  dim_cond: int = None  # Full dimension of all context variables concatenated
  # Same length as context_entity_bonds if embed_cond_equal is False
  # (if True, first item is size of deep set layer, rest is ignored)
  dim_cond_map: Iterable[int] = (50,)
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  is_potential: bool = False
  layer_norm: bool = False
  embed_cond_equal: bool = (
      False  # Whether all context variables should be treated as set or not
  )
  context_entity_bonds: Iterable[Tuple[int, int]] = (
      (0, 10),
      (10, 20),
  )  # (start, stop) slicing bounds per context modality in c;
  # should be contiguous, non-overlapping by default.
  num_contexts: int = 2

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      c: Optional[jnp.ndarray] = None
  ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:  # noqa: D102
    """Forward pass: map (x, c) -> x + residual.

    Args:
        x: Input data of shape ``(batch, dim_data)``.
        c: Context vector of shape ``(batch, dim_cond)``.  May
            contain multiple modalities concatenated along the last
            axis.  ``context_entity_bonds`` specifies which slice
            ``c[:, start:stop]`` belongs to each modality.  Slices
            should generally be contiguous and non-overlapping, e.g.
            ``((0, 10), (10, 20))`` for two 10-dim modalities.

    Returns:
        Mapped output of shape ``(batch, dim_data)``.
    """
    return_batch = False
    if isinstance(x, dict):
      c = x["c"]
      x = x["X"]
      return_batch = True

    n_input = x.shape[-1]

    # Chunk the inputs
    contexts = [
        c[:, e[0]:e[1]]
        for i, e in enumerate(self.context_entity_bonds)
        if i < self.num_contexts
    ]

    if not self.embed_cond_equal:
      # Each context is processed by a different layer,
      # good for combining modalities
      assert len(self.context_entity_bonds) == len(self.dim_cond_map), (
          "Length of context entity bonds and context map sizes have to "
          f"match: {self.context_entity_bonds} != {self.dim_cond_map}"
      )

      layers = [
          nn.Dense(self.dim_cond_map[i], use_bias=True)
          for i in range(len(contexts))
      ]
      embeddings = [
          self.act_fn(layers[i](context)) for i, context in enumerate(contexts)
      ]
      cond_embedding = jnp.concatenate(embeddings, axis=1)
    else:
      # We can use any number of contexts from the same modality,
      # via a permutation-invariant deep set layer.
      sizes = [c.shape[-1] for c in contexts]
      if not len(set(sizes)) == 1:
        raise ValueError(
            "For embedding a set, all contexts need same length ,"
            f"not {sizes}"
        )
      layer = nn.Dense(self.dim_cond_map[0], use_bias=True)
      embeddings = [self.act_fn(layer(context)) for context in contexts]
      # Average along stacked dimension
      # (alternatives like summing are possible)
      cond_embedding = jnp.mean(jnp.stack(embeddings), axis=0)

    z = jnp.concatenate((x, cond_embedding), axis=1)
    if self.layer_norm:
      n = nn.LayerNorm()
      z = n(z)

    for n_hidden in self.dim_hidden:
      wx = nn.Dense(n_hidden, use_bias=True)
      z = self.act_fn(wx(z))
    wx = nn.Dense(n_input, use_bias=True)

    y = x + wx(z)

    if return_batch:
      return {"X": y, "c": c}
    return y

  def create_train_state(
      self,
      rng: jnp.ndarray,
      optimizer: optax.OptState,
      dim_data: int,
      **kwargs: Any,
  ) -> PotentialTrainState:
    """Create initial `TrainState`."""
    c = jnp.ones((1, self.dim_cond))  # (n_batch, embed_dim)
    x = jnp.ones((1, dim_data))  # (n_batch, data_dim)
    params = self.init(rng, x=x, c=c)["params"]
    return PotentialTrainState.create(
        apply_fn=self.apply,
        params=params,
        tx=optimizer,
        potential_value_fn=self.potential_value_fn,
        potential_gradient_fn=self.potential_gradient_fn,
        **kwargs,
    )
