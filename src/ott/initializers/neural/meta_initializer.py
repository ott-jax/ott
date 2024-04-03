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
import functools
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax.core import frozen_dict
from flax.training import train_state

from ott import utils
from ott.geometry import geometry
from ott.initializers.linear import initializers
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

__all__ = ["MetaInitializer"]


@jax.tree_util.register_pytree_node_class
class MetaInitializer(initializers.DefaultInitializer):
  """Meta OT Initializer with a fixed geometry :cite:`amos:22`.

  This initializer consists of a predictive model that outputs the
  :math:`f` duals to solve the entropy-regularized OT problem given
  input probability weights ``a`` and ``b``, and a given (assumed to be
  fixed) geometry ``geom``.

  The model's parameters are learned using a training set of OT
  instances (multiple pairs of probability weights), that assume the
  **same** geometry ``geom`` is used throughout, both for training and
  evaluation.

  Args:
    geom: The fixed geometry of the problem instances.
    meta_model: The model to predict the potential :math:`f` from the measures.
    opt: The optimizer to update the parameters. If :obj:`None`, use
      :func:`optax.adam` with :math:`0.001` learning rate.
    rng: The PRNG key to use for initializing the model.
    state: The training state of the model to start from.

  Examples:
    The following code shows a simple
    example of using ``update`` to train the model, where
    ``a`` and ``b`` are the weights of the measures and
    ``geom`` is the fixed geometry.

    .. code-block:: python

      meta_initializer = init_lib.MetaInitializer(geom)
      while training():
        a, b = sample_batch()
        loss, init_f, meta_initializer.state = meta_initializer.update(
          meta_initializer.state, a=a, b=b
        )
  """

  def __init__(
      self,
      geom: geometry.Geometry,
      meta_model: nn.Module,
      opt: Optional[optax.GradientTransformation
                   ] = optax.adam(learning_rate=1e-3),  # noqa: B008
      rng: Optional[jax.Array] = None,
      state: Optional[train_state.TrainState] = None,
  ):
    self.geom = geom
    self.opt = opt
    self.rng = utils.default_prng_key(rng)

    na, nb = geom.shape
    self.meta_model = meta_model

    if state is None:
      # Initialize the model's training state.
      placeholder = jnp.concatenate([jnp.zeros(na), jnp.zeros(nb)], axis=-1)
      params = self.meta_model.init(self.rng, placeholder)["params"]
      self.state = train_state.TrainState.create(
          apply_fn=self.meta_model.apply, params=params, tx=opt
      )
    else:
      self.state = state

    self.update_impl = self._get_update_fn()

  def update(
      self, state: train_state.TrainState, a: jnp.ndarray, b: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, train_state.TrainState]:
    r"""Update the meta model with the dual objective.

    The goal is for the model to match the optimal duals, i.e.,
    :math:`\hat f_\theta \approx f^\star`.
    This can be done by training the predictions of :math:`\hat f_\theta`
    to optimize the dual objective, which :math:`f^\star` also optimizes for.
    The overall learning setup can thus be written as:

    .. math::
      \min_\theta\; {\mathbb E}_{(\alpha,\beta)\sim{\mathcal{D}}}\;
      J(\hat f_\theta(a, b); \alpha, \beta),

    where :math:`a,b` are the probabilities of the measures :math:`\alpha,\beta`
    ,:math:`\mathcal{D}` is a meta distribution of optimal transport problems,

    .. math::
      -J(f; \alpha, \beta, c) := \langle f, a\rangle + \langle g, b \rangle -
      \varepsilon\left\langle \exp\{f/\varepsilon\}, K\exp\{g/\varepsilon\}
      \right\rangle

    is the entropic dual objective,
    and :math:`K_{i,j} := -C_{i,j}/\varepsilon` is the *Gibbs kernel*.

    Args:
      state: Optimizer state of the meta model.
      a: Probabilities of the :math:`\alpha` measure's atoms.
      b: Probabilities of the :math:`\beta` measure's atoms.

    Returns:
      The training loss, :math:`f`, and updated state.
    """
    return self.update_impl(state, a, b)

  def init_dual_a(  # noqa: D102
      self,
      ot_prob: linear_problem.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    del rng
    # Detect if the problem is batched.
    assert ot_prob.a.ndim in (1, 2)
    assert ot_prob.b.ndim in (1, 2)
    vmap_a_val = 0 if ot_prob.a.ndim == 2 else None
    vmap_b_val = 0 if ot_prob.b.ndim == 2 else None

    if vmap_a_val is not None or vmap_b_val is not None:
      compute_f_maybe_batch = jax.vmap(
          self._compute_f, in_axes=(vmap_a_val, vmap_b_val, None)
      )
    else:
      compute_f_maybe_batch = self._compute_f

    init_f = compute_f_maybe_batch(ot_prob.a, ot_prob.b, self.state.params)
    return init_f if lse_mode else ot_prob.geom.scaling_from_potential(init_f)

  def _get_update_fn(self):
    """Return the implementation (and jitted) update function."""

    def dual_obj_loss_single(params, a, b):
      f_pred = self._compute_f(a, b, params)
      g_pred = self.geom.update_potential(
          f_pred, jnp.zeros_like(b), jnp.log(b), 0, axis=0
      )
      g_pred = jnp.where(jnp.isfinite(g_pred), g_pred, 0.0)

      ot_prob = linear_problem.LinearProblem(geom=self.geom, a=a, b=b)
      dual_obj = sinkhorn.compute_kl_reg_cost(
          f_pred, g_pred, ot_prob, lse_mode=True
      )
      loss = -dual_obj
      return loss, f_pred

    def loss_batch(params, a, b):
      loss_fn = functools.partial(dual_obj_loss_single, params=params)
      loss, f_pred = jax.vmap(loss_fn)(a=a, b=b)
      return jnp.mean(loss), f_pred

    @jax.jit
    def update(state, a, b):
      a = jnp.atleast_2d(a)
      b = jnp.atleast_2d(b)
      grad_fn = jax.value_and_grad(loss_batch, has_aux=True)
      (loss, init_f), grads = grad_fn(state.params, a, b)
      return loss, init_f, state.apply_gradients(grads=grads)

    return update

  def _compute_f(
      self, a: jnp.ndarray, b: jnp.ndarray,
      params: frozen_dict.FrozenDict[str, jnp.ndarray]
  ) -> jnp.ndarray:
    r"""Predict the optimal :math:`f` potential.

    Args:
      a: Probabilities of the :math:`\alpha` measure's atoms.
      b: Probabilities of the :math:`\beta` measure's atoms.
      params: The parameters of the Meta model.

    Returns:
      The :math:`f` potential.
    """
    return self.meta_model.apply({"params": params},
                                 jnp.concatenate([a, b], axis=-1))

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [self.geom, self.meta_model, self.opt], {
        "rng": self.rng,
        "state": self.state
    }
