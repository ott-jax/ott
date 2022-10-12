# Copyright 2022 The OTT Authors
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
"""Sinkhorn initializers."""
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from ott.core import linear_problems, sinkhorn
from ott.geometry import geometry, pointcloud

__all__ = [
    "DefaultInitializer", "GaussianInitializer", "SortingInitializer",
    "MetaInitializer"
]


@jax.tree_util.register_pytree_node_class
class SinkhornInitializer(ABC):
  """Base class for Sinkhorn initializers."""

  @abstractmethod
  def init_dual_a(
      self, ot_prob: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialization for Sinkhorn potential/scaling f_u."""

  @abstractmethod
  def init_dual_b(
      self, ot_prob: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialization for Sinkhorn potential/scaling g_v."""

  def __call__(
      self,
      ot_prob: linear_problems.LinearProblem,
      a: Optional[jnp.ndarray],
      b: Optional[jnp.ndarray],
      lse_mode: bool,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize Sinkhorn potentials/scalings f_u and g_v.

    Args:
      ot_prob: Linear OT problem.
      a: Initial potential/scaling f_u. If ``None``, it will be initialized using
        :meth:`init_dual_a`.
      b: Initial potential/scaling g_v. If ``None``, it will be initialized using
        :meth:`init_dual_b`.
      lse_mode: Return potentials if true, scalings otherwise.

    Returns:
      The initial potentials/scalings.
    """
    n, m = ot_prob.geom.shape
    if a is None:
      a = self.init_dual_a(ot_prob, lse_mode=lse_mode)
    if b is None:
      b = self.init_dual_b(ot_prob, lse_mode=lse_mode)

    assert a.shape == (
        n,
    ), f"Expected `f_u` to have shape `{n,}`, found `{a.shape}`."
    assert b.shape == (
        m,
    ), f"Expected `g_v` to have shape `{m,}`, found `{b.shape}`."

    # cancel dual variables for zero weights
    a = jnp.where(ot_prob.a > 0., a, -jnp.inf if lse_mode else 0.)
    b = jnp.where(ot_prob.b > 0., b, -jnp.inf if lse_mode else 0.)

    return a, b

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "SinkhornInitializer":
    return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class DefaultInitializer(SinkhornInitializer):
  """Default initialization of Sinkhorn dual potentials/primal scalings."""

  def init_dual_a(
      self, ot_prob: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialize Sinkhorn potential/scaling f_u.

    Args:
      ot_prob: OT problem between discrete distributions of size n and m.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      potential/scaling, array of size n.
    """
    a = ot_prob.a
    init_dual_a = jnp.zeros_like(a) if lse_mode else jnp.ones_like(a)
    return init_dual_a

  def init_dual_b(
      self, ot_prob: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    """Initialize Sinkhorn potential/scaling g_v.

    Args:
      ot_prob: OT problem between discrete distributions of size n and m.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      potential/scaling, array of size m.
    """
    b = ot_prob.b
    init_dual_b = jnp.zeros_like(b) if lse_mode else jnp.ones_like(b)
    return init_dual_b


@jax.tree_util.register_pytree_node_class
class GaussianInitializer(DefaultInitializer):
  """Gaussian initializer :cite:`thornton2022rethinking:22`.

  Compute Gaussian approximations of each point cloud, then compute closed from
  Kantorovich potential between Gaussian approximations using Brenier's theorem
  (adapt convex/Brenier potential to Kantorovich). Use this Gaussian potential
  to initialize Sinkhorn potentials/scalings.
  """

  def init_dual_a(
      self,
      ot_prob: linear_problems.LinearProblem,
      lse_mode: bool,
  ) -> jnp.ndarray:
    """Gaussian initialization function.

    Args:
      ot_prob: OT problem between discrete distributions of size n and m.
      lse_mode: Return potential if true, scaling if false.

    Returns:
      potential/scaling, array of size n.
    """
    # import Gaussian here due to circular imports
    from ott.tools.gaussian_mixture import gaussian

    assert isinstance(
        ot_prob.geom, pointcloud.PointCloud
    ), "Gaussian initializer valid only for point clouds."

    x, y = ot_prob.geom.x, ot_prob.geom.y
    a, b = ot_prob.a, ot_prob.b

    gaussian_a = gaussian.Gaussian.from_samples(x, weights=a)
    gaussian_b = gaussian.Gaussian.from_samples(y, weights=b)
    # Brenier potential for cost ||x-y||^2/2, multiply by two for ||x-y||^2
    f_potential = 2 * gaussian_a.f_potential(dest=gaussian_b, points=x)
    f_potential = f_potential - jnp.mean(f_potential)
    f_u = f_potential if lse_mode else ot_prob.geom.scaling_from_potential(
        f_potential
    )
    return f_u


@jax.tree_util.register_pytree_node_class
class SortingInitializer(DefaultInitializer):
  """Sorting initializer :cite:`thornton2022rethinking:22`.

  Solves non-regularized OT problem via sorting, then compute potential through
  iterated minimum on C-transform and use this potential to initialize
  regularized potential.

  Args:
    vectorized_update: Use vectorized inner loop if true.
    tolerance: DualSort convergence threshold.
    max_iter: Max DualSort steps.
  """

  def __init__(
      self,
      vectorized_update: bool = True,
      tolerance: float = 1e-2,
      max_iter: int = 100
  ):
    super().__init__()
    self.tolerance = tolerance
    self.max_iter = max_iter
    self.vectorized_update = vectorized_update

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return ([], {
        'tolerance': self.tolerance,
        'max_iter': self.max_iter,
        'vectorized_update': self.vectorized_update
    })

  def _init_sorting_dual(
      self, modified_cost: jnp.ndarray, init_f: jnp.ndarray
  ) -> jnp.ndarray:
    """Run DualSort algorithm.

    Args:
      modified_cost: cost matrix minus diagonal column-wise.
      init_f: potential f, array of size n. This is the starting potential,
        which is then updated to make the init potential, so an init of an init.

    Returns:
      potential f, array of size n.
    """

    def body_fn(
        state: Tuple[jnp.ndarray, float, int]
    ) -> Tuple[jnp.ndarray, float, int]:
      prev_f, _, it = state
      new_f = fn(prev_f, modified_cost)
      diff = jnp.sum((new_f - prev_f) ** 2)
      it += 1
      return new_f, diff, it

    def cond_fn(state: Tuple[jnp.ndarray, float, int]) -> bool:
      _, diff, it = state
      return jnp.logical_and(diff > self.tolerance, it < self.max_iter)

    fn = _vectorized_update if self.vectorized_update else _coordinate_update
    state = (init_f, jnp.inf, 0)  # init, error, iter
    f_potential, _, _ = jax.lax.while_loop(
        cond_fun=cond_fn, body_fun=body_fn, init_val=state
    )

    return f_potential

  def init_dual_a(
      self,
      ot_prob: linear_problems.LinearProblem,
      lse_mode: bool,
      init_f: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Apply DualSort algorithm.

    Args:
      ot_prob: OT problem.
      lse_mode: Return potential if true, scaling if false.
      init_f: potential f, array of size n. This is the starting potential,
        which is then updated to make the init potential, so an init of an init.

    Returns:
      potential/scaling f_u, array of size n.
    """
    assert not ot_prob.geom.is_online, \
        "Sorting initializer does not work for online geometry."
    # check for sorted x, y requires point cloud and could slow initializer
    cost_matrix = ot_prob.geom.cost_matrix

    assert cost_matrix.shape[0] == cost_matrix.shape[
        1], "Requires square cost matrix."

    modified_cost = cost_matrix - jnp.diag(cost_matrix)[None, :]

    n = cost_matrix.shape[0]
    init_f = jnp.zeros(n) if init_f is None else init_f

    f_potential = self._init_sorting_dual(modified_cost, init_f)
    f_potential = f_potential - jnp.mean(f_potential)

    f_u = f_potential if lse_mode else ot_prob.geom.scaling_from_potential(
        f_potential
    )

    return f_u


@jax.tree_util.register_pytree_node_class
class MetaInitializer(DefaultInitializer):
  """Meta OT Initializer with a fixed geometry :cite:`amos:22`.

  This initializer consists of a predictive model that outputs the
  :math:`f` duals to solve the entropy-regularized OT problem given
  input probability weights ``a`` and ``b``, and a given (assumed to be
  fixed) geometry ``geom``.
  The model's parameters are learned using a training set of OT
  instances (multiple pairs of probability weights), that assume the
  **same** geometry ``geom`` is used throughout, both for training and
  evaluation. The meta model defaults to the MLP in
  :class:`~ott.core.initializers.MetaMLP` and, with batched problem
  instances passed into :meth:`update`.

  **Sample training usage.** The following code shows a simple
  example of using ``update`` to train the model, where
  ``a`` and ``b`` are the weights of the measures and
  ``geom`` is the fixed geometry.

  .. code-block:: python

    meta_initializer = init_lib.MetaInitializer(geom=geom)
    while training():
      a, b = sample_batch()
      loss, init_f, meta_initializer.state = meta_initializer.update(
        meta_initializer.state, a=a, b=b)

  Args:
    geom: The fixed geometry of the problem instances.
    meta_model: The model to predict the potential :math:`f` from the measures.
    opt: The optimizer to update the parameters.
    rng: The PRNG key to use for initializing the model.
    state: The training state of the model to start from.
  """

  def __init__(
      self,
      geom: geometry.Geometry,
      meta_model: Optional[nn.Module] = None,
      opt: optax.GradientTransformation = optax.adam(learning_rate=1e-3),
      rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(0),
      state: Optional[train_state.TrainState] = None
  ):
    self.geom = geom
    self.dtype = geom.x.dtype
    self.opt = opt
    self.rng = rng

    na, nb = geom.shape
    self.meta_model = MetaMLP(
        potential_size=na
    ) if meta_model is None else meta_model

    if state is None:
      # Initialize the model's training state.
      a_placeholder = jnp.zeros(na, dtype=self.dtype)
      b_placeholder = jnp.zeros(nb, dtype=self.dtype)
      params = self.meta_model.init(rng, a_placeholder, b_placeholder)['params']
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

    where :math:`a,b` are the probabilities of the measures :math:`\alpha,\beta`,
    :math:`\mathcal{D}` is a meta distribution of optimal transport problems,

    .. math::
      -J(f; \alpha, \beta, c) := \langle f, a\rangle + \langle g, b \rangle -
        \varepsilon\left\langle \exp\{f/\varepsilon\}, K\exp\{g/\varepsilon\}\right\rangle

    is the entropic dual objective,
    and :math:`K_{i,j} := -C_{i,j}/\varepsilon` is the *Gibbs kernel*.

    Args:
      state: Optimizer state of the meta model.
      a: Probabilites of the :math:`\alpha` measure's atoms.
      b: Probabilites of the :math:`\beta` measure's atoms.

    Returns:
      The training loss, :math:`f`, and updated state.
    """
    return self.update_impl(state, a, b)

  def init_dual_a(
      self, ot_prob: linear_problems.LinearProblem, lse_mode: bool
  ) -> jnp.ndarray:
    # Detect if the problem is batched.
    assert ot_prob.a.ndim in (1, 2) and ot_prob.b.ndim in (1, 2)
    vmap_a_val = 0 if ot_prob.a.ndim == 2 else None
    vmap_b_val = 0 if ot_prob.b.ndim == 2 else None

    if vmap_a_val is not None or vmap_b_val is not None:
      compute_f_maybe_batch = jax.vmap(
          self._compute_f, in_axes=(vmap_a_val, vmap_b_val, None)
      )
    else:
      compute_f_maybe_batch = self._compute_f

    init_f = compute_f_maybe_batch(ot_prob.a, ot_prob.b, self.state.params)
    f_u = init_f if lse_mode else ot_prob.geom.scaling_from_potential(init_f)
    return f_u

  def _get_update_fn(self):
    """Return the implementation (and jitted) update function."""

    def dual_obj_loss_single(params, a, b):
      f_pred = self._compute_f(a, b, params)
      g_pred = self.geom.update_potential(
          f_pred, jnp.zeros_like(b), jnp.log(b), 0, axis=0
      )
      g_pred = jnp.where(jnp.isfinite(g_pred), g_pred, 0.)

      ot_prob = linear_problems.LinearProblem(geom=self.geom, a=a, b=b)
      dual_obj = sinkhorn.ent_reg_cost(f_pred, g_pred, ot_prob, lse_mode=True)
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

  def _compute_f(self, a, b, params):
    r"""Predict the optimal :math:`f` potential.

    Args:
      a: Probabilites of the :math:`\alpha` measure's atoms.
      b: Probabilites of the :math:`\beta` measure's atoms.
      params: The parameters of the Meta model.

    Returns:
      The :math:`f` potential.
    """
    return self.meta_model.apply({'params': params}, a, b)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [self.geom, self.meta_model, self.opt], {
        'rng': self.rng,
        'state': self.state
    }


class MetaMLP(nn.Module):
  r"""A Meta MLP potential for :class:`~ott.core.initializers.MetaInitializer`.

  This provides an MLP :math:`\hat f_\theta(a, b)` that maps from the probabilities
  of the measures to the optimal dual potentials :math:`f`.

  Args:
    potential_size: The dimensionality of :math:`f`.
    num_hidden_units: The number of hidden units in each layer.
    num_hidden_layers: The number of hidden layers.
  """

  potential_size: int
  num_hidden_units: int = 512
  num_hidden_layers: int = 3

  @nn.compact
  def __call__(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    r"""Make a prediction.

    Args:
      a: Probabilites of the :math:`\alpha` measure's atoms.
      b: Probabilites of the :math:`\beta` measure's atoms.

    Returns:
      The :math:`f` potential.
    """
    dtype = a.dtype
    z = jnp.concatenate((a, b))
    for _ in range(self.num_hidden_layers):
      z = nn.relu(nn.Dense(self.num_hidden_units, dtype=dtype)(z))
    f = nn.Dense(self.potential_size, dtype=dtype)(z)
    return f


def _vectorized_update(
    f: jnp.ndarray, modified_cost: jnp.ndarray
) -> jnp.ndarray:
  """Inner loop DualSort Update.

  Args:
    f : potential f, array of size n.
    modified_cost: cost matrix minus diagonal column-wise.

  Returns:
    updated potential vector, f.
  """
  return jnp.min(modified_cost + f[None, :], axis=1)


def _coordinate_update(
    f: jnp.ndarray, modified_cost: jnp.ndarray
) -> jnp.ndarray:
  """Coordinate-wise updates within inner loop.

  Args:
    f: potential f, array of size n.
    modified_cost: cost matrix minus diagonal column-wise.

  Returns:
    updated potential vector, f.
  """

  def body_fn(i: int, f: jnp.ndarray) -> jnp.ndarray:
    new_f = jnp.min(modified_cost[i, :] + f)
    return f.at[i].set(new_f)

  return jax.lax.fori_loop(0, len(f), body_fn, f)
