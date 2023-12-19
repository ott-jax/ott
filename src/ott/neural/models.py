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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.core import frozen_dict
from flax.training import train_state

from ott import utils
from ott.geometry import geometry
from ott.initializers.linear import initializers as lin_init
from ott.neural import layers
from ott.neural.solvers import neuraldual
from ott.problems.linear import linear_problem

__all__ = ["ICNN", "MLP", "MetaInitializer"]

# wrap to silence docs linter
DEFAULT_KERNEL_INIT = lambda *a, **k: nn.initializers.normal()(*a, **k)
DEFAULT_RECTIFIER = nn.activation.relu
DEFAULT_ACTIVATION = nn.activation.relu


class ICNN(neuraldual.BaseW2NeuralDual):
  """Input convex neural network (ICNN).

  Implementation of input convex neural networks as introduced in
  :cite:`amos:17` with initialization schemes proposed by :cite:`bunne:22`.

  Args:
    dim_data: data dimensionality.
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is 1 by default.
    rank: rank of the matrices :math:`A_i` used as low-rank factors
      for the quadratic potentials.
    init_fn: Initializer for the kernel weight matrices.
      The default is :func:`~flax.linen.initializers.normal`.
    act_fn: choice of activation function used in network architecture,
      needs to be convex. The default is :func:`~flax.linen.activation.relu`.
    pos_weights: Enforce positive weights with a projection.
      If :obj:`False`, the positive weights should be enforced with clipping
      or regularization in the loss.
    rectifier_fn: function to ensure the non negativity of the weights.
      The default is :func:`~flax.linen.activation.relu`.
    gaussian_map_samples: Tuple of source and target points, used to initialize
      the ICNN to mimic the linear Bures map that morphs the (Gaussian
      approximation) of the input measure to that of the target measure. If
      :obj:`None`, the identity initialization is used, and ICNN mimics half the
      squared Euclidean norm.
  """

  dim_data: int
  dim_hidden: Sequence[int]
  rank: int = 1
  init_fn: Callable[[jax.Array, Tuple[int, ...], Any],
                    jnp.ndarray] = DEFAULT_KERNEL_INIT
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = DEFAULT_ACTIVATION
  pos_weights: bool = False
  rectifier_fn: Callable[[jnp.ndarray], jnp.ndarray] = DEFAULT_RECTIFIER
  gaussian_map_samples: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None

  @property
  def is_potential(self) -> bool:  # noqa: D102
    return True

  def setup(self) -> None:  # noqa: D102
    dim_hidden = list(self.dim_hidden) + [1]
    # final layer computes average, still with normalized rescaling
    self.w_zs = [self._get_wz(dim) for dim in dim_hidden[1:]]
    # subsequent layers re-injected into convex functions
    self.w_xs = [self._get_wx(dim) for dim in dim_hidden]
    self.pos_def_potentials = self._get_pos_def_potentials()

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    w_x, *w_xs = self.w_xs
    assert len(self.w_zs) == len(w_xs), (len(self.w_zs), len(w_xs))

    z = self.act_fn(w_x(x))
    for w_z, w_x in zip(self.w_zs, w_xs):
      z = self.act_fn(w_z(z) + w_x(x))
    z = z + self.pos_def_potentials(x)

    return z.squeeze()

  def _get_wz(self, dim: int) -> nn.Module:
    if self.pos_weights:
      return layers.PositiveDense(
          dim,
          kernel_init=self.init_fn,
          use_bias=False,
          rectifier_fn=self.rectifier_fn,
      )

    return nn.Dense(
        dim,
        kernel_init=self.init_fn,
        use_bias=False,
    )

  def _get_wx(self, dim: int) -> nn.Module:
    return layers.PosDefPotentials(
        rank=self.rank,
        num_potentials=dim,
        use_linear=True,
        use_bias=True,
        kernel_diag_init=nn.initializers.zeros,
        kernel_lr_init=self.init_fn,
        kernel_linear_init=self.init_fn,
        # TODO(michalk8): why us this for bias?
        bias_init=self.init_fn,
    )

  def _get_pos_def_potentials(self) -> layers.PosDefPotentials:
    kwargs = {
        "num_potentials": 1,
        "use_linear": True,
        "use_bias": True,
        "bias_init": nn.initializers.zeros
    }

    if self.gaussian_map_samples is None:
      return layers.PosDefPotentials(
          rank=self.rank,
          kernel_diag_init=nn.initializers.ones,
          kernel_lr_init=nn.initializers.zeros,
          kernel_linear_init=nn.initializers.zeros,
          **kwargs,
      )

    source, target = self.gaussian_map_samples
    return layers.PosDefPotentials.init_from_samples(
        source,
        target,
        rank=self.dim_data,
        # TODO(michalk8): double check if this makes sense
        kernel_diag_init=nn.initializers.zeros,
        **kwargs,
    )


class MLP(neuraldual.BaseW2NeuralDual):
  """A generic, not-convex MLP.

  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The output
      dimension of the last layer is automatically set to 1 if
      :attr:`is_potential` is ``True``, or the dimension of the input otherwise
    is_potential: Model the potential if ``True``, otherwise
      model the gradient of the potential
    act_fn: Activation function
  """

  dim_hidden: Sequence[int]
  is_potential: bool = True
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
    squeeze = x.ndim == 1
    if squeeze:
      x = jnp.expand_dims(x, 0)
    assert x.ndim == 2, x.ndim
    n_input = x.shape[-1]

    z = x
    for n_hidden in self.dim_hidden:
      Wx = nn.Dense(n_hidden, use_bias=True)
      z = self.act_fn(Wx(z))

    if self.is_potential:
      Wx = nn.Dense(1, use_bias=True)
      z = Wx(z).squeeze(-1)

      quad_term = 0.5 * jax.vmap(jnp.dot)(x, x)
      z += quad_term
    else:
      Wx = nn.Dense(n_input, use_bias=True)
      z = x + Wx(z)

    return z.squeeze(0) if squeeze else z


@jax.tree_util.register_pytree_node_class
class MetaInitializer(lin_init.DefaultInitializer):
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
    self.dtype = geom.x.dtype
    self.opt = opt
    self.rng = utils.default_prng_key(rng)

    na, nb = geom.shape
    self.meta_model = meta_model

    if state is None:
      # Initialize the model's training state.
      a_placeholder = jnp.zeros(na, dtype=self.dtype)
      b_placeholder = jnp.zeros(nb, dtype=self.dtype)
      params = self.meta_model.init(self.rng, a_placeholder,
                                    b_placeholder)["params"]
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
      ot_prob: "linear_problem.LinearProblem",
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
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

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
    return self.meta_model.apply({"params": params}, a, b)

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
    return [self.geom, self.meta_model, self.opt], {
        "rng": self.rng,
        "state": self.state
    }
