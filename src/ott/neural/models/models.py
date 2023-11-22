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
import abc
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import frozen_dict
from flax.training import train_state
from jax.nn import initializers

from ott import utils
from ott.geometry import geometry
from ott.initializers.linear import initializers as lin_init
from ott.math import matrix_square_root
from ott.neural.models import layers
from ott.neural.solvers import neuraldual
from ott.problems.linear import linear_problem

__all__ = ["ICNN", "MLP", "MetaInitializer"]


class ICNN(neuraldual.BaseW2NeuralDual):
  """Input convex neural network (ICNN) architecture with initialization.

  Implementation of input convex neural networks as introduced in
  :cite:`amos:17` with initialization schemes proposed by :cite:`bunne:22`.

  Args:
    dim_data: data dimensionality.
    dim_hidden: sequence specifying size of hidden dimensions. The
      output dimension of the last layer is 1 by default.
    init_std: value of standard deviation of weight initialization method.
    init_fn: choice of initialization method for weight matrices (default:
      :func:`jax.nn.initializers.normal`).
    act_fn: choice of activation function used in network architecture
      (needs to be convex, default: :obj:`jax.nn.relu`).
    pos_weights: Enforce positive weights with a projection.
      If ``False``, the positive weights should be enforced with clipping
      or regularization in the loss.
    gaussian_map_samples: Tuple of source and target points, used to initialize
      the ICNN to mimic the linear Bures map that morphs the (Gaussian
      approximation) of the input measure to that of the target measure. If
      ``None``, the identity initialization is used, and ICNN mimics half the
      squared Euclidean norm.
  """
  dim_data: int
  dim_hidden: Sequence[int]
  init_std: float = 1e-2
  init_fn: Callable = jax.nn.initializers.normal
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  pos_weights: bool = True
  gaussian_map_samples: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None

  @property
  def is_potential(self) -> bool:  # noqa: D102
    return True

  def setup(self) -> None:  # noqa: D102
    self.num_hidden = len(self.dim_hidden)

    if self.pos_weights:
      hid_dense = layers.PositiveDense
      # this function needs to be the inverse map of function
      # used in PositiveDense layers
      rescale = hid_dense.inv_rectifier_fn
    else:
      hid_dense = nn.Dense
      rescale = lambda x: x
    self.use_init = False
    # check if Gaussian map was provided
    if self.gaussian_map_samples is not None:
      factor, mean = self._compute_gaussian_map_params(
          self.gaussian_map_samples
      )
    else:
      factor, mean = self._compute_identity_map_params(self.dim_data)

    w_zs = []
    # keep track of previous size to normalize accordingly
    normalization = 1

    for i in range(1, self.num_hidden):
      w_zs.append(
          hid_dense(
              self.dim_hidden[i],
              kernel_init=initializers.constant(rescale(1.0 / normalization)),
              use_bias=False,
          )
      )
      normalization = self.dim_hidden[i]
    # final layer computes average, still with normalized rescaling
    w_zs.append(
        hid_dense(
            1,
            kernel_init=initializers.constant(rescale(1.0 / normalization)),
            use_bias=False,
        )
    )
    self.w_zs = w_zs

    # positive definite potential (the identity mapping or linear OT)
    self.pos_def_potential = layers.PosDefPotentials(
        self.dim_data,
        num_potentials=1,
        kernel_init=lambda *_: factor,
        bias_init=lambda *_: mean,
        use_bias=True,
    )

    # subsequent layers re-injected into convex functions
    w_xs = []
    for i in range(self.num_hidden):
      w_xs.append(
          nn.Dense(
              self.dim_hidden[i],
              kernel_init=self.init_fn(self.init_std),
              bias_init=initializers.constant(0.),
              use_bias=True,
          )
      )
    # final layer, to output number
    w_xs.append(
        nn.Dense(
            1,
            kernel_init=self.init_fn(self.init_std),
            bias_init=initializers.constant(0.),
            use_bias=True,
        )
    )
    self.w_xs = w_xs

  @staticmethod
  def _compute_gaussian_map_params(
      samples: Tuple[jnp.ndarray, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    from ott.tools.gaussian_mixture import gaussian
    source, target = samples
    # print(source)
    # print(type(source))
    g_s = gaussian.Gaussian.from_samples(source)
    g_t = gaussian.Gaussian.from_samples(target)
    lin_op = g_s.scale.gaussian_map(g_t.scale)
    b = jnp.squeeze(g_t.loc) - jnp.linalg.solve(lin_op, jnp.squeeze(g_t.loc))
    lin_op = matrix_square_root.sqrtm_only(lin_op)
    return jnp.expand_dims(lin_op, 0), jnp.expand_dims(b, 0)

  @staticmethod
  def _compute_identity_map_params(
      input_dim: int
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A = jnp.eye(input_dim).reshape((1, input_dim, input_dim))
    b = jnp.zeros((1, input_dim))
    return A, b

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> float:  # noqa: D102
    z = self.act_fn(self.w_xs[0](x))
    for i in range(self.num_hidden):
      z = jnp.add(self.w_zs[i](z), self.w_xs[i + 1](x))
      z = self.act_fn(z)
    z += self.pos_def_potential(x)
    return z.squeeze()


class MLP(neuraldual.BaseW2NeuralDual):
  """A generic, typically not-convex (w.r.t input) MLP.

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
      TODO(marcocuturi): add explanation here what arguments to expect.
    opt: The optimizer to update the parameters. If ``None``, use
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
      state: Optional[train_state.TrainState] = None
  ):
    self.geom = geom
    self.dtype = geom.x.dtype
    self.opt = opt
    self.rng = utils.default_prng_key(rng)

    na, nb = geom.shape
    # TODO(michalk8): add again some default MLP
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
      g_pred = jnp.where(jnp.isfinite(g_pred), g_pred, 0.)

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


class Block(nn.Module):
  dim: int = 128
  num_layers: int = 3
  act_fn: Any = nn.silu
  out_dim: int = 32

  @nn.compact
  def __call__(self, x):
    for i in range(self.num_layers):
      x = nn.Dense(self.dim, name="fc{0}".format(i))(x)
      x = self.act_fn(x)
    return nn.Dense(self.out_dim)(x)


class BaseNeuralVectorField(nn.Module, abc.ABC):

  @abc.abstractmethod
  def __call__(
      self,
      t: jax.Array,
      x: jax.Array,
      condition: Optional[jax.Array] = None,
      keys_model: Optional[jax.Array] = None
  ) -> jnp.ndarray:  # noqa: D102):
    pass


class NeuralVectorField(BaseNeuralVectorField):
  output_dim: int
  condition_dim: int
  latent_embed_dim: int
  condition_embed_dim: Optional[int] = None
  t_embed_dim: Optional[int] = None
  joint_hidden_dim: Optional[int] = None
  num_layers_per_block: int = 3
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
  n_frequencies: int = 128

  def time_encoder(self, t: jax.Array) -> jnp.array:
    freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
    t = freq * t
    return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

  def __post_init__(self):

    # set embedded dim from latent embedded dim
    if self.condition_embed_dim is None:
      self.condition_embed_dim = self.latent_embed_dim
    if self.t_embed_dim is None:
      self.t_embed_dim = self.latent_embed_dim

    # set joint hidden dim from all embedded dim
    concat_embed_dim = (
        self.latent_embed_dim + self.condition_embed_dim + self.t_embed_dim
    )
    if self.joint_hidden_dim is not None:
      assert (self.joint_hidden_dim >= concat_embed_dim), (
          "joint_hidden_dim must be greater than or equal to the sum of "
          "all embedded dimensions. "
      )
      self.joint_hidden_dim = self.latent_embed_dim
    else:
      self.joint_hidden_dim = concat_embed_dim
    super().__post_init__()

  @nn.compact
  def __call__(
      self,
      t: jax.Array,
      x: jax.Array,
      condition: Optional[jax.Array],
      keys_model: Optional[jax.Array] = None,
  ) -> jax.Array:

    t = self.time_encoder(t)
    t = Block(
        dim=self.t_embed_dim,
        out_dim=self.t_embed_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn,
    )(
        t
    )

    x = Block(
        dim=self.latent_embed_dim,
        out_dim=self.latent_embed_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn
    )(
        x
    )

    if self.condition_dim > 0:
      condition = Block(
          dim=self.condition_embed_dim,
          out_dim=self.condition_embed_dim,
          num_layers=self.num_layers_per_block,
          act_fn=self.act_fn
      )(
          condition
      )
      concatenated = jnp.concatenate((t, x, condition), axis=-1)
    else:
      concatenated = jnp.concatenate((t, x), axis=-1)

    out = Block(
        dim=self.joint_hidden_dim,
        out_dim=self.joint_hidden_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn,
    )(
        concatenated
    )

    return nn.Dense(
        self.output_dim,
        use_bias=True,
    )(
        out
    )

  def create_train_state(
      self,
      rng: jax.random.PRNGKeyArray,
      optimizer: optax.OptState,
      input_dim: int,
  ) -> train_state.TrainState:
    params = self.init(
        rng, jnp.ones((1, 1)), jnp.ones((1, input_dim)),
        jnp.ones((1, self.condition_dim))
    )["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )


class BaseRescalingNet(nn.Module, abc.ABC):

  @abc.abstractmethod
  def __call___(
      self, x: jax.Array, condition: Optional[jax.Array] = None
  ) -> jax.Array:
    pass


class Rescaling_MLP(nn.Module):
  hidden_dim: int
  cond_dim: int
  is_potential: bool = False
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.selu

  @nn.compact
  def __call__(
      self, x: jnp.ndarray, condition: Optional[jax.Array]
  ) -> jnp.ndarray:  # noqa: D102
    x = Block(
        dim=self.latent_embed_dim,
        out_dim=self.latent_embed_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn
    )(
        x
    )
    if self.condition_dim > 0:
      condition = Block(
          dim=self.condition_embed_dim,
          out_dim=self.condition_embed_dim,
          num_layers=self.num_layers_per_block,
          act_fn=self.act_fn
      )(
          condition
      )
      concatenated = jnp.concatenate((x, condition), axis=-1)
    else:
      concatenated = x

    out = Block(
        dim=self.joint_hidden_dim,
        out_dim=self.joint_hidden_dim,
        num_layers=self.num_layers_per_block,
        act_fn=self.act_fn,
    )(
        concatenated
    )

    return jnp.exp(out)

  def create_train_state(
      self,
      rng: jax.random.PRNGKeyArray,
      optimizer: optax.OptState,
      input_dim: int,
  ) -> train_state.TrainState:
    params = self.init(
        rng, jnp.ones((1, input_dim)), jnp.ones((1, self.cond_dim))
    )["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )
