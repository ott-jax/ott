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
import collections
import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp

import optax
from flax.core import frozen_dict
from flax.training import train_state

from ott import utils
from ott.geometry import costs, pointcloud
from ott.neural.networks import potentials
from ott.solvers import linear
from ott.solvers.linear import sinkhorn

__all__ = ["monge_gap", "monge_gap_from_samples", "MongeGapEstimator"]


def monge_gap(
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    reference_points: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    relative_epsilon: Optional[bool] = None,
    scale_cost: Union[int, float, Literal["mean", "max_cost", "median"]] = 1.0,
    return_output: bool = False,
    **kwargs: Any
) -> Union[float, Tuple[float, sinkhorn.SinkhornOutput]]:
  r"""Monge gap regularizer :cite:`uscidda:23`.

  For a cost function :math:`c` and empirical reference measure
  :math:`\hat{\rho}_n=\frac{1}{n}\sum_{i=1}^n \delta_{x_i}`, the
  (entropic) Monge gap of a map function
  :math:`T:\mathbb{R}^d\rightarrow\mathbb{R}^d` is defined as:

  .. math::
    \mathcal{M}^c_{\hat{\rho}_n, \varepsilon} (T)
    = \frac{1}{n} \sum_{i=1}^n c(x_i, T(x_i)) -
    W_{c, \varepsilon}(\hat{\rho}_n, T \sharp \hat{\rho}_n)

  See :cite:`uscidda:23` Eq. (8). This function is a thin wrapper that calls
  :func:`~ott.neural.methods.monge_gap.monge_gap_from_samples`.

  Args:
    map_fn: Callable corresponding to map :math:`T` in definition above. The
      callable should be vectorized (e.g. using :func:`~jax.vmap`), i.e,
      able to process a *batch* of vectors of size `d`, namely
      ``map_fn`` applied to an array returns an array of the same shape.
    reference_points: Array of `[n,d]` points, :math:`\hat\rho_n`.
    cost_fn: An object of class :class:`~ott.geometry.costs.CostFn`.
    epsilon: Regularization parameter. See
      :class:`~ott.geometry.pointcloud.PointCloud`
    relative_epsilon: when `False`, the parameter ``epsilon`` specifies the
      value of the entropic regularization parameter. When `True`, ``epsilon``
      refers to a fraction of the
      :attr:`~ott.geometry.pointcloud.PointCloud.mean_cost_matrix`, which is
      computed adaptively using ``source`` and ``target`` points.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean' and 'max_cost'. Alternatively, a float factor can be
      given to rescale the cost such that ``cost_matrix /= scale_cost``.
    return_output: boolean to also return the
      :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`.
    kwargs: holds the kwargs to instantiate the or
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver to
      compute the regularized OT cost.

  Returns:
    The Monge gap value and optionally the
    :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
  """
  target = map_fn(reference_points)
  return monge_gap_from_samples(
      source=reference_points,
      target=target,
      cost_fn=cost_fn,
      epsilon=epsilon,
      relative_epsilon=relative_epsilon,
      scale_cost=scale_cost,
      return_output=return_output,
      **kwargs
  )


def monge_gap_from_samples(
    source: jnp.ndarray,
    target: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    relative_epsilon: Optional[bool] = None,
    scale_cost: Union[int, float, Literal["mean", "max_cost", "median"]] = 1.0,
    return_output: bool = False,
    **kwargs: Any
) -> Union[float, Tuple[float, sinkhorn.SinkhornOutput]]:
  r"""Monge gap, instantiated in terms of samples before / after applying map.

  .. math::
    \frac{1}{n} \sum_{i=1}^n c(x_i, y_i)) -
    W_{c, \varepsilon}(\frac{1}{n}\sum_i \delta_{x_i},
    \frac{1}{n}\sum_i \delta_{y_i})

  where :math:`W_{c, \varepsilon}` is an entropy-regularized optimal transport
  cost, the :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.ent_reg_cost`.

  Args:
    source: samples from first measure, array of shape ``[n, d]``.
    target: samples from second measure, array of shape ``[n, d]``.
    cost_fn: a cost function between two points in dimension :math:`d`.
      If :obj:`None`, :class:`~ott.geometry.costs.SqEuclidean` is used.
    epsilon: Regularization parameter. See
      :class:`~ott.geometry.pointcloud.PointCloud`
    relative_epsilon: when `False`, the parameter ``epsilon`` specifies the
      value of the entropic regularization parameter. When `True`, ``epsilon``
      refers to a fraction of the
      :attr:`~ott.geometry.pointcloud.PointCloud.mean_cost_matrix`, which is
      computed adaptively using ``source`` and ``target`` points.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean' and 'max_cost'. Alternatively, a float factor can be
      given to rescale the cost such that ``cost_matrix /= scale_cost``.
    return_output: boolean to also return the
      :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`.
    kwargs: holds the kwargs to instantiate the or
      :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver to
      compute the regularized OT cost.

  Returns:
    The Monge gap value and optionally the
    :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
  """
  cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
  geom = pointcloud.PointCloud(
      x=source,
      y=target,
      cost_fn=cost_fn,
      epsilon=epsilon,
      relative_epsilon=relative_epsilon,
      scale_cost=scale_cost,
  )
  gt_displacement_cost = jnp.mean(jax.vmap(cost_fn)(source, target))
  out = linear.solve(geom=geom, **kwargs)
  loss = gt_displacement_cost - out.ent_reg_cost
  return (loss, out) if return_output else loss


class MongeGapEstimator:
  r"""Mapping estimator between probability measures.

  It estimates a map :math:`T` by minimizing the loss:

  .. math::
    \text{min}_{\theta}\; \Delta(T_\theta \sharp \mu, \theta)
    + \lambda R(T_\theta \sharp \rho, \rho)

  where :math:`\Delta` is a fitting loss and :math:`R` is a regularizer.
  :math:`\Delta` allows to fit the marginal constraint, i.e. transport
  :math:`\mu` to  :math:`\nu` via :math:`T`, while :math:`R`
  is a regularizer imposing an inductive bias on the learned map. The
  regularizer in this case is a function used to compute a metric between two
  sets of points.

  For instance, :math:`\Delta` can be the
  :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`
  and :math:`R` the :func:`~ott.neural.methods.monge_gap.monge_gap_from_samples`
  :cite:`uscidda:23` for a given cost function :math:`c`.
  In that case, it estimates a :math:`c`-OT map, i.e. a map :math:`T`
  optimal for the Monge problem induced by :math:`c`.

  Args:
    dim_data: input dimensionality of data required for network init.
    model: network architecture for map :math:`T`.
    optimizer: optimizer function for map :math:`T`.
    fitting_loss: function that outputs a fitting loss :math:`\Delta` between
      two families of points, as well as any log object.
    regularizer: function that outputs a score from two families of points,
      here assumed to be of the same size, as well as any log object.
    regularizer_strength: strength of the :attr:`regularizer`.
    num_train_iters: number of total training iterations.
    logging: option to return logs.
    valid_freq: frequency with training and validation are logged.
    rng: random key used for seeding for network initializations.
  """

  def __init__(
      self,
      dim_data: int,
      model: potentials.BasePotential,
      optimizer: Optional[optax.OptState] = None,
      fitting_loss: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                      Tuple[float, Optional[Any]]]] = None,
      regularizer: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                     Tuple[float, Optional[Any]]]] = None,
      regularizer_strength: Union[float, Sequence[float]] = 1.0,
      num_train_iters: int = 10_000,
      logging: bool = False,
      valid_freq: int = 500,
      rng: Optional[jax.Array] = None,
  ):
    self._fitting_loss = fitting_loss
    self._regularizer = regularizer
    # Can use either a fixed strength, or generalize to a schedule.
    self.regularizer_strength = jnp.repeat(
        jnp.atleast_2d(regularizer_strength),
        num_train_iters,
        total_repeat_length=num_train_iters,
        axis=0
    ).ravel()
    self.num_train_iters = num_train_iters
    self.logging = logging
    self.valid_freq = valid_freq
    self.rng = utils.default_prng_key(rng)

    # set default optimizer
    if optimizer is None:
      optimizer = optax.adam(learning_rate=0.001)

    # setup training
    self.setup(dim_data, model, optimizer)

  def setup(
      self,
      dim_data: int,
      neural_net: potentials.BasePotential,
      optimizer: optax.OptState,
  ):
    """Setup all components required to train the network."""
    # neural network
    self.state_neural_net = neural_net.create_train_state(
        self.rng, optimizer, dim_data
    )

    # step function
    self.step_fn = self._get_step_fn()

  @property
  def regularizer(self) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
    """Regularizer added to the fitting loss.

    Can be, e.g. the :func:`~ott.neural.methods.monge_gap.monge_gap_from_samples`.
    If no regularizer is passed for solver instantiation,
    or regularization weight :attr:`regularizer_strength` is 0,
    return 0 by default along with an empty set of log values.
    """  # noqa: E501
    if self._regularizer is not None:
      return self._regularizer
    return lambda *_, **__: (0.0, None)

  @property
  def fitting_loss(self) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
    """Fitting loss to fit the marginal constraint.

    Can be, e.g. :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.
    If no fitting_loss is passed for solver instantiation, return 0 by default,
    and no log values.
    """
    if self._fitting_loss is not None:
      return self._fitting_loss
    return lambda *_, **__: (0.0, None)

  @staticmethod
  def _generate_batch(
      loader_source: Iterator[jnp.ndarray],
      loader_target: Iterator[jnp.ndarray],
  ) -> Dict[str, jnp.ndarray]:
    """Generate batches a batch of samples.

    ``loader_source`` and ``loader_target`` can be training or
    validation dataloaders.
    """
    return {
        "source": next(loader_source),
        "target": next(loader_target),
    }

  def train_map_estimator(
      self,
      trainloader_source: Iterator[jnp.ndarray],
      trainloader_target: Iterator[jnp.ndarray],
      validloader_source: Iterator[jnp.ndarray],
      validloader_target: Iterator[jnp.ndarray],
  ) -> Tuple[train_state.TrainState, Dict[str, Any]]:
    """Training loop."""
    # define logs
    logs = collections.defaultdict(lambda: collections.defaultdict(list))

    # try to display training progress with tqdm
    try:
      from tqdm import trange
      tbar = trange(self.num_train_iters, leave=True)
    except ImportError:
      tbar = range(self.num_train_iters)

    for step in tbar:
      #  update step
      is_logging_step = (
          self.logging and ((step % self.valid_freq == 0) or
                            (step == self.num_train_iters - 1))
      )
      train_batch = self._generate_batch(
          loader_source=trainloader_source,
          loader_target=trainloader_target,
      )
      valid_batch = (
          None if not is_logging_step else self._generate_batch(
              loader_source=validloader_source,
              loader_target=validloader_target,
          )
      )
      self.state_neural_net, current_logs = self.step_fn(
          self.state_neural_net, train_batch, valid_batch, is_logging_step, step
      )

      # store and print metrics if logging step
      if is_logging_step:
        for log_key in current_logs:
          for metric_key in current_logs[log_key]:
            logs[log_key][metric_key].append(current_logs[log_key][metric_key])

        # update the tqdm bar if tqdm is available
        if not isinstance(tbar, range):
          reg_msg = (
              "NA" if current_logs["eval"]["regularizer"] == 0.0 else
              f"{current_logs['eval']['regularizer']:.4f}"
          )
          postfix_str = (
              f"fitting_loss: {current_logs['eval']['fitting_loss']:.4f}, "
              f"regularizer: {reg_msg} ,"
              f"total: {current_logs['eval']['total_loss']:.4f}"
          )
          tbar.set_postfix_str(postfix_str)

    return self.state_neural_net, logs

  def _get_step_fn(self) -> Callable:
    """Create a one step training and evaluation function."""

    def loss_fn(
        params: frozen_dict.FrozenDict, apply_fn: Callable,
        batch: Dict[str, jnp.ndarray], step: int
    ) -> Tuple[float, Dict[str, float]]:
      """Loss function."""
      # map samples with the fitted map
      mapped_samples = apply_fn({"params": params}, batch["source"])

      # compute the loss
      val_fitting_loss, log_fitting_loss = self.fitting_loss(
          mapped_samples, batch["target"]
      )
      val_regularizer, log_regularizer = self.regularizer(
          batch["source"], mapped_samples
      )
      val_tot_loss = (
          val_fitting_loss + self.regularizer_strength[step] * val_regularizer
      )

      # store training logs
      loss_logs = {
          "total_loss": val_tot_loss,
          "fitting_loss": val_fitting_loss,
          "regularizer": val_regularizer,
          "log_regularizer": log_regularizer,
          "log_fitting": log_fitting_loss,
      }

      return val_tot_loss, loss_logs

    @functools.partial(jax.jit, static_argnums=3)
    def step_fn(
        state_neural_net: train_state.TrainState,
        train_batch: Dict[str, jnp.ndarray],
        valid_batch: Optional[Dict[str, jnp.ndarray]] = None,
        is_logging_step: bool = False,
        step: int = 0
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
      """One step function."""
      # compute loss and gradients
      grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
      (_, current_train_logs), grads = grad_fn(
          state_neural_net.params, state_neural_net.apply_fn, train_batch, step
      )

      # logging step
      current_logs = {"train": current_train_logs, "eval": {}}
      if is_logging_step:
        _, current_eval_logs = loss_fn(
            params=state_neural_net.params,
            apply_fn=state_neural_net.apply_fn,
            batch=valid_batch,
            step=step
        )
        current_logs["eval"] = current_eval_logs

      # update state
      return state_neural_net.apply_gradients(grads=grads), current_logs

    return step_fn
