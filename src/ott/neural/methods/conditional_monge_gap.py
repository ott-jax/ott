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
import logging
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
from ott.geometry import costs, pointcloud, segment
from ott.neural.networks.conditional_perturbation_network import (
    ConditionalPerturbationNetwork,
)
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

logger = logging.getLogger(__name__)

__all__ = [
    "cmonge_gap_from_samples",
    "ConditionalMongeGapEstimator",
]


def cmonge_gap_from_samples(
    source: jnp.ndarray,
    target: jnp.ndarray,
    condition: jnp.ndarray,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    relative_epsilon: Optional[Literal["mean", "std"]] = None,
    scale_cost: Union[float, Literal["mean", "max_cost", "median"]] = 1.0,
    return_output: bool = False,
    num_segments: Optional[int] = None,
    max_measure_size: Optional[int] = None,
    **kwargs: Any,
) -> Union[float, Tuple[float, jnp.ndarray]]:
  r"""Conditional Monge gap from samples using the segment interface.

  Computes the average Monge gap across conditions:

  .. math::

      \frac{1}{K} \sum_{k=1}^{K} \left[
      \frac{1}{n_k} \sum_{i:\, c_i = k} c(x_i, y_i) -
      W_{c, \varepsilon}\!\bigl(\hat{\rho}_{n_k}^{(k)},\,
      \hat{\nu}_{n_k}^{(k)}\bigr) \right]

  where :math:`W_{c, \varepsilon}` is the
  :term:`entropy-regularized optimal transport` cost.

  This implementation uses :func:`~ott.geometry.segment._segment_interface`
  to pad and ``vmap`` across conditions, making it fully JIT-compatible.

  Args:
      source: samples from first measure, array of shape ``[n, d]``.
      target: samples from second measure, array of shape ``[n, d]``.
          Assumed paired with ``source``, i.e. ``target[i] = T(source[i])``.
      condition: integer array of shape ``[n]`` indicating the condition
          for each source-target pair. Values in ``range(num_segments)``.
      cost_fn: a cost function between two points in dimension :math:`d`.
          If :obj:`None`, :class:`~ott.geometry.costs.SqEuclidean` is used.
      epsilon: regularization parameter. See
          :class:`~ott.geometry.pointcloud.PointCloud`.
      relative_epsilon: when set, ``epsilon`` refers to a fraction of the
          :attr:`~ott.geometry.pointcloud.PointCloud.mean_cost_matrix`.
      scale_cost: option to rescale the cost matrix. Implemented scalings
          are ``'median'``, ``'mean'`` and ``'max_cost'``. Alternatively, a
          float factor can be given to rescale the cost such that
          ``cost_matrix /= scale_cost``.
      return_output: if :obj:`True`, also return per-condition Monge gaps.
      num_segments: number of distinct conditions. Required for JIT.
      max_measure_size: maximum number of points in any single condition
          (used for padding). Required for JIT.
      kwargs: keyword arguments for the
          :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver.

  Returns:
      The average Monge gap across conditions and, when ``return_output``
      is :obj:`True`, a ``[num_segments]`` array of per-condition gaps.
  """
  cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
  dim = source.shape[1]
  padding_vector = cost_fn._padder(dim=dim)

  # Warn if any condition is heavily padded (>10x below max_measure_size),
  # which can cause numerical differences vs non-padded Sinkhorn solves.
  # Skipped silently under JIT where condition values are traced.
  if max_measure_size is not None and num_segments is not None:
    try:
      counts = jnp.bincount(condition, length=num_segments)
      min_count = int(jnp.min(counts))
      if min_count > 0 and max_measure_size // min_count >= 10:
        logger.warning(
            "Condition with %d samples will be padded to %d "
            "(%.0fx). Per-condition Monge gap values may differ "
            "from non-padded monge_gap_from_samples calls.",
            min_count,
            max_measure_size,
            max_measure_size / min_count,
        )
    except jax.errors.ConcretizationTypeError:
      pass

  # NOTE: Eval function takes some logic from:
  # ott.neural.methods.monge_gap.monge_gap_from_samples`
  # as well as `ott.geometry.segment.py`
  def eval_fn(
      padded_x: jnp.ndarray,
      padded_y: jnp.ndarray,
      padded_weight_x: jnp.ndarray,
      padded_weight_y: jnp.ndarray,
  ) -> jnp.ndarray:
    """Monge gap for a single (padded) condition segment."""
    # Displacement cost: weighted mean of pairwise costs c(x_i, T(x_i)).
    # Padded entries have weight 0, so they do not contribute.
    pairwise_costs = jax.vmap(cost_fn)(padded_x, padded_y)
    displacement_cost = jnp.sum(pairwise_costs * padded_weight_x)

    # Entropy-regularized OT cost W_{c,ε}.
    geom = pointcloud.PointCloud(
        padded_x,
        padded_y,
        cost_fn=cost_fn,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon,
        scale_cost=scale_cost,
    )
    prob = linear_problem.LinearProblem(
        geom, a=padded_weight_x, b=padded_weight_y
    )
    solver = sinkhorn.Sinkhorn(**kwargs)
    out = solver(prob)

    return displacement_cost - out.ent_reg_cost

  per_condition_gaps = segment._segment_interface(
      x=source,
      y=target,
      eval_fn=eval_fn,
      num_segments=num_segments,
      max_measure_size=max_measure_size,
      segment_ids_x=condition,
      segment_ids_y=condition,
      indices_are_sorted=False,
      padding_vector=padding_vector,
  )

  avg_gap = jnp.mean(per_condition_gaps)
  return (avg_gap, per_condition_gaps) if return_output else avg_gap


class ConditionalMongeGapEstimator:
  r"""Conditional map estimator between probability measures.

  Estimates a condition-dependent map :math:`T(\cdot, c)` by minimizing:

  .. math::

      \min_\theta \; \Delta\bigl(T_\theta(\cdot, c) \sharp \mu,\, \nu\bigr)
      + \lambda \; R_{\text{cond}}\bigl(T_\theta(\cdot, c) \sharp \rho,\,
      \rho \mid c\bigr)

  where :math:`\Delta` is a fitting loss (e.g.
  :func:`~ott.tools.sinkhorn_divergence.sinkdiv`),
  :math:`R_{\text{cond}}` is the conditional Monge gap regularizer
  :func:`cmonge_gap_from_samples`, and :math:`c` is a condition label.

  This mirrors :class:`~ott.neural.methods.monge_gap.MongeGapEstimator`
  but handles condition-aware maps and per-condition regularization.

  Args:
      dim_data: input dimensionality of the data.
      model: a :class:`~ott.neural.networks.\
conditional_perturbation_network.ConditionalPerturbationNetwork` or any
          ``nn.Module`` whose ``__call__`` signature is ``(x, c)``.
      optimizer: optimizer for the map parameters.
      fitting_loss: callable ``(mapped, target) -> (loss, log)`` that
          measures how well the pushforward matches the target distribution.
      regularizer: callable ``(source, mapped, condition_labels) ->
          (loss, log)`` that computes the conditional Monge gap or similar
          per-condition regularizer.
      regularizer_strength: scalar or schedule for :math:`\lambda`.
      num_train_iters: number of training iterations.
      logging: whether to record train/eval metrics.
      valid_freq: how often to evaluate on the validation set.
      rng: random seed.
  """

  def __init__(
      self,
      dim_data: int,
      model: ConditionalPerturbationNetwork,
      optimizer: Optional[optax.OptState] = None,
      fitting_loss: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                      Tuple[float, Optional[Any]]]] = None,
      regularizer: Optional[Callable[
          [jnp.ndarray, jnp.ndarray, jnp.ndarray],
          Tuple[float, Optional[Any]],
      ]] = None,
      regularizer_strength: Union[float, Sequence[float]] = 1.0,
      num_train_iters: int = 10_000,
      logging: bool = False,
      valid_freq: int = 500,
      rng: Optional[jax.Array] = None,
  ):
    self._fitting_loss = fitting_loss
    self._regularizer = regularizer
    self.regularizer_strength = jnp.repeat(
        jnp.atleast_2d(regularizer_strength),
        num_train_iters,
        total_repeat_length=num_train_iters,
        axis=0,
    ).ravel()
    self.num_train_iters = num_train_iters
    self.logging = logging
    self.valid_freq = valid_freq
    self.rng = utils.default_prng_key(rng)

    if optimizer is None:
      optimizer = optax.adam(learning_rate=0.001)

    self.setup(dim_data, model, optimizer)

  def setup(
      self,
      dim_data: int,
      neural_net: ConditionalPerturbationNetwork,
      optimizer: optax.OptState,
  ):
    """Set up all components required to train the network."""
    self.state_neural_net = neural_net.create_train_state(
        self.rng, optimizer, dim_data
    )
    self.step_fn = self._get_step_fn()

  @property
  def regularizer(
      self,
  ) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[float,
                                                               Optional[Any]]]:
    """Conditional regularizer ``(source, mapped, labels) -> (loss, log)``.

    Defaults to zero if not provided.
    """
    if self._regularizer is not None:
      return self._regularizer
    return lambda *_, **__: (0.0, None)

  @property
  def fitting_loss(
      self,
  ) -> Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, Optional[Any]]]:
    """Fitting loss ``(mapped, target) -> (loss, log)``.

    Defaults to zero if not provided.
    """
    if self._fitting_loss is not None:
      return self._fitting_loss
    return lambda *_, **__: (0.0, None)

  @staticmethod
  def _generate_batch(
      loader_source: Iterator[jnp.ndarray],
      loader_target: Iterator[jnp.ndarray],
      loader_condition: Iterator[jnp.ndarray],
      loader_label: Iterator[jnp.ndarray],
  ) -> Dict[str, jnp.ndarray]:
    """Generate a batch of samples from all four iterators."""
    return {
        "source": next(loader_source),
        "target": next(loader_target),
        "condition": next(loader_condition),
        "condition_labels": next(loader_label),
    }

  def train_map_estimator(
      self,
      trainloader_source: Iterator[jnp.ndarray],
      trainloader_target: Iterator[jnp.ndarray],
      trainloader_condition: Iterator[jnp.ndarray],
      trainloader_label: Iterator[jnp.ndarray],
      validloader_source: Iterator[jnp.ndarray],
      validloader_target: Iterator[jnp.ndarray],
      validloader_condition: Iterator[jnp.ndarray],
      validloader_label: Iterator[jnp.ndarray],
  ) -> Tuple[train_state.TrainState, Dict[str, Any]]:
    """Training loop."""
    logs = collections.defaultdict(lambda: collections.defaultdict(list))

    try:
      from tqdm import trange

      tbar = trange(self.num_train_iters, leave=True)
    except ImportError:
      tbar = range(self.num_train_iters)

    for step in tbar:
      is_logging_step = self.logging and ((step % self.valid_freq == 0) or
                                          (step == self.num_train_iters - 1))
      train_batch = self._generate_batch(
          trainloader_source,
          trainloader_target,
          trainloader_condition,
          trainloader_label,
      )
      valid_batch = (
          None if not is_logging_step else self._generate_batch(
              validloader_source,
              validloader_target,
              validloader_condition,
              validloader_label,
          )
      )
      self.state_neural_net, current_logs = self.step_fn(
          self.state_neural_net,
          train_batch,
          valid_batch,
          is_logging_step,
          step,
      )

      if is_logging_step:
        for log_key in current_logs:
          for metric_key in current_logs[log_key]:
            logs[log_key][metric_key].append(current_logs[log_key][metric_key])
        if not isinstance(tbar, range):
          reg_msg = (
              "NA" if current_logs["eval"]["regularizer"] == 0.0 else
              f"{current_logs['eval']['regularizer']:.4f}"
          )
          postfix_str = (
              f"fitting_loss:"
              f" {current_logs['eval']['fitting_loss']:.4f}, "
              f"regularizer: {reg_msg} ,"
              f"total: {current_logs['eval']['total_loss']:.4f}"
          )
          tbar.set_postfix_str(postfix_str)

    return self.state_neural_net, logs

  def _get_step_fn(self) -> Callable:
    """Create a one-step training and evaluation function."""

    def loss_fn(
        params: frozen_dict.FrozenDict,
        apply_fn: Callable,
        batch: Dict[str, jnp.ndarray],
        step: int,
    ) -> Tuple[float, Dict[str, float]]:
      """Loss function with conditional map and regularizer."""
      # Apply the conditional map: T(source, condition)
      mapped_samples = apply_fn({"params": params}, batch["source"],
                                batch["condition"])

      # Fitting loss: Δ(T(x,c), y)
      val_fitting_loss, log_fitting_loss = self.fitting_loss(
          mapped_samples, batch["target"]
      )

      # Conditional regularizer: R(x, T(x,c), labels)
      val_regularizer, log_regularizer = self.regularizer(
          batch["source"], mapped_samples, batch["condition_labels"]
      )

      val_tot_loss = (
          val_fitting_loss + self.regularizer_strength[step] * val_regularizer
      )

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
        step: int = 0,
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
      """One step function."""
      grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
      (_, current_train_logs), grads = grad_fn(
          state_neural_net.params,
          state_neural_net.apply_fn,
          train_batch,
          step,
      )

      current_logs = {"train": current_train_logs, "eval": {}}
      if is_logging_step:
        _, current_eval_logs = loss_fn(
            params=state_neural_net.params,
            apply_fn=state_neural_net.apply_fn,
            batch=valid_batch,
            step=step,
        )
        current_logs["eval"] = current_eval_logs

      return state_neural_net.apply_gradients(grads=grads), current_logs

    return step_fn
