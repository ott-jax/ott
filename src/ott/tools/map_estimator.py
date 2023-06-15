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

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from tqdm import trange

from ott.solvers.nn import models


class MapEstimator:
  r"""Mapping estimator between probability measures.

  It estimates a map :math:`T` by minimizing the loss:
  .. math::

  \text{min}_{\theta}\; \Delta(T_\theta \sharp \mu, \theta)
  + \lambda R(T_\theta)

  where :math:`\Delta` is a fitting loss and :math:`R` is a regularizer.
  :math:`\Delta` allows to fit the marginal constraint, i.e. transport
  :math:`\mu` to  :math:`\nu` via :math:`T`, while :math:`R`
  is a regularizer imposing an inductive bias on the learned map.

  For instance, :math:`\Delta` can be the
  {function}`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`
  and :math:`R` the {function}`~ott.solvers.nn.losses.monge_gap`
  (see :cite:`uscidda:23) for a given cost function :math:`c`.
  In that case, it estimates a :math:`c`-OT map, i.e. a map :math:`T`
  optimal for the Monge problem induced by :math:`c`.

  Args:
  dim_data: input dimensionality of data required for network init
  model: network architecture for map :math:`T`.
  optimizer: optimizer function for map :math:`T`.
  fitting_loss: fitting loss :math:`Delta` to fit the marginal constraint
  regularizer: regularizer :math:`R` to impose an inductive bias
  on the map :math:`T`
  num_train_iters: number of total training iterations
  num_inner_iters: number of training iterations of :math:`g` per iteration
  valid_freq: frequency with training and validation are logged
  logging: option to return logs
  rng: random key used for seeding for network initializations
  """

  def __init__(
      self,
      dim_data: int,
      model: models.ModelBase,
      optimizer: optax.OptState,
      fitting_loss: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                      float]] = None,
      regularizer: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = None,
      lambd_regularizer: float = 1.,
      num_train_iters: int = 10_000,
      logging: bool = False,
      valid_freq: int = 500,
      rng: Optional[jax.random.PRNGKey] = None,
  ):
    self._fitting_loss = fitting_loss
    self._regularizer = regularizer
    self.lambd_regularizer = lambd_regularizer
    self.num_train_iters = num_train_iters
    self.logging = logging
    self.valid_freq = valid_freq
    if rng is not None:
      self.rng = rng
    else:
      self.rng = jax.random.PRNGKey(0)
    self.setup(dim_data, model, optimizer)

  def setup(
      self,
      dim_data,
      neural_net,
      optimizer,
  ) -> None:
    """Setup all components required to train the network."""
    # neural network
    self.state_neural_net = neural_net.create_train_state(
        self.rng, optimizer, dim_data
    )

    # step function
    self.step_fn = self._get_step_fn()

    return

  @property
  def regularizer(self) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
    """Regularizer added to the fitting loss.

    Can be for instance the {function}`~ott.solvers.nn.losses.monge_gap`.
    If no ``regularizer`` is passed for solver instanciation,
    or regularization weight ``lambd_regularizer``is 0.,
    return 0. by default.
    In that case, only the ``fitting loss`` is minimized.
    """
    if self._regularizer is not None:
      return self._regularizer
    return lambda *args, **kwargs: 0.

  @property
  def fitting_loss(self) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
    """Fitting loss to fit the marginal constraint.

    Can be for instance the
    {function}`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.
    If no ``fitting_loss` is passed for solver instanciation,
    return 0. by default.
    In that case, only the ``regularizer`` is minimized.
    """
    if self._fitting_loss is not None:
      return self._fitting_loss
    return lambda *args, **kwargs: 0.

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
    logs = defaultdict(lambda: defaultdict(list))

    tbar = trange(self.num_train_iters, leave=True)
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
          self.state_neural_net, train_batch, valid_batch, is_logging_step
      )

      # store and print metrics if logging step
      if is_logging_step:
        for log_key in logs:
          for metric_key in logs[log_key]:
            logs[log_key][metric_key].append(current_logs[log_key][metric_key])
        reg_msg = (
            "not computed" if current_logs["eval"]["regularizer"] == 0. else
            f"{current_logs['eval']['regularizer']:.4f}"
        )
        postfix_str = (
            f"fitting_loss: {current_logs['eval']['fitting_loss']:.4f}, "
            f"regularizer: {reg_msg}."
        )
        tbar.set_postfix_str(postfix_str)

    return self.state_neural_net, logs

  def _get_step_fn(self) -> Callable:
    """Create a one step training and evaluation function."""

    def loss_fn(
        params: FrozenDict,
        apply_fn: Callable,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
      """Loss function."""
      # map samples with the fitted map
      mapped_samples = apply_fn({"params": params}, batch["source"])

      # compute the loss
      val_fitting_loss = self.fitting_loss(
          samples=batch["target"], mapped_samples=mapped_samples
      )
      val_regularizer = self.regularizer(
          samples=batch["source"], mapped_samples=mapped_samples
      )
      val_tot_loss = (
          val_fitting_loss + self.lambd_regularizer * val_regularizer
      )

      # store training logs
      loss_logs = {
          "total_loss": val_tot_loss,
          "fitting_loss": val_fitting_loss,
          "regularizer": val_regularizer
      }

      return val_tot_loss, loss_logs

    @partial(jax.jit, static_argnums=3)
    def step_fn(
        state_neural_net: train_state.TrainState,
        train_batch: Dict[str, jnp.ndarray],
        valid_batch: Optional[Dict[str, jnp.ndarray]] = None,
        is_logging_step: bool = False,
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
      """Step function."""
      # compute loss and gradients
      grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
      (_, current_train_logs), grads = grad_fn(
          state_neural_net.params, state_neural_net.apply_fn, train_batch
      )

      # logging step
      current_logs = {"train": current_train_logs, "eval": {}}
      if is_logging_step:
        _, current_eval_logs = loss_fn(
            params=state_neural_net.params,
            apply_fn=state_neural_net.apply_fn,
            batch=valid_batch
        )
        current_logs["eval"] = current_eval_logs

      # update state
      return state_neural_net.apply_gradients(grads=grads), current_logs

    return step_fn
