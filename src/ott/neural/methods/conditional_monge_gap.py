import collections
import functools

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import frozen_dict
from flax.training import train_state
from flax.training.orbax_utils import save_args_from_target
from jax.tree_util import tree_map
from orbax.checkpoint import PyTreeCheckpointer
from ott.geometry import costs, pointcloud
from ott.solvers import linear
from ott.solvers.linear import sinkhorn
from ott.neural.networks.conditional_perturbation_network import (
    ConditionalPerturbationNetwork,
)

T = TypeVar("T", bound="ConditionalMongeGapEstimator")


def cmonge_gap_from_samples(
    source: jnp.ndarray,
    target: jnp.ndarray,
    target_condition: jnp.ndarray,
    source_condition: Optional[jnp.ndarray],
    equal_conditions: bool = False,
    cost_fn: Optional[costs.CostFn] = None,
    epsilon: Optional[float] = None,
    relative_epsilon: Optional[Literal["mean", "std"]] = None,
    scale_cost: Union[float, Literal["mean", "max_cost", "median"]] = 1.0,
    return_output: bool = False,
    rng: Optional[jax.Array] = None,
    **kwargs: Any,
) -> Union[float, Tuple[float, sinkhorn.SinkhornOutput]]:
    r"""Monge gap, instantiated in terms of samples before / after applying map.

    .. math::
    \sum_{i=1}{K} \frac{1}{n} \sum_{i=1}^n c(x_i, y_i)) -
    W_{c, \varepsilon}(\frac{1}{n}\sum_i \delta_{x_i},
    \frac{1}{n}\sum_i \delta_{y_i})

    where :math:`W_{c, \varepsilon}` is an
    :term:`entropy-regularized optimal transport`
    cost, the :attr:`~ott.solvers.linear.sinkhorn.SinkhornOutput.ent_reg_cost`.

    Args:
    source: samples from first measure, array of shape ``[n, d]``.
    target: samples from second measure, array of shape ``[n, d]``.
    target_condition: array indicating condition to which each target sample
        belongs, `integer array of shape ``[n]``.
    source_condition: array indicating condition to which each source sample
        belongs, `integer array of shape ``[n]``.
        If `equal_condition` is `None` and  `source_condition` is `False`
        per condition same number of source cells are sampled as there are
        target cells.
    equal_conditions: whether source and target samples come from the same
        (order)  of conditions. In this case `target_conditions` is used
        for both.
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
    rng: random key used for sampling source samples.
    kwargs: holds the kwargs to instantiate the or
        :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` solver to
        compute the regularized OT cost.

    Returns:
    The average Monge gap value over all conditions and optionally the
    list of Monge gap per condition and :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
    """
    key = jax.random.PRNGKey(rng)
    cost_fn = costs.SqEuclidean() if cost_fn is None else cost_fn
    all_losses = []
    all_outs = []
    for c in jnp.unique(target_condition):
        key, _ = jax.random.split(key, 2)
        c_target = target[target_condition == c]
        if equal_conditions:
            c_source = source[target_condition == c]
        elif source_condition:
            c_source = source[source_condition == c]
        else:
            c_source = jax.random.choice(
                key,
                len(source),
                size=len(c_target),
                replace=len(c_target) > len(source),
            )
        geom = pointcloud.PointCloud(
            x=c_source,
            y=c_target,
            cost_fn=cost_fn,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon,
            scale_cost=scale_cost,
        )
        gt_displacement_cost = jnp.mean(jax.vmap(cost_fn)(source, target))
        out = linear.solve(geom=geom, **kwargs)
        loss = gt_displacement_cost - out.ent_reg_cost
        all_losses.append(loss)
        if return_output:
            all_outs.append(out)
    loss = sum(all_losses) / len(all_losses)  # average

    return (loss, out, all_losses) if return_output else loss


class ConditionalMongeGapEstimator:
    r"""Monge Gap Estimator which optimizes over multiple conditions

    .. math::
      \text{min}_{\theta}\; \sum_{i=1}^{K} \Delta(T_\theta \sharp \mu, \theta)
      + \lambda R(T_\theta \sharp \rho, \rho)

    Args:
      dim_data: input dimensionality of data required for network init.
      dim_cond: input dimensionality of condition embedding,
        required for network init
      model: network architecture for map :math:`T`,
        should be a `ConditionalPerturbationNetwork`.
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
        dim_cond: int,
        model: ConditionalPerturbationNetwork,
        optimizer: Optional[optax.OptState] = None,
        fitting_loss: Optional[
            Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, Optional[Any]]]
        ] = None,
        regularizer: Optional[
            Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, Optional[Any]]]
        ] = None,
        regularizer_strength: Union[float, Sequence[float]] = 1.0,
        num_train_iters: int = 10_000,
        logging: bool = False,
        valid_freq: int = 500,
        grad_acc_steps: int = 1,
        rng: Optional[jax.Array] = None,
    ) -> None:
        self._fitting_loss = fitting_loss
        self._regularizer = regularizer
        self.num_train_iters = self.num_train_iters
        self.grad_acc_steps = grad_acc_steps
        # Can use either a fixed strength, or generalize to a schedule.
        self.regularizer_strength = jnp.repeat(
            jnp.atleast_2d(regularizer_strength),
            num_train_iters,
            total_repeat_length=num_train_iters,
            axis=0,
        ).ravel()
        self.logging = logging
        self.valid_freq = valid_freq
        self.rng = jax.random.PRNGKey(rng)

        # set default optimizer
        if optimizer is None:
            optimizer = optax.adam(learning_rate=0.001)

        # setup training
        self.setup(dim_data, dim_cond, model, optimizer)

    def setup(
        self,
        dim_data: int,
        dim_cond: int,
        neural_net: ConditionalPerturbationNetwork,
        optimizer: optax.OptState,
    ):
        """Setup all components required to train the network"""

        # neural network
        self.rng, rng = jax.random.split(self.key, 2)
        self.state_neural_net = neural_net.create_train_state(
            self.rng, optimizer, dim_data, dim_cond
        )

        # step function
        self.step_fn = self._get_step_fn()

        @property
        def regularizer(self) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
            """Regularizer added to the fitting loss.

            Can be, e.g. the
            :func:`~ott.neural.methods.monge_gap.monge_gap_from_samples`.
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

            Can be, e.g. :func:`~ott.tools.sinkhorn_divergence.sinkdiv`.
            If no fitting_loss is passed for solver instantiation, return 0 by default,
            and no log values.
            """
            if self._fitting_loss is not None:
                return self._fitting_loss
            return lambda *_, **__: (0.0, None)

    def _generate_batch(
        self, loader_source, loader_target
    ) -> Dict[str, jnp.ndarray]:
        """Generate a batch of condition and samples."""
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
    ):
        """The dataloaders should return a dict with key `X`.
        The target dataloaders should additionally include a key
        `c_embed`, which has the embedded condition in `dim_cond`."""
        # define logs
        logs = collections.defaultdict(lambda: collections.defaultdict(list))

        # try to display training progress with tqdm
        try:
            from tqdm import trange

            tbar = trange(self.num_train_iters, leave=True)
        except ImportError:
            tbar = range(self.num_train_iters)

        grads = tree_map(jnp.zeros_like, self.state_neural_net.params)
        for step in tbar:
            #  update step
            is_logging_step = self.logging and (
                (step % self.valid_freq == 0)
                or (step == self.num_train_iters - 1)
            )
            is_gradient_acc_step = (step + 1) % self.grad_acc_steps == 0
            train_batch, condition = self._generate_batch(
                trainloader_source, trainloader_target
            )
            valid_batch, _ = (
                None
                if not is_logging_step
                else self._generate_batch(
                    validloader_source, validloader_target
                )
            )

            self.state_neural_net, grads, current_logs = self.step_fn(
                self.state_neural_net,
                grads=grads,
                train_batch=train_batch,
                valid_batch=valid_batch,
                is_logging_step=is_logging_step,
                is_gradient_acc_step=is_gradient_acc_step,
            )

            # store and print metrics if logging step
            if is_logging_step:
                for log_key in current_logs:
                    for metric_key in current_logs[log_key]:
                        logs[log_key][metric_key].append(
                            current_logs[log_key][metric_key]
                        )

            # update the tqdm bar if tqdm is available
            if not isinstance(tbar, range):
                reg_msg = (
                    "NA"
                    if current_logs["eval"]["regularizer"] == 0.0
                    else f"{current_logs['eval']['regularizer']:.4f}"
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
            params: frozen_dict.FrozenDict,
            apply_fn: Callable,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[float, Dict[str, float]]:
            """Loss function."""
            # map samples with the fitted map
            mapped_samples = apply_fn(
                {"params": params},
                batch["source"]["X"],
                batch["target"]["c_embed"],
            )

            # compute the loss
            val_fitting_loss, log_fitting_loss = self.fitting_loss(
                batch["target"]["X"], mapped_samples
            )
            val_regularizer, log_regularizer = self.regularizer(
                batch["source"]["X"], mapped_samples
            )
            val_tot_loss, log_regularizer = val_fitting_loss + val_regularizer

            # store training logs
            loss_logs = {
                "total_loss": val_tot_loss,
                "fitting_loss": val_fitting_loss,
                "regularizer": val_regularizer,
                "log_regularizer": log_regularizer,
                "log_fitting": log_fitting_loss,
            }

            return val_tot_loss, loss_logs

        @functools.partial(jax.jit, static_argnums=[4, 5])
        def step_fn(
            state_neural_net: train_state.TrainState,
            grads: frozen_dict.FrozenDict,
            train_batch: Dict[str, jnp.ndarray],
            valid_batch: Optional[Dict[str, jnp.ndarray]] = None,
            is_logging_step: bool = False,
            is_gradient_acc_step: bool = False,
        ) -> Tuple[
            train_state.TrainState, frozen_dict.FrozenDict, Dict[str, float]
        ]:
            """Step function."""
            # compute loss and gradients
            grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
            (_, current_train_logs), step_grads = grad_fn(
                state_neural_net.params,
                state_neural_net.apply_fn,
                train_batch,
            )
            # Accumulate gradients
            grads = tree_map(lambda g, step_g: g + step_g, grads, step_grads)

            # logging step
            current_logs = {"train": current_train_logs, "eval": {}}
            if is_logging_step:
                _, current_eval_logs = loss_fn(
                    params=state_neural_net.params,
                    apply_fn=state_neural_net.apply_fn,
                    batch=valid_batch,
                )
                current_logs["eval"] = current_eval_logs

            # update state
            if is_gradient_acc_step:
                state_neural_net = state_neural_net.apply_gradients(
                    grads=tree_map(lambda g: g / self.grad_acc_steps, grads)
                )
                # Reset gradients
                grads = tree_map(jnp.zeros_like, grads)

            return state_neural_net, grads, current_logs

        return step_fn

    def transport(self, x, c):
        return self.state_neural_net.apply_fn(
            {"params": self.state_neural_net.params}, x, c
        )

    @property
    def model(self) -> nn.Module:
        return self.state_neural_net

    @model.setter
    def model(self, value: nn.Module):
        """Setter for the model to be checkpointed."""
        self.state_neural_net = value

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """Abstract method for saving model parameters to a pickle file.

        Args:
            path: Path where the checkpoint should be saved. Defaults to None in which case
                it is retrieved from config.
            config: The model training configuration with a `checkpointing_path` field.
                Defaults to None.
                NOTE: If `config` and `path` are both not None, `path` takes preference.
        """
        if path is None:
            raise ValueError(
                "Checkpoint cannot be saved. Provide a checkpoint save path"
            )
        try:
            checkpointer = PyTreeCheckpointer()
            save_args = save_args_from_target(self.model)
            checkpointer.save(path, self.model, save_args=save_args, force=True)
        except Exception as e:
            raise Exception(f"Error in saving checkpoint to {path}: {e}")

    @classmethod
    def load_checkpoint(
        cls: Type[T],
        ckpt_path: Path = None,
        *args,
        **kwargs,
    ) -> T:
        """
        Loading a model from a checkpoint

        Args:
            cls: Class object to be created.
            ckpt_path: Optional path from where checkpoint is restored.
                Defaults to None, in that case inferred from config.
            *args: args normally given to `ConditionalMongeGapEstimator`

        Returns:
            Class object with restored weights.
        """
        try:
            out_class = cls(
                *args,
                **kwargs,
            )
            checkpointer = PyTreeCheckpointer()
            out_class.model = checkpointer.restore(
                ckpt_path, item=out_class.model
            )
            return out_class
        except Exception as e:
            raise Exception(
                f"Failed to load checkpoin from {ckpt_path}: {e}\nAre you sure"
                "checkpoint was saved and correct path is provided?"
            )


# Add condition embedding to dataloader --> Users
# How to sample condition...?
# Optim/embedding/regularization factory?
