import collections
import functools

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
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
from ott.neural.methods.monge_gap import monge_gap_from_samples
from ott.solvers.linear import sinkhorn
from ott.neural.networks.conditional_perturbation_network import (
    ConditionalPerturbationNetwork,
)

T = TypeVar("T", bound="ConditionalMongeGapEstimator")


def cmonge_gap_from_samples(
    source: jnp.ndarray,
    target: jnp.ndarray,
    condition: jnp.ndarray,
    return_output: bool = False,
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
    condition: array indicating condition for each source-target sample
        `integer array of shape ``[n]``.
    return_output: boolean to also return the
        :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`.
    kwargs: holds the kwargs to the function
        :function:`~ott.neural.methods.monge_gap.monge_gap_from_samples`

    Returns:
    The average Monge gap value over all conditions and optionally the
    list of Monge gap per condition and :class:`~ott.solvers.linear.sinkhorn.SinkhornOutput`
    """
    all_gaps = []
    all_outs = []
    for c in jnp.unique(condition):
        c_target = target[condition == c]
        c_source = source[condition == c]

        if return_output:
            monge_gap, out = monge_gap_from_samples(
                target=c_target, source=c_source, return_output=True, **kwargs
            )
            all_outs.append(out)
        else:
            monge_gap = monge_gap_from_samples(
                target=c_target, source=c_source, return_output=False, **kwargs
            )
        all_gaps.append(monge_gap)

    condition_monge_gap = sum(all_gaps) / len(all_gaps)  # average

    return (
        (condition_monge_gap, all_outs, all_gaps)
        if return_output
        else condition_monge_gap
    )
