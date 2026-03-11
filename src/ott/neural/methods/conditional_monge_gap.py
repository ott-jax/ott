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
from typing import (
    Any,
    Literal,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud, segment
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

__all__ = ["cmonge_gap_from_samples"]


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
