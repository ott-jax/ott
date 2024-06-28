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
from typing import NamedTuple, Optional, Tuple

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp

from ott.math import utils as mu
from ott.problems.linear import linear_problem

__all__ = [
    "UnivariateOutput",
    "uniform_solver",
    "quantile_solver",
    "north_west_solver",
]


class UnivariateOutput(NamedTuple):
  r"""Output of the univariate solver.

  Args:
    prob: Linear problem between two weighted ``[n, d]`` and ``[m, d]``
      point clouds.
    ot_costs: Array of shape ``[d,]`` of OT costs, computed independently along
      each of the :math:`d` slices.
    paired_indices: Array of shape ``[d, 2, n + m]``, of :math:`n + m` pairs
      of indices, for which the optimal transport assigns mass, on each slice
      of the :math:`d` slices described in the dataset. Namely, for each index
      :math:`0 <= k < n + m`, :math:`0 <= s < d`, if one has
      :math:`i := \text{paired_indices}[s, 0, k]` and
      :math:`j := \text{paired_indices}[s, 1, k]`, then point :math:`i` in
      the first point cloud sends mass to point :math:`j` in the second,
      in slice :math:`s`.
    mass_paired_indices: ``[d, n + m]`` array of weights. Using the notation
      above, if :math:`0 <= k < n + m`, and :math:`0 <= s < d`  then writing
      :math:`i := \text{paired_indices}[s, 0, k]` and
      :math:`j := \text{paired_indices}[s, 1, k]`, point :math:`i` sends
      :math:`\text{mass_paired_indices}[s, k]` to point :math:`j`.
    dual_a: Array of shape ``[n,]`` containing the first dual variable.
    dual_b: Array of shape ``[m,]`` containing the second dual variable.
  """
  prob: linear_problem.LinearProblem
  ot_costs: jnp.ndarray
  paired_indices: Optional[jnp.ndarray] = None
  mass_paired_indices: Optional[jnp.ndarray] = None
  dual_a: Optional[jnp.ndarray] = None
  dual_b: Optional[jnp.ndarray] = None

  @property
  def transport_matrices(self) -> jesp.BCOO:
    """Array of shape ``[d, n, m]`` containing all transport matrices.

    The matrices will be sparse, with at most :math:`n + m` entries
    for each of the :math:`d` slices.
    """
    b = len(self.ot_costs)
    n, m = self.prob.geom.shape
    data = self.mass_paired_indices
    indices = self.paired_indices.swapaxes(1, 2)
    return jesp.BCOO((data, indices), shape=(b, n, m))

  @property
  def mean_transport_matrix(self) -> jesp.BCOO:
    """Mean transport matrix, averaged over :math:`d` slices."""
    sparse_mean = jesp.sparsify(jnp.mean)
    return sparse_mean(self.transport_matrices, axis=0)

  @property
  def dual_costs(self) -> jnp.ndarray:
    """Array of shape ``[d,]`` containing the dual costs."""
    assert self.dual_a is not None, "Dual variables have not been computed."
    dual_obj = jnp.sum(self.dual_a * self.prob.a[None, :], axis=1)
    dual_obj += jnp.sum(self.dual_b * self.prob.b[None, :], axis=1)
    return dual_obj


def uniform_solver(
    prob: linear_problem.LinearProblem,
    return_transport: bool = False,
) -> UnivariateOutput:
  """Univariate solver between two equally sized and uniformly weighted distributions.

  Args:
    prob: Problem with two :class:`point clouds <ott.geometry.pointcloud.PointCloud>`
      of shapes ``[n, d]`` and ``[n, d]`` and a ground
      :class:`translation-invariant cost <ott.geometry.costs.TICost>`.
      The ``[n,]`` sized probability weights are stored
      in attributes :attr:`~ott.problems.linear.linear_problem.LinearProblem.a`
      and :attr:`~ott.problems.linear.linear_problem.LinearProblem.b`.
    return_transport: Whether to also return the mapped pairs used to compute
      the :attr:`~ott.solvers.linear.univariate.UnivariateOutput.transport_matrices`.

  Returns:
    The univariate output. Note that the
    :attr:`~ott.solvers.linear.univariate.UnivariateOutput.mass_paired_indices`
    can be :math:`0` in some entries, but always sums to :math:`1`
    for each of the :math:`d` slices.
  """  # noqa: E501
  assert prob.is_equal_size, "Source and target have different sizes."
  assert prob.is_uniform, "Source or target marginals are not uniform."

  geom = prob.geom
  n, _ = geom.shape
  x, y, cost_fn = geom.x, geom.y, geom.cost_fn

  i_x, i_y = jnp.argsort(x, axis=0), jnp.argsort(y, axis=0)
  x = jnp.take_along_axis(x, i_x, axis=0)
  y = jnp.take_along_axis(y, i_y, axis=0)
  ot_costs = ((1.0 / n) * jax.vmap(cost_fn.h, in_axes=1)(x - y)).T

  if return_transport:
    paired_indices = jnp.stack([i_x, i_y]).transpose([2, 0, 1])
    mass_paired_indices = jnp.ones((len(ot_costs), n)) / n
  else:
    paired_indices = mass_paired_indices = None

  return UnivariateOutput(
      prob,
      ot_costs,
      paired_indices=paired_indices,
      mass_paired_indices=mass_paired_indices,
  )


def quantile_solver(
    prob: linear_problem.LinearProblem,
    return_transport: bool = False,
) -> UnivariateOutput:
  """Univariate solver between quantile functions of distributions.

  Args:
    prob: Problem with two :class:`point clouds <ott.geometry.pointcloud.PointCloud>`
      of shapes ``[n, d]`` and ``[m, d]`` and a ground
      :class:`translation-invariant cost <ott.geometry.costs.TICost>`.
      The ``[n,]`` and ``[m,]`` sized probability weights vectors are stored
      in attributes :attr:`~ott.problems.linear.linear_problem.LinearProblem.a`
      and :attr:`~ott.problems.linear.linear_problem.LinearProblem.b`.
    return_transport: Whether to also return the mapped pairs used to compute
      the :attr:`~ott.solvers.linear.univariate.UnivariateOutput.transport_matrices`.

  Returns:
    The univariate output. Note that the
    :attr:`~ott.solvers.linear.univariate.UnivariateOutput.mass_paired_indices`
    can be :math:`0` in some entries, but always sums to :math:`1`
    for each of the :math:`d` slices.

  Notes:
    This function was inspired by :func:`~scipy.stats.wasserstein_distance`,
    but can be used with other costs, not just :math:`c(x, y) = |x - y|`.
  """  # noqa: E501

  @functools.partial(jax.vmap, in_axes=[1, 1])
  def dist(x: jnp.ndarray, y: jnp.ndarray):
    x, i_x = mu.sort_and_argsort(x, argsort=True)
    y, i_y = mu.sort_and_argsort(y, argsort=True)

    all_values = jnp.concatenate([x, y])
    all_values_sorted, all_values_sorter = mu.sort_and_argsort(
        all_values, argsort=True
    )

    x_pdf = jnp.concatenate([prob.a[i_x], jnp.zeros_like(prob.b)])
    x_pdf = x_pdf[all_values_sorter]
    y_pdf = jnp.concatenate([jnp.zeros_like(prob.a), prob.b[i_y]])
    y_pdf = y_pdf[all_values_sorter]

    x_cdf = jnp.cumsum(x_pdf)
    y_cdf = jnp.cumsum(y_pdf)

    x_y_cdfs = jnp.concatenate([x_cdf, y_cdf])
    quantile_levels, _ = mu.sort_and_argsort(x_y_cdfs, argsort=False)

    i_x_cdf_inv = jnp.searchsorted(x_cdf, quantile_levels)
    x_cdf_inv = all_values_sorted[i_x_cdf_inv]
    i_y_cdf_inv = jnp.searchsorted(y_cdf, quantile_levels)
    y_cdf_inv = all_values_sorted[i_y_cdf_inv]

    diff_q = jnp.diff(quantile_levels)
    successive_costs = jax.vmap(prob.geom.cost_fn.h)(
        x_cdf_inv[1:, None] - y_cdf_inv[1:, None]
    )
    cost = jnp.sum(successive_costs * diff_q)

    if not return_transport:
      return cost, None, None

    n = x.shape[0]
    i_in_sorted_x_of_quantile = all_values_sorter[i_x_cdf_inv] % n
    i_in_sorted_y_of_quantile = all_values_sorter[i_y_cdf_inv] - n
    orig_i = i_x[i_in_sorted_x_of_quantile][1:]
    orig_j = i_y[i_in_sorted_y_of_quantile][1:]
    paired_indices, mass_paired_indices = jnp.stack([orig_i, orig_j]), diff_q

    return cost, paired_indices, mass_paired_indices

  ot_costs, pi, mpi = dist(prob.geom.x, prob.geom.y)
  return UnivariateOutput(
      prob,
      ot_costs,
      paired_indices=pi,
      mass_paired_indices=mpi,
  )


def north_west_solver(prob: linear_problem.LinearProblem) -> UnivariateOutput:
  """Univariate solver that implements the north-west corner rule.

  This rule is described in :cite:`peyre:19`, sec. 3.4.2 and the dual variables
  are stored as described in :cite:`sejourne:22`, alg. 3.

  Args:
    prob: Problem with two :class:`point clouds <ott.geometry.pointcloud.PointCloud>`
      of shapes ``[n, d]`` and ``[m, d]`` and a ground
      :class:`translation-invariant cost <ott.geometry.costs.TICost>`.
      The ``[n,]`` and ``[m,]`` sized probability weights are stored
      in attributes :attr:`~ott.problems.linear.linear_problem.LinearProblem.a`
      and :attr:`~ott.problems.linear.linear_problem.LinearProblem.b`.

  Returns:
    The univariate output. Note that the
    :attr:`~ott.solvers.linear.univariate.UnivariateOutput.mass_paired_indices`
    can be :math:`0` in some entries, but always sums to :math:`1`
    for each of the :math:`d` slices.
  """  # noqa: E501

  class State(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    paired_indices: jnp.ndarray
    mass_paired_indices: jnp.ndarray
    dual_a: jnp.ndarray
    dual_b: jnp.ndarray

  def dual_a_update(state: State, i: int,
                    j: int) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    next_ixs = jnp.array([i + 1, j])
    val = cost_fn.h(state.x[i + 1, None] - state.y[j, None]) - state.dual_b[j]
    da = state.dual_a.at[i + 1].set(val)
    return state._replace(dual_a=da), state.a[i], next_ixs

  def dual_b_update(state: State, i: int,
                    j: int) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    next_ixs = jnp.array([i, j + 1])
    val = cost_fn.h(state.x[i, None] - state.y[j + 1, None]) - state.dual_a[i]
    db = state.dual_b.at[j + 1].set(val)
    return state._replace(dual_b=db), state.b[j], next_ixs

  def body_fun(ix: int, state: State) -> State:
    i, j = state.paired_indices[0, ix], state.paired_indices[1, ix]
    state, min_ab, next_ixs = jax.lax.cond(
        state.a[i] < state.b[j], dual_a_update, dual_b_update, state, i, j
    )

    pi = state.paired_indices.at[:, ix + 1].set(next_ixs)
    mpi = state.mass_paired_indices.at[ix].set(min_ab)
    a_ = state.a.at[i].set(state.a[i] - min_ab)
    b_ = state.b.at[j].set(state.b[j] - min_ab)

    return state._replace(
        paired_indices=pi, mass_paired_indices=mpi, a=a_, b=b_
    )

  @functools.partial(jax.vmap, in_axes=[1, 1])
  def dist(x: jnp.ndarray, y: jnp.ndarray):
    x, i_x = mu.sort_and_argsort(x, argsort=True)
    y, i_y = mu.sort_and_argsort(y, argsort=True)
    sorted_a, sorted_b = a[i_x], b[i_y]

    paired_indices = jnp.zeros((2, q), dtype=int)
    mass_paired_indices = jnp.zeros(q)

    state = State(
        x,
        y,
        a=sorted_a,
        b=sorted_b,
        paired_indices=paired_indices,
        mass_paired_indices=mass_paired_indices,
        dual_a=jnp.zeros(n),
        dual_b=jnp.zeros(m).at[0].set(cost_fn.h(x[0, None] - y[0, None])),
    )
    state = jax.lax.fori_loop(0, q - 1, body_fun, state)

    ot_cost = jnp.sum(state.dual_a * sorted_a
                     ) + jnp.sum(state.dual_b * sorted_b)

    p_final = jnp.maximum(state.a[-1], state.b[-1])
    mass_paired_indices = state.mass_paired_indices.at[-1].set(p_final)

    return (
        ot_cost,
        state.paired_indices,
        mass_paired_indices,
        # restore the original order
        state.dual_a[jnp.argsort(i_x)],
        state.dual_b[jnp.argsort(i_y)],
    )

  a, b = prob.a, prob.b
  n, m = prob.geom.shape
  q = m + n - 1
  cost_fn = prob.geom.cost_fn

  ot_costs, pi, mpi, dual_a, dual_b = dist(prob.geom.x, prob.geom.y)
  return UnivariateOutput(
      prob,
      ot_costs,
      paired_indices=pi,
      mass_paired_indices=mpi,
      dual_a=dual_a,
      dual_b=dual_b,
  )
