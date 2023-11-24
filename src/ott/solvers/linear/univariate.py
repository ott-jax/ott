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

from typing import NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem

__all__ = ["UnivariateOutput", "UnivariateSolver"]


class UnivariateOutput(NamedTuple):  # noqa: D101
  prob: linear_problem.LinearProblem
  ot_costs: float
  paired_indices: jax.Array
  mass_paired_indices: jax.Array

  @property
  def transport_matrices(self) -> jax.Array:
    """Output a ``[d,n,m]`` tensor of all ``[n,m]`` transport matrices."""
    assert self.paired_indices is not None, "[d,n,m] Transports *not* computed"

    n, m = self.prob.geom.shape
    if self.prob.is_equal_size and self.prob.is_uniform:
      transport_matrices_from_indices = jax.vmap(
          lambda idx, idy: jnp.eye(n)[idx, :][:, idy].T, in_axes=[0, 0]
      )
      return transport_matrices_from_indices(
          self.paired_indices[:, 0, :], self.paired_indices[:, 1, :]
      )

    # raveled indexing of entries.
    indices = self.paired_indices[:, 0] * m + self.paired_indices[:, 1]
    # segment sum is needed to collect several contributions
    return jax.vmap(
        lambda idx, mass: jax.ops.segment_sum(
            mass, idx, indices_are_sorted=True, num_segments=n * m
        ).reshape(n, m),
        in_axes=[0, 0]
    )(indices, self.mass_paired_indices)

  @property
  def mean_transport_matrix(self) -> jax.Array:
    """Return the mean tranport matrix, averaged over slices."""
    return jnp.mean(self.transport_matrices, axis=0)


@jax.tree_util.register_pytree_node_class
class UnivariateSolver:
  r"""Univariate solver to compute 1D OT distance over slices of data.

  Computes 1-Dimensional optimal transport distance between two $d$-dimensional
  point clouds. The total distance is the sum of univariate Wasserstein
  distances on the $d$ slices of data: given two weighted point-clouds, stored
  as ``[n,d]`` and ``[m,d]`` in a
  :class:`~ott.problems.linear.linear_problem.LinearProblem` object, with
  respective weights ``a`` and ``b``, the solver
  computes ``d`` OT distances between each of these ``[n,1]`` and ``[m,1]``
  slices. The distance is computed using the analytical formula by default,
  which involves sorting each of the slices independently. The optimal transport
  matrices are also outputted when possible (described in sparse form, i.e.
  pairs of indices and mass transferred between those indices).

  When weights ``a`` and ``b`` are uniform, and ``n=m``, the computation only
  involves comparing sorted entries per slice, and ``d`` assignments are given.

  The user may also supply a ``num_subsamples`` parameter to extract as many
  points from the original point cloud, sampled with probability masses ``a``
  and ``b``. This then simply applied the method above to the subsamples, to
  output ``d`` costs, but assignments are not provided.

  When the problem is not uniform or not of equal size, the method defaults to
  an inversion of the CDF, and outputs both costs and transport matrix in sparse
  form.

  When a ``quantiles`` argument is passed, either specifying explicit quantiles
  or a grid of quantiles, the distance is evaluated by comparing the quantiles
  of the two point clouds on each slice. The OT costs are returned but
  assignments are not provided.

  Args:
    num_subsamples: option to reduce the size of inputs by doing random
      subsampling, taken into account marginal probabilities.
    quantiles: When a vector or a number of quantiles is passed, the distance
      is computed by evaluating the cost function on the sectional (one for each
      dimension) quantiles of the two point cloud distributions described in the
      problem.
  """

  def __init__(
      self,
      num_subsamples: Optional[int] = None,
      quantiles: Optional[Union[int, jnp.ndarray]] = None,
  ):
    self._quantiles = quantiles
    self.num_subsamples = num_subsamples

  @property
  def quantiles(self):
    """Quantiles' values used to evaluate OT cost."""
    if self._quantiles is None:
      return None
    if isinstance(self._quantiles, int):
      return jnp.linspace(0.0, 1.0, self._quantiles)
    return self._quantiles

  @property
  def num_quantiles(self):
    """Number of quantiles used to evaluate OT cost."""
    return 0 if self._quantiles is None else self.quantiles.shape[0]

  def __call__(
      self,
      prob: linear_problem.LinearProblem,
      rng: Optional[jax.Array] = None,
  ) -> float:
    """Computes Univariate Distance between the `d` dimensional slices.

    Args:
      prob: describing, in its geometry attribute, the two point clouds
        ``x`` and ``y`` (of respective sizes ``[n,d]`` and ``[m,d]``) and
        a ground ``cost_fn`` for between two scalars. The ``[n,]`` and ``[m,]``
        size probability weights vectors are stored in attributes ``a`` and
        ``b``.
      rng: used for random downsampling, if used.
      return_transport: whether to return an average transport matrix (across
        slices). Not available when approximating the distance computation
        using subsamples, or quantiles.

    Returns:
      The OT distance, and possibly the transport matrix averaged by
        considering all matrices arising from 1D transport on each of the ``d``
        dimensional slices of the input.
    """
    geom = prob.geom
    n, m = geom.shape
    rng = jax.random.PRNGKey(0) if rng is None else rng
    geom_is_pc = isinstance(geom, pointcloud.PointCloud)
    assert geom_is_pc, "Geometry object in problem must be a PointCloud."
    cost_is_TI = isinstance(geom.cost_fn, costs.TICost)
    assert cost_is_TI, "Geometry's cost must be translation invariant."
    x, y = geom.x, geom.y

    # check if problem has the property uniform / same number of points
    is_uniform_same_size = prob.is_uniform and prob.is_equal_size
    if self.num_subsamples:
      rng1, rng2 = jax.random.split(rng, 2)
      if prob.is_uniform:
        x = x[jnp.linspace(0, n, num=self.num_subsamples).astype(int), :]
        y = y[jnp.linspace(0, m, num=self.num_subsamples).astype(int), :]
      else:
        x = jax.random.choice(rng1, x, (self.num_subsamples,), p=prob.a, axis=0)
        y = jax.random.choice(rng2, y, (self.num_subsamples,), p=prob.b, axis=0)
        n = m = self.num_subsamples
      # now that both are subsampled, consider them as uniform/same size.
      is_uniform_same_size = True

    if self.quantiles is None:
      if is_uniform_same_size:
        i_x, i_y = jnp.argsort(x, axis=0), jnp.argsort(y, axis=0)
        x = jnp.take_along_axis(x, i_x, axis=0)
        y = jnp.take_along_axis(y, i_y, axis=0)
        ot_costs = jax.vmap(geom.cost_fn.h, in_axes=[0])(x.T - y.T) / n

        if self.num_subsamples:
          # When subsampling, the pairing computed have no meaning w.r.t.
          # original data.
          paired_indices, mass_paired_indices = None, None
        else:
          paired_indices = jnp.stack([i_x, i_y]).transpose([2, 0, 1])
          mass_paired_indices = jnp.ones((n,)) / n

      else:
        ot_costs, paired_indices, mass_paired_indices = jax.vmap(
            self._quantile_distance_and_transport,
            in_axes=[1, 1, None, None, None]
        )(x, y, prob.a, prob.b, geom.cost_fn)

    else:
      assert prob.is_uniform, "Quantile method only valid for uniform marginals"
      x_q = jnp.quantile(x, self.quantiles, axis=0)
      y_q = jnp.quantile(y, self.quantiles, axis=0)
      ot_costs = jax.vmap(geom.cost_fn.pairwise, in_axes=[1, 1])(x_q, y_q)
      ot_costs /= self.num_quantiles
      paired_indices = None
      mass_paired_indices = None

    return UnivariateOutput(
        prob=prob,
        ot_costs=ot_costs,
        paired_indices=paired_indices,
        mass_paired_indices=mass_paired_indices
    )

  def _quantile_distance_and_transport(
      self, x: jnp.ndarray, y: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray,
      cost_fn: costs.TICost
  ):
    # Implementation inspired by `scipy` implementation for
    # :func:<scipy.stats.wasserstein_distance>
    def sort_and_argsort(x: jnp.array, return_argsort: bool, **kwargs):
      if return_argsort:
        i_x = jnp.argsort(x, **kwargs)
        return x[i_x], i_x
      return jnp.sort(x), None

    x, i_x = sort_and_argsort(x, True)
    y, i_y = sort_and_argsort(y, True)

    all_values = jnp.concatenate([x, y])
    all_values_sorted, all_values_sorter = sort_and_argsort(all_values, True)

    x_pdf = jnp.concatenate([a[i_x], jnp.zeros_like(b)])[all_values_sorter]
    y_pdf = jnp.concatenate([jnp.zeros_like(a), b[i_y]])[all_values_sorter]

    x_cdf = jnp.cumsum(x_pdf)
    y_cdf = jnp.cumsum(y_pdf)

    x_y_cdfs = jnp.concatenate([x_cdf, y_cdf])
    quantile_levels, _ = sort_and_argsort(x_y_cdfs, False)

    i_x_cdf_inv = jnp.searchsorted(x_cdf, quantile_levels)
    x_cdf_inv = all_values_sorted[i_x_cdf_inv]
    i_y_cdf_inv = jnp.searchsorted(y_cdf, quantile_levels)
    y_cdf_inv = all_values_sorted[i_y_cdf_inv]

    diff_q = jnp.diff(quantile_levels)
    cost = jnp.sum(
        jax.vmap(cost_fn.h)(y_cdf_inv[1:, None] - x_cdf_inv[1:, None]) * diff_q
    )

    n = x.shape[0]

    i_in_sorted_x_of_quantile = all_values_sorter[i_x_cdf_inv] % n
    i_in_sorted_y_of_quantile = all_values_sorter[i_y_cdf_inv] - n

    orig_i = (i_x[i_in_sorted_x_of_quantile])[1:]
    orig_j = (i_y[i_in_sorted_y_of_quantile])[1:]

    return cost, jnp.stack([orig_i, orig_j]), diff_q

  def tree_flatten(self):  # noqa: D102
    aux_data = vars(self).copy()
    return [], aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):  # noqa: D102
    return cls(*children, **aux_data)
