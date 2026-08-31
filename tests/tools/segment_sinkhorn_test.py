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
import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import segment_sinkhorn
from ott.tools.gaussian_mixture import gaussian_mixture
from tests import _utils

#: Padding size used when segmenting the measures below.
MAX_MEASURE_SIZE = 20


@pytest.fixture(scope="module")
def clouds(rng: jax.Array) -> _utils.PointClouds:
  """Point clouds drawn exactly as this module always has."""
  n, m, dim = 13, 17, 4
  rng_xy, rng_a, rng_b = jr.split(rng, 3)
  rng_x, rng_y, _, _ = jr.split(rng_xy, 4)
  return _utils.PointClouds(
      x=jr.uniform(rng_x, (n, dim)),
      y=jr.uniform(rng_y, (m, dim)),
      a=_utils.random_probs(rng_a, n, offset=0.0),
      b=_utils.random_probs(rng_b, m, offset=0.0),
  )


class TestSegmentSinkhorn:

  @pytest.mark.parametrize("shuffle", [False, True])
  def test_segment_sinkhorn_result(
      self, clouds: _utils.PointClouds, rng: jax.Array, shuffle: bool
  ):
    # Test that segmented sinkhorn gives the same results as run separately:
    rngs = jr.split(rng, 2)
    x, y = clouds.x, clouds.y
    geom_kwargs = {"epsilon": 0.014}
    sinkhorn_kwargs = {"threshold": 2e-3}

    geom = pointcloud.PointCloud(x, y, **geom_kwargs)
    prob = linear_problem.LinearProblem(geom, a=clouds.a, b=clouds.b)
    solver = sinkhorn.Sinkhorn(**sinkhorn_kwargs)
    true_cost = solver(prob).reg_ot_cost

    if shuffle:
      # Now, shuffle the order of both arrays, but
      # still maintain the segment assignments:
      idx_x = jr.permutation(
          rngs[0], jnp.arange(x.shape[0] * 2), independent=True
      )
      idx_y = jr.permutation(
          rngs[1], jnp.arange(y.shape[0] * 2), independent=True
      )
    else:
      idx_x = jnp.arange(x.shape[0] * 2)
      idx_y = jnp.arange(y.shape[0] * 2)

    # Duplicate arrays:
    x_copied = jnp.concatenate((x, x))[idx_x]
    a_copied = jnp.concatenate((clouds.a, clouds.a))[idx_x]
    segment_ids_x = jnp.arange(2).repeat(x.shape[0])[idx_x]

    y_copied = jnp.concatenate((y, y))[idx_y]
    b_copied = jnp.concatenate((clouds.b, clouds.b))[idx_y]
    segment_ids_y = jnp.arange(2).repeat(y.shape[0])[idx_y]

    seg_cost = segment_sinkhorn.segment_sinkhorn(
        x_copied,
        y_copied,
        num_segments=2,
        max_measure_size=MAX_MEASURE_SIZE,
        segment_ids_x=segment_ids_x,
        segment_ids_y=segment_ids_y,
        indices_are_sorted=False,
        weights_x=a_copied,
        weights_y=b_copied,
        sinkhorn_kwargs=sinkhorn_kwargs,
        **geom_kwargs
    )

    np.testing.assert_allclose(
        true_cost.repeat(2), seg_cost, rtol=1e-4, atol=1e-4
    )

  def test_segment_sinkhorn_different_segment_sizes(self):
    # Test other array sizes
    x1 = jnp.arange(10)[:, None].repeat(2, axis=1) - 0.1
    y1 = jnp.arange(11)[:, None].repeat(2, axis=1) + 0.1

    # Should have larger divergence since further apart:
    x2 = jnp.arange(12)[:, None].repeat(2, axis=1) - 0.1
    y2 = 2 * jnp.arange(13)[:, None].repeat(2, axis=1) + 0.1

    sink = jax.jit(
        segment_sinkhorn.segment_sinkhorn,
        static_argnames=["num_segments", "max_measure_size"],
    )
    seg_cost = sink(
        jnp.concatenate((x1, x2)),
        jnp.concatenate((y1, y2)),
        num_segments=2,
        max_measure_size=14,
        num_per_segment_x=(10, 12),
        num_per_segment_y=(11, 13),
        epsilon=0.01
    )

    assert seg_cost.shape[0] == 2
    assert seg_cost[1] > seg_cost[0]

    true_cost = []
    solver = jax.jit(sinkhorn.Sinkhorn())
    for x, y in zip((x1, x2), (y1, y2)):
      geom = pointcloud.PointCloud(x, y, epsilon=1e-2)
      prob = linear_problem.LinearProblem(geom)
      true_cost.append(solver(prob).reg_ot_cost)

    np.testing.assert_allclose(seg_cost, true_cost, atol=1e-4, rtol=1e-4)

  def test_sinkhorn_divergence_segment_custom_padding(self, rng):
    rngs = jr.split(rng, 4)
    dim = 3
    b_cost = costs.Bures(dim)

    num_per_segment_x = (5, 2)
    num_per_segment_y = (3, 5)
    ns = num_per_segment_x + num_per_segment_y

    means_and_covs_to_x = jax.vmap(
        costs.mean_and_cov_to_x, in_axes=[0, 0, None]
    )

    def g(rng, n):
      out = gaussian_mixture.GaussianMixture.from_random(
          rng, n_components=n, n_dimensions=dim
      )
      return means_and_covs_to_x(out.loc, out.covariance, dim)

    x1, x2, y1, y2 = (g(rngs[i], ns[i]) for i in range(4))

    true_reg_ot_cost = []
    solver = jax.jit(sinkhorn.Sinkhorn())
    for x, y in zip((x1, x2), (y1, y2)):
      geom = pointcloud.PointCloud(x, y, cost_fn=b_cost, epsilon=1e-1)
      prob = linear_problem.LinearProblem(geom)
      true_reg_ot_cost.append(solver(prob).reg_ot_cost)

    x = jnp.vstack((x1, x2))
    y = jnp.vstack((y1, y2))

    segmented_reg_ot_cost = segment_sinkhorn.segment_sinkhorn(
        x,
        y,
        num_segments=2,
        max_measure_size=5,
        cost_fn=b_cost,
        num_per_segment_x=num_per_segment_x,
        num_per_segment_y=num_per_segment_y,
        sinkhorn_kwargs={"lse_mode": True},
        epsilon=0.1,
    )
    np.testing.assert_allclose(
        segmented_reg_ot_cost, true_reg_ot_cost, atol=1e-7, rtol=1e-7
    )
